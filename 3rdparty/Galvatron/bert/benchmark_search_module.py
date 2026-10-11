"""专门计时 Galvatron 策略搜索模块(论文创新点:决策树分解 + DP 搜索,§4.3)

隔离一切无关开销(profile 读取、代价模型参数构建只做一次,不计时),
只测搜索引擎本身,并给出两级粒度:

  每轮(round)= 搜索脚本里 for bsz 循环的一次 dp_on_model.fit(bsz)
    └─ 轮内按 pp_deg 拆分:每个 pp_deg 一次 _build_dp_and_run
         ├─ 代价估计构建(memcost/timecost/inter 矩阵)  → 毫秒级
         └─ DPAlg.fit(DP 求解内核)                    → 秒级(论文 Fig 4 的主体)

用法:
  python benchmark_search_module.py --layer_num 24 --memory_constraint 8 --bsz_start 8 --max_rounds 5
  python benchmark_search_module.py --layer_num 24 --memory_constraint 8 --bsz 32 --breakdown
输出时间单位自动适配(ms / s)。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np
from utils import (MemoryCostModel, TimeCostModel_with_overlap, DpOnModel,
                   read_allreduce_bandwidth_config, read_json_config)
from utils.dp_utils import DPAlg


def fmt(t):
    return "%10.2f ms" % (t * 1000) if t < 1 else "%10.3f s " % t


def build_search_env(layer_num, max_mem_G, stype):
    """一次性准备好搜索所需的一切(不计入时延)"""
    comm_coe_dict = read_allreduce_bandwidth_config(
        "../env_configs/allreduce_bandwidth_8_gpus.json", gpu_num=8)
    overlap_coe = read_json_config("../env_configs/overlap_coefficient.json")["overlap_coe"]
    fwd_time = read_json_config("./configs/forward_profiling_config.json")["fwd_time_hidden_1280"]

    if stype == "full":
        strategies = [[1,1,8,{'fsdp':0}],[1,1,8,{'fsdp':1}],
                      [1,2,4,{'tp':0,'fsdp':0}],[1,2,4,{'tp':1,'fsdp':0}],
                      [1,2,4,{'tp':0,'fsdp':1}],[1,2,4,{'tp':1,'fsdp':1}],
                      [1,4,2,{'tp':0,'fsdp':0}],[1,4,2,{'tp':1,'fsdp':0}],
                      [1,4,2,{'tp':0,'fsdp':1}],[1,4,2,{'tp':1,'fsdp':1}],
                      [1,8,1,{}],
                      [2,1,4,{'fsdp':0}],[2,1,4,{'fsdp':1}],
                      [2,2,2,{'tp':0,'fsdp':0}],[2,2,2,{'tp':1,'fsdp':0}],
                      [2,2,2,{'tp':0,'fsdp':1}],[2,2,2,{'tp':1,'fsdp':1}],
                      [2,4,1,{}],
                      [4,1,2,{'fsdp':0}],[4,1,2,{'fsdp':1}],
                      [4,2,1,{}],
                      [8,1,1,{}]]
    elif stype == "dp+tp":
        strategies = [[1,1,8,{'fsdp':0}],
                      [1,2,4,{'tp':0,'fsdp':0}],[1,2,4,{'tp':1,'fsdp':0}],
                      [1,4,2,{'tp':0,'fsdp':0}],[1,4,2,{'tp':1,'fsdp':0}],
                      [1,8,1,{}]]
    else:  # dp+pp
        strategies = [[1,1,8,{'fsdp':0}],
                      [2,1,4,{'fsdp':0}],
                      [4,1,2,{'fsdp':0}],
                      [8,1,1,{}]]

    def optimal_chunk_func(local_bsz, strategy):
        if strategy[0] == 1:
            return 1
        if local_bsz < 6:
            return local_bsz / 2
        return local_bsz / 8 + 2

    memcost_args = {"parameter_size": 76.98,
                    "tp_activation_per_bsz_dict": {1: 94.07, 2: 52.475, 4: 31.754375, 8: 21.61171875},
                    "other_model_states": 790, "other_activation_per_bsz": 300}
    timecost_args = {"parameter_size": 76.98, "microbatch": True,
                     "optimal_chunk_func": optimal_chunk_func,
                     "sequence_length": 512, "hidden_size": 1280,
                     "forward_computation_time": fwd_time,
                     "bct_fct_coe": 2, "extra_overhead": 0,
                     "comm_coe_dict": comm_coe_dict,
                     "dp_overlap_coe": overlap_coe, "bct_overlap_coe": overlap_coe}

    dp_on_model = DpOnModel(strategies, MemoryCostModel, TimeCostModel_with_overlap,
                            memcost_args, timecost_args,
                            max_mem=max_mem_G * 1024, layer_num=layer_num,
                            search_history={}, comm_coe_dict=comm_coe_dict)
    return dp_on_model, strategies, memcost_args, timecost_args, comm_coe_dict


def time_one_round(dp_on_model, bsz, per_pp_timing):
    """计时一轮 dp_on_model.fit(bsz),并可选按 pp_deg 拆分"""
    orig = dp_on_model._build_dp_and_run
    timings = []

    def wrapped(pp_deg, b):
        t0 = time.perf_counter()
        r = orig(pp_deg, b)
        timings.append((pp_deg, time.perf_counter() - t0))
        return r

    if per_pp_timing:
        dp_on_model._build_dp_and_run = wrapped
    t0 = time.perf_counter()
    result = dp_on_model.fit(bsz, print_=False)
    total = time.perf_counter() - t0
    dp_on_model._build_dp_and_run = orig
    return total, timings, result  # result = (cost, res_list, pp_deg, mem_remain, mem_cost)


def breakdown_single(dp_on_model, strategies, memcost_args, timecost_args,
                     comm_coe_dict, bsz, layer_num, max_mem):
    """单轮细拆:代价估计构建 vs DPAlg 求解内核 (逻辑忠实复刻 _build_dp_and_run)"""
    print("== 单轮细拆 (bsz=%d, 每个 pp_deg 各一行) ==" % bsz)
    rows = []
    for pp_deg in [8, 4, 2, 1]:
        strategy_set = [s for s in strategies if s[0] == pp_deg]
        n_st = len(strategy_set)
        layer_num_pp = layer_num // pp_deg

        t0 = time.perf_counter()
        intra = [TimeCostModel_with_overlap(s, bsz, **timecost_args).gen_result() for s in strategy_set]
        mem_list = [MemoryCostModel(s, bsz, **memcost_args).get_memory_cost() for s in strategy_set]
        other_mem = int(np.ceil(mem_list[0]["other"]))
        v = np.ceil(np.array([c["enc_total"] for c in mem_list])).astype(np.int32)
        v = v.reshape(1, -1).repeat(layer_num_pp, axis=0)
        intra_cost = np.array(intra, dtype=np.float64).reshape(1, -1).repeat(layer_num_pp, axis=0)
        t_est = time.perf_counter() - t0

        t0 = time.perf_counter()
        inter = np.zeros((n_st, n_st))
        for i in range(n_st):
            for j in range(n_st):
                sj, si = strategy_set[j], strategy_set[i]
                case1 = sj[1] > si[1]
                case2 = case3 = False
                if "tp" in sj[-1] and "tp" in si[-1]:
                    case2 = (sj[1] == si[1] and sj[-1]["tp"] != si[-1]["tp"])
                    case3 = (si[1] * si[2] == 8 and si[1] == 4 and sj[1] == 2
                             and sj[-1]["tp"] != si[-1]["tp"])
                if case1 or case2 or case3:
                    act = 2 * bsz / sj[2]
                    inter[i, j] = (sj[1] - 1) * act / sj[1]
                tp_s, dp_s = sj[1], sj[2]
                if tp_s == 1 or dp_s == 1:
                    coe = comm_coe_dict[pp_deg]["%d" % tp_s]
                else:
                    coe = comm_coe_dict[pp_deg]["%d_%d" % (tp_s, sj[-1]["tp"])]
                inter[i, j] *= coe * 0.001
                # fsdp 排序微偏置(原代码 L230-236)
                if i != j and np.array_equal(si[:3], sj[:3]):
                    c1 = "tp" not in si[-1] and si[-1]["fsdp"] != sj[-1]["fsdp"]
                    c2 = "tp" in si[-1] and si[-1]["tp"] == sj[-1]["tp"] and si[-1]["fsdp"] != sj[-1]["fsdp"]
                    if (c1 or c2) and si[-1]["fsdp"]:
                        inter[i, j] = 1e-4
        inter = np.expand_dims(inter, 0).repeat(layer_num_pp, axis=0)
        inter[0, :, :] = 0
        t_inter = time.perf_counter() - t0

        if max_mem - other_mem <= 0:  # 原代码 dp_utils.py:241 的 OOM 短路
            print("  pp_deg=%d (%2d 策略/层, %3d 层): other_mem=%dMB > 预算 → 不可行(短路)"
                  % (pp_deg, n_st, layer_num_pp, other_mem))
            rows.append({"pp_deg": pp_deg, "n_strategies": n_st, "layers": layer_num_pp,
                         "infeasible": True, "other_mem_MB": other_mem})
            continue

        dp = DPAlg(max_mem - other_mem, layer_num_pp, n_st)
        dp.set_v_and_cost(v, intra_cost, inter)
        t0 = time.perf_counter()
        dp.fit()
        t_dp = time.perf_counter() - t0

        print("  pp_deg=%d (%2d 策略/层, %3d 层): 代价估计构建 %s (含inter矩阵 %s) | DPAlg.fit %s"
              % (pp_deg, n_st, layer_num_pp, fmt(t_est + t_inter), fmt(t_inter), fmt(t_dp)))
        rows.append({"pp_deg": pp_deg, "n_strategies": n_st, "layers": layer_num_pp,
                     "t_cost_model_s": round(t_est, 6), "t_inter_matrix_s": round(t_inter, 6),
                     "t_dp_fit_s": round(t_dp, 6), "dp_mem_MB": int(max_mem - other_mem)})

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_num", type=int, default=24)
    parser.add_argument("--memory_constraint", type=int, default=8, help="内存预算 (GB)")
    parser.add_argument("--type", type=str, default="full", choices=["full", "dp+tp", "dp+pp"])
    parser.add_argument("--bsz_start", type=int, default=8, help="循环模式起始 batch size")
    parser.add_argument("--bsz_scale", type=int, default=8)
    parser.add_argument("--max_rounds", type=int, default=5, help="最多计时多少轮")
    parser.add_argument("--bsz", type=int, default=32, help="breakdown 模式的 batch size")
    parser.add_argument("--breakdown", action="store_true", help="只做单轮细拆")
    args = parser.parse_args()

    dp_on_model, strategies, mc, tc, coe = build_search_env(
        args.layer_num, args.memory_constraint, args.type)
    print("[env] layers=%d, mem=%dG, type=%s, 策略总数=%d (准备完成,不计入时延)\n"
          % (args.layer_num, args.memory_constraint, args.type, len(strategies)))

    results = {"env": {"layer_num": args.layer_num, "memory_constraint_G": args.memory_constraint,
                       "type": args.type, "n_strategies": len(strategies)}}

    if args.breakdown:
        results["breakdown_bsz"] = args.bsz
        results["breakdown"] = breakdown_single(dp_on_model, strategies, mc, tc, coe, args.bsz,
                                                args.layer_num, args.memory_constraint * 1024)
    else:
        print("== 每轮策略搜索时延 (round = 一次 DpOnModel.fit(bsz), 轮内按 pp_deg 拆分) ==")
        print("%4s %6s | %12s | %s" % ("轮次", "bsz", "整轮时延", "pp_deg 拆分 (pp: 时延)"))
        bsz = args.bsz_start
        rounds = []
        for r in range(args.max_rounds):
            total, timings, (cost, res_list, best_pp, mem_rem, mem_cost) = \
                time_one_round(dp_on_model, bsz, per_pp_timing=True)
            detail = ", ".join("%d: %s" % (p, fmt(t)) for p, t in timings)
            status = "可行 best_pp=%d cost=%.3f" % (best_pp, cost) if best_pp > 0 else "不可行(全 OOM)"
            print("%4d %6d | %s | %s\n        -> %s" % (r + 1, bsz, fmt(total), detail, status))
            rounds.append({"round": r + 1, "bsz": bsz, "total_s": round(total, 3),
                           "per_pp_s": {str(p): round(t, 5) for p, t in timings},
                           "best_pp": int(best_pp), "cost": float(cost) if best_pp > 0 else None})
            if best_pp == -1:
                break
            bsz += args.bsz_scale
        results["rounds"] = rounds

    out_path = os.path.join(HERE, "search_module_benchmark_results_%s.json"
                            % ("breakdown" if args.breakdown else "rounds"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to %s" % out_path)


if __name__ == "__main__":
    main()
