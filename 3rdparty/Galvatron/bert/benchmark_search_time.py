"""Galvatron 自动搜索计时基准(复现论文 Figure 4)

两种模式:
1. 端到端模式(默认,与论文一致):子进程运行原始 search_layerwise_hp_8gpus.py,计墙钟时间
   - layer_sweep: types {full, dp+tp, dp+pp} x layers {24,48,96} @ mem 8G   (对应 Fig 4b)
   - mem_sweep:   type full x layers 48 x mems {4,16}G                      (对应 Fig 4a)
   - single:      自指定一组参数
2. DP 内核模式(--dp-only):进程内单独计时一次 DpOnModel.fit(bsz),看纯 DP 的 O(L*M*S^2) 扩展

用法示例:
   python benchmark_search_time.py --modes layer_sweep mem_sweep
   python benchmark_search_time.py --modes single --single_args 32 8 full
   python benchmark_search_time.py --dp-only --dp_layer_nums 24 48 96 --dp_bsz 32
结果写入 ./search_time_benchmark_results.json,单次日志在 /tmp/galvatron_search_bench/
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = "/tmp/galvatron_search_bench"
RESULT_PATH = os.path.join(HERE, "search_time_benchmark_results.json")


def run_search_subprocess(layer_num, mem, stype):
    """忠实计时:墙钟跑一遍论文原始搜索脚本"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "search_%dL_%dG_%s.log" % (layer_num, mem, stype))
    cmd = [sys.executable, "search_layerwise_hp_8gpus.py",
           "--hidden_size", "1280",
           "--layer_num", str(layer_num),
           "--memory_constraint", str(mem),
           "--type", stype]
    t0 = time.time()
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    elapsed = time.time() - t0
    # 从日志解析最优吞吐(最优 bsz 那行)
    throughput, optimal_bsz = None, None
    with open(log_path) as f:
        for line in f:
            if line.startswith("Optimal bsz ="):
                optimal_bsz = int(line.split("=")[1].split()[0])
                throughput = float(line.split("Max throughput=")[1].split()[0])
    return {"layer_num": layer_num, "memory_constraint_G": mem, "type": stype,
            "wall_time_s": round(elapsed, 2), "exit_code": p.returncode,
            "optimal_bsz": optimal_bsz, "predicted_throughput": throughput,
            "log": log_path}


def dp_only_benchmark(layer_nums, bsz, max_mem_G=8):
    """进程内单独计时一次 DP fit(纯搜索内核,不含 bsz 循环)"""
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, ".."))
    from utils import (MemoryCostModel, TimeCostModel_with_overlap, DpOnModel,
                       read_allreduce_bandwidth_config, read_json_config)

    comm_coe_dict = read_allreduce_bandwidth_config(
        "../env_configs/allreduce_bandwidth_8_gpus.json", gpu_num=8)
    overlap_coe = read_json_config("../env_configs/overlap_coefficient.json")["overlap_coe"]
    fwd_time = read_json_config("./configs/forward_profiling_config.json")["fwd_time_hidden_1280"]

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

    results = []
    for layer_num in layer_nums:
        dp_on_model = DpOnModel(strategies, MemoryCostModel, TimeCostModel_with_overlap,
                                memcost_args, timecost_args,
                                max_mem=max_mem_G * 1024, layer_num=layer_num,
                                comm_coe_dict=comm_coe_dict)
        t0 = time.time()
        min_cost, res_list, pp_deg, mem_remain, mem_cost = dp_on_model.fit(bsz, print_=False)
        elapsed = time.time() - t0
        results.append({"layer_num": layer_num, "bsz": bsz, "dp_fit_time_s": round(elapsed, 3),
                        "feasible": pp_deg > 0})
        print("  layers=%3d  DpOnModel.fit(bsz=%d) = %8.3f s  feasible=%s"
              % (layer_num, bsz, elapsed, pp_deg > 0))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=[],
                        choices=["layer_sweep", "mem_sweep", "single"],
                        help="要运行的基准组合")
    parser.add_argument("--single_args", nargs=3, type=str, metavar=("LAYERS", "MEM_G", "TYPE"),
                        help="single 模式参数,如: 32 8 full")
    parser.add_argument("--dp-only", action="store_true", help="只跑进程内 DP 内核计时")
    parser.add_argument("--dp_layer_nums", nargs="+", type=int, default=[24, 48, 96])
    parser.add_argument("--dp_bsz", type=int, default=32)
    args = parser.parse_args()

    all_results = {"end_to_end": [], "dp_only": []}

    if args.dp_only:
        print("== DP kernel scaling (single DpOnModel.fit, no bsz loop) ==")
        all_results["dp_only"] = dp_only_benchmark(args.dp_layer_nums, args.dp_bsz)

    jobs = []
    if "layer_sweep" in args.modes:
        for stype in ["full", "dp+tp", "dp+pp"]:
            for layer_num in [24, 48, 96]:
                jobs.append((layer_num, 8, stype))
    if "mem_sweep" in args.modes:
        for mem in [4, 16]:
            jobs.append((48, mem, "full"))
    if "single" in args.modes:
        l, m, t = args.single_args
        jobs.append((int(l), int(m), t))

    if jobs:
        print("== End-to-end search wall time (paper's original script) ==")
        for i, (layer_num, mem, stype) in enumerate(jobs):
            print("[%d/%d] layers=%d mem=%dG type=%s ..."
                  % (i + 1, len(jobs), layer_num, mem, stype), flush=True)
            r = run_search_subprocess(layer_num, mem, stype)
            all_results["end_to_end"].append(r)
            print("    -> %8.2f s (exit %d, opt_bsz %s, pred_tput %s samples/s)"
                  % (r["wall_time_s"], r["exit_code"], r["optimal_bsz"],
                     r["predicted_throughput"]), flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to %s" % RESULT_PATH)


if __name__ == "__main__":
    main()
