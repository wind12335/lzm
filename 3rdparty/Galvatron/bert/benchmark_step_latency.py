"""Galvatron 版 HDP 时延基准:对照 Google HDP 论文(ICLR'18)监测的数值逐一抓取。

Google HDP 论文监测的数值(§2/§3 "Measuring Reward"、Table 1、§Overhead):
  1. r_d     : 一个训练步(1 forward + 1 backward + 1 参数更新)的秒数;
               协议 = 跑 10 步,丢前 5 步 warmup,取后 5 步中位数;
  2. reward  : -sqrt(r_d);非法策略(OOM)记 -10(OOM 本身是被监测的失败信号);
  3. Runtime Reduction % : (最优先前方案 r_d - 本方案 r_d) / 最优先前方案 r_d;
  4. 策略搜索开销:policy 训练 1000 迭代,最多 ~3 小时(Fig 3);
  5. Table 2: 搜索方差(10 次试验的 best/median/worst)。

Galvatron 对应模块的抓取方式:
  1-2. train_hp_hdp_measure.py 子进程跑真实训练,逐 rank 输出 HDP_RESULT_JSON;
  3.   同协议测 Galvatron 自动搜索策略 vs 手工均匀基线(纯 PP / 纯 TP / 纯 DP);
  4.   DP 搜索墙钟(读 search_time_benchmark_results.json,若存在);
  5.   Galvatron 搜索是确定性 DP,无试验方差 —— 单步时延的 min/max 给出测量噪声下界。

用法(在 bert/ 下):
  python benchmark_step_latency.py                      # 默认: galvatron_8G pp8 tp8 dp8
  python benchmark_step_latency.py --runs galvatron_8G pp8
  python benchmark_step_latency.py --warmup 10 --measure 10   # 放宽协议
结果打印 Table-1 风格对比表,并存 ./benchmark_step_latency_results.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(HERE, "benchmark_step_latency_results.json")

# ---- 放置方案注册表(全部 8×TitanV, BERT-Huge-32: hidden 1280 / 32 层 / seq 512)----
# json 模式  : galvatron 搜索出的策略配置(--galvatron_config_path)
# global 模式: 手工均匀策略(对应 HDP 的 human-expert / GPU-only / CPU-only 基线)
# nproc 规则 : 8 // pp_deg(repo 约定,进程内每 stage 一卡,见 hybrid_parallel_model.py:148)
PLACEMENTS = {
    "galvatron_8G": {
        "desc": "Galvatron 自动搜索(8G 预算, pp4+混合tp/dp)",
        "mode": "json", "config": "./configs/galvatron_config_8gpus_1280hidden_32layers_8G_full.json",
        "pred_tput": 8.53,  # 搜索时代价模型给出的预测吞吐 samples/s
    },
    "galvatron_10G": {
        "desc": "Galvatron 自动搜索(10G 预算, pp4+dp)",
        "mode": "json", "config": "./configs/galvatron_config_8gpus_1280hidden_32layers_10G_full.json",
        "pred_tput": None,
    },
    "galvatron_12G": {
        "desc": "Galvatron 自动搜索(12G 预算)——12GB 卡装不下,预期 OOM→-10",
        "mode": "json", "config": "./configs/galvatron_config_8gpus_1280hidden_32layers_12G_full.json",
        "pred_tput": 12.08,  # 搜索预测吞吐 @bsz16(需 24GB 卡才能真正跑)
    },
    "pp8": {
        "desc": "纯流水线(8 stage 各 1 卡)≈ HDP human-expert",
        "mode": "global", "pp_deg": 8, "tp": 1, "fsdp": 0, "bsz": 8, "chunks": -1,
    },
    "tp8": {
        "desc": "纯张量并行(Megatron 风格)≈ HDP 手工切分",
        "mode": "global", "pp_deg": 1, "tp": 8, "fsdp": 0, "bsz": 8, "chunks": -1,
    },
    "dp8": {
        "desc": "纯数据并行(每卡全模型)≈ HDP GPU-only(预期 OOM→reward -10)",
        "mode": "global", "pp_deg": 1, "tp": 1, "fsdp": 0, "bsz": 8, "chunks": -1,
    },
}

MODEL_ARGS = ["--vocab_size", "30522", "--hidden_size", "1280",
              "--num_hidden_layers", "32", "--num_attention_heads", "16",
              "--seq_length", "512", "--epochs", "5", "--lr", "1e-4",
              "--adam_weight_decay", "0.01", "--dropout_prob", "0.1",
              "--check_loss", "0", "--profile", "0", "--apply_strategy", "0"]


def read_pp_deg_from_config(config_path):
    with open(os.path.join(HERE, config_path)) as f:
        return json.load(f)["pp_deg"]


def run_placement(name, spec, warmup, measure, port, timeout):
    pp_deg = (read_pp_deg_from_config(spec["config"]) if spec["mode"] == "json" else spec["pp_deg"])
    nproc = 8 // pp_deg
    cmd = [sys.executable, "-m", "torch.distributed.launch",
           "--nproc_per_node=%d" % nproc, "--master_port", str(port),
           "train_hp_hdp_measure.py"] + MODEL_ARGS + \
          ["--warmup_steps", str(warmup), "--measure_steps", str(measure)]
    if spec["mode"] == "json":
        cmd += ["--galvatron_config_path", spec["config"]]
    else:
        cmd += ["--pp_deg", str(spec["pp_deg"]), "--global_tp_deg", str(spec["tp"]),
                "--global_tp_consec", "1", "--chunks", str(spec["chunks"]),
                "--fsdp", str(spec["fsdp"]), "--global_train_batch_size", str(spec["bsz"])]

    print("[%s] nproc=%d (pp_deg=%d) 启动: %s" % (name, nproc, pp_deg, spec["desc"]), flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=HERE, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.stdout.decode("utf-8", errors="replace")
        exit_code = p.returncode
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or b"").decode("utf-8", errors="replace")
        exit_code = -99
    wall = round(time.time() - t0, 1)

    ranks = []
    for line in output.splitlines():
        if line.startswith("HDP_RESULT_JSON "):
            ranks.append(json.loads(line[len("HDP_RESULT_JSON "):]))

    oom = "CUDA out of memory" in output or "OOM" in output
    if ranks and not oom:
        # r_d 取各 rank 中位数的最大值(同步执行,最慢 rank 决定一步时延)
        r_d = max(r["r_d_median_s"] for r in ranks)
        peak_mem = max(r["peak_mem_allocated_MB"] for r in ranks)
        reward = -r_d ** 0.5
        status = "ok"
    else:
        r_d, peak_mem, reward, status = None, None, -10.0, "OOM/invalid"

    rec = {"name": name, "desc": spec["desc"], "nproc": nproc, "exit_code": exit_code,
           "wall_s": wall, "r_d_s": r_d, "reward": round(reward, 4), "status": status,
           "peak_mem_MB": peak_mem,
           "per_rank": [{k: r[k] for k in ("rank", "r_d_median_s", "r_d_min_s", "r_d_max_s",
                                           "peak_mem_allocated_MB")} for r in ranks],
           "pred_tput": spec.get("pred_tput")}
    # HDP 没有代价模型;Galvatron 特有:预测 vs 实测误差(论文宣称 <5%)
    if r_d is not None and spec.get("pred_tput"):
        bsz = read_bsz(spec)
        pred_r_d = bsz / spec["pred_tput"]
        rec["pred_r_d_s"] = round(pred_r_d, 4)
        rec["pred_error"] = round((r_d - pred_r_d) / r_d, 4)
    print("    -> %s" % json.dumps({k: rec[k] for k in
          ("r_d_s", "reward", "status", "peak_mem_MB", "wall_s") if k in rec}), flush=True)
    return rec


def read_bsz(spec):
    if spec["mode"] == "json":
        with open(os.path.join(HERE, spec["config"])) as f:
            return json.load(f)["global_bsz"]
    return spec["bsz"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["galvatron_8G", "pp8", "tp8", "dp8"],
                        choices=sorted(PLACEMENTS), help="要测的放置方案")
    parser.add_argument("--warmup", type=int, default=5, help="HDP 协议 warmup 步数")
    parser.add_argument("--measure", type=int, default=5, help="HDP 协议计时步数(取中位数)")
    parser.add_argument("--timeout", type=int, default=1800, help="单方案超时(秒)")
    args = parser.parse_args()

    records = []
    for i, name in enumerate(args.runs):
        records.append(run_placement(name, PLACEMENTS[name], args.warmup,
                                     args.measure, 9950 + i, args.timeout))

    # ---- 汇总:HDP Table 1 风格 ----
    print("\n== HDP 风格对比表 (BERT-Huge-32, 8xTitanV, %d warmup + %d 步取中位) ==" % (args.warmup, args.measure))
    print("%-14s %-38s %10s %8s %10s  %s" % ("方案", "说明", "r_d(s)", "reward", "peakMem(MB)", "状态"))
    for r in records:
        print("%-14s %-38s %10s %8s %10s  %s" % (
            r["name"], r["desc"],
            "%.3f" % r["r_d_s"] if r["r_d_s"] else "OOM",
            "%.3f" % r["reward"] if r["status"] == "ok" else "-10",
            "%.0f" % r["peak_mem_MB"] if r["peak_mem_MB"] else "-",
            r["status"]))

    # Runtime Reduction %:基线 = 非自动搜索方案中最快者(HDP 定义:相对最优先前方案)
    baselines = [r for r in records if not r["name"].startswith("galvatron") and r["r_d_s"]]
    galva = [r for r in records if r["name"].startswith("galvatron") and r["r_d_s"]]
    if baselines and galva:
        best_prior = min(b["r_d_s"] for b in baselines)
        for g in galva:
            red = (best_prior - g["r_d_s"]) / best_prior * 100
            g["runtime_reduction_pct"] = round(red, 1)
            print("\nRuntime Reduction (%s vs 最优手工基线 %.3fs): %.1f%%  [HDP 论文 NMT-2layer 为 60.6%%]"
                  % (g["name"], best_prior, red))

    # 搜索开销对照(HDP: RL policy 1000 迭代 ~3h;Galvatron: 确定性 DP,分钟级)
    search_file = os.path.join(HERE, "search_time_benchmark_results.json")
    if os.path.exists(search_file):
        with open(search_file) as f:
            s = json.load(f)
        full_runs = [e for e in s.get("end_to_end", []) if e.get("type") == "full"]
        if full_runs:
            best = min(full_runs, key=lambda e: e["wall_time_s"])
            print("\n策略搜索开销: Galvatron DP 搜索最短 %.1f s (%d 层@%dG) vs HDP RL policy ~3 h (1000 迭代)"
                  % (best["wall_time_s"], best["layer_num"], best["memory_constraint_G"]))

    with open(RESULT_PATH, "w") as f:
        json.dump({"protocol": "HDP ICLR'18: 10 steps, drop 5 warmup, median of 5",
                   "warmup": args.warmup, "measure": args.measure, "records": records},
                  f, indent=2, ensure_ascii=False)
    print("\nResults saved to %s" % RESULT_PATH)


if __name__ == "__main__":
    main()
