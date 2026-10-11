# lzm 仓库详细解析与 Titan V 复现

> 源仓库：`/data/coding/lzm`（原"测试hierarchical和transAware"，含 git 历史，248MB）
> 本机：8× Titan V，Python 3.8.20，galvatron conda env（torch 1.10.2+cu113）+ tf conda env（py3.8，tensorflow-cpu 2.13.1）
> 复现日期：2026-10-11，结果目录 `/data/coding/lzm/results_titanv_20261011/`

---

## 一、仓库里到底有哪几个方法

```
/data/coding/lzm/
├── PaperCode/
│   ├── TransAware/
│   │   ├── transaware/            # 方法1 TensorFlow 原版（datasets 在这里：NMT/GNMT/NasNet 的 pkl）
│   │   ├── transaware_pytorch/    # ★ 方法1 PyTorch 移植版（本机跑的就是它）
│   │   └── Hierarchical/Trinity/  # ★ hierarchical 基线（TensorFlow，TF1 风格）
│   └── FasterTransAware/          # 方法2（C++/FasterTransformer；build/、3rdparty/ 未入仓，本机不可编译）
├── results_local_20261011/        # 海光 DCU 机器（py3.11/torch2.7）的复现日志
├── results_titanv_20261011/       # 本机 8×Titan V 全部实验日志（统一实验原始记录）
├── docs/                          # 全部文档：方法讲解 / 执行手册 / 本文件 / 迁移契约 / 云端原始结果
├── hsdag_unified.py               # HSDAG 适配器归档副本（原件在 /data/coding/HSDAG/）
├── run_search_time_comparison.sh  # 四方法统一搜索耗时一键脚本
└── README.md                      # 数据集与实验入口索引
```

- **hierarchical 基线 = Trinity**（`Hierarchical/Trinity/`）。论文语境里它是与 Google HDP 同族的"层级式 RL 放置"基线。
- **自研方法1 = TransAware**（算子→GPU 放置的 RL 搜索：Graphormer 图编码 + LSTM 策略 + PPO）。
- **自研方法2 = FasterTransAware**（C++ 算子融合加速 FasterTransformer），未入仓部分见 §5，本机无法复现，不纳入本次三方对比。

---

## 二、hierarchical（Trinity）解析

### 2.1 代码链路与运行方式

```
Trinity_main_test.py                     # 入口（计时包住 schedule_graph）
  ├─ TrinityControllerTest._buildCluster # 用 TF grappler 伪造虚拟 GPU 集群
  │    device_properties: 'GeForce GTX TITAN X', 12GB, bandwidth=336MB/s …（写死的 TITAN X 属性）
  ├─ readfile.ReadPickleFile             # 读 pkl（TF MetaGraphDef + StepStats）
  ├─ graph_scheduling_base.GraphSchedulingBase   # --is-base True → hierarchical 基线主体 ★
  └─ graph_scheduling.GraphScheduling            # --is-base False → aware 基线（TransAware 的对照）
```

- `hierarchical_test.sh`：`--is-base True`（走 GraphSchedulingBase），`--is-sim True`（用模拟器评估），
  `--n-devs 2`，`--all-steps 140`，数据集 GNMT。
- `aware_test_nmt.sh`：`--is-base False`（走 GraphScheduling），NMT 数据集，n-devs 4。
- 依赖链：`Trinity_main_test → graph_scheduling* → Trinity_program_norml → Trinity.ge(Node2Vec/DeepWalk 图嵌入) → pandas/gensim/fastdtw`。
- **不需要真 GPU**：`--is-sim True` + grappler 虚拟集群，`tensorflow-cpu` 即可。

### 2.2 它监控/上报的数值

每轮迭代打印（`graph_scheduling_base.compute_rewards`）：

| 打印项 | 含义 |
|---|---|
| `模拟环境（执行时间）` | 模拟器估的 step 执行时间（秒） |
| `本次调度时间` | 本轮 placement 的调度时间（sim 模式下经常是 ~1e-7 的退化值） |
| `本次调度巅峰内存` | 模拟峰值显存（GB） |
| `本次调度总通信` | 模拟总通信量（GB） |
| `最佳调度时间` | 历史最优 |

结束打印 `【Trinity 策略搜索统计】总耗时`（= 包住 `schedule_graph` 的 wall 秒数），
并写 `output/<model>/<n_devs>/all/output_final.json`。

### 2.3 已有参照数字

| 来源 | NMT_4layers | GNMT | 说明 |
|---|---|---|---|
| 云端 2025-12（TF 版） | 230.52s | 237.98s | 脚本上报的策略搜索执行时间 |
| K100 双卡 2026-04 | — | 256.67s (n4) / 166.70s (n2) | ROCm 机器 |
| 海光 DCU 2026-10-11 | **未跑通** | **未跑通** | import 阶段缺 pandas 失败（hierarchical_GNMT.log） |

---

## 三、TransAware（PyTorch 移植版）解析

### 3.1 主链路

```
strategy_search.py（入口，StubUnpickler 屏蔽 pkl 内嵌 TF 对象，免装 TF）
  → Graph（节点 cost/mem/device，边 weight=num_bytes）
  → PyAwareInterface.startup_strategy()
    → ProgressivePlacer.search_policy()
       convert_input_data()  # alpha*cost+beta*mem 硬编码特征 + 边权"距离矩阵"（numpy）
       Graphormer 编码 (1, node_num, 512)
       × n_rounds 轮（默认 18，原版硬编码）：
          LSTM+PPO 前向 → 生成 device placement
          → 纯 Python 事件驱动模拟器评估 run_time / peak_mem
          → compute_rewards → PPO 反向更新
```

文件对应关系与"如实保留的退化行为"（浮点特征走 nn.Embedding 需 clamp、中心性度数≈node_num 需 clamp、
距离矩阵实为边权）见 `transaware_pytorch/README.md`——这些是原 TF 实现就有的怪癖，移植版保持一致。

### 3.2 模拟器语义（run_time 是怎么来的）

`simulator.py`：**HDP 家族的标准事件驱动模拟器**（与 Google HDP/DevicePlacement 论文同构）：

- 节点 `compute_cost` 来自 pkl 内嵌 TF StepStats 的微秒级画像（在别的机器上量出来的）；
- 通信：跨设备边 `output_memory / transfer_speed`，`transfer_speed=7600`；
- 每算子开销 `delta1=5.7µs`，总线开销 `delta2=25µs`；
- `AwareSimulator.simulate` 返回 `run_time/1e6`（µs→s）。

⚠️ **因此 run_time/peak_mem 是"虚拟设备模型上的估计值"，不是本机 Titan V 的实测值**；
placement 也从不真正执行。这一点在与 Galvatron 对比时必须写清口径。

### 3.3 已有参照数字（best run_time，模拟器口径）

| 来源 | NMT_4layers (n4) | GNMT (n4) | 搜索耗时 |
|---|---|---|---|
| 云端 TF 版 2025-12 | — | — | 139.51s / 155.92s |
| 海光 DCU 2026-10-11 | 0.2474s @ep17 / 6.18GB | 0.3473s @ep17 / 7.69GB | 341.0s / 394.7s |
| **本机 Titan V 2026-10-11** | **0.2457s @ep17 / 6.17GB** | **0.4191s @ep17 / 7.69GB** | **21.6s / 34.5s** |

GNMT 与 DCU 差 ~20% 属正常：README 注明"随机初始化、不设固定 seed，多次运行数值不同"。
NMT 与 DCU 几乎一致（0.2457 vs 0.2474）。

---

## 四、本机 Titan V 复现结果（2026-10-11）

### 4.1 TransAware（galvatron env：py3.8 + torch 1.10.2，CPU 执行搜索+模拟）

| 模型 | n_devs | best run_time (ep17) | peak_mem | 搜索耗时 | 整进程 wall | 日志 |
|---|---|---|---|---|---|---|
| NMT_4layers (1503 节点) | 4 | 0.2457 s | 6.17 GB | 21.55 s | 23.1 s | transaware_NMT_4layers_4devs.log |
| GNMT (1743 节点) | 4 | 0.4191 s | 7.69 GB | 34.47 s | 36.0 s | transaware_GNMT_4devs.log |
| NMT_4layers | 8 | 0.3501 s | 6.17 GB | 24.31 s | 25.9 s | transaware_NMT_8devs.log |
| GNMT | 8 | 0.2841 s | 7.69 GB | 37.27 s | ~39 s | transaware_GNMT_8devs.log |

- 两模型 18 轮均单调收敛、placement 全覆盖（1503/1503、1743/1743）✓
- placement 各轮保持同一划分、cross_edges 数与 DCU 一致（NMT n4: 406 条）→ 链路行为与 DCU 复现一致
- 本机 CPU（64 核）比 DCU 机快一个量级：搜索 21–37s vs DCU 341–395s

### 4.2 hierarchical（Trinity，tf env：py3.8 + tensorflow-cpu 2.13.1）

环境：`/data/miniconda/envs/tf`（tensorflow-cpu 2.13.1 + networkx/pandas/gensim/fastdtw + **joblib、tqdm**——后两个是指南未列的隐藏依赖）。

**兼容性补丁**（TF1 私有 API 在 TF 2.13 已移除，集中垫片在 `Trinity_main_test.py` import 区，
不改任何算法文件；DCU 上从未跑到这一步，属首次打通）：
- `tensorflow.python.framework.ops.IndexedSlices` → 从 `indexed_slices` 模块补回（4 个文件共 8 处引用）
- `tensorflow.python.ops.array_ops.stack` → 用 `tf.stack` 补回（ColorRL_program.py:832 一处）

| 模型 | n-devs | 搜索耗时（脚本上报） | 整进程 wall | 日志 |
|---|---|---|---|---|
| GNMT | 2（云端原口径） | **141.94 s** | 2m26s | hierarchical_GNMT_2devs.log |
| NMT_4layers | 2（云端原口径） | **138.30 s** | 2m22s | hierarchical_NMT_4layers_2devs.log |
| GNMT | 8 | **718.10 s** | 12m03s | hierarchical_GNMT_8devs.log |
| NMT_4layers | 8 | **705.97 s** | 11m51s | hierarchical_NMT_4layers_8devs.log |

- 输出工件：`output/<模型>/<n_devs>/all/output_{10..140}.json`、`output_final.json`、
  `*_ungrouped_pl.{json,pkl}`（节点→设备 placement）。
- 行为与 K100 日志特征一致：sim 模式下 `本次调度时间` 退化为 ~1e-8，有效指标是
  `模拟环境（执行时间）`（GNMT n2 约 0.010–0.012 s 量级）与通信/内存序列。
- **搜索耗时随设备数强增长**（GNMT：n2 142s → n8 718s，约 5×），而 TransAware 几乎不随 n_devs
  变化（NMT 21.6→24.3s，GNMT 34.5→37.3s，n4→n8）。

云端/本机对照（n-devs 2 口径）：

| 模型 | 云端 TF 2025-12 | K100 2026-04 | **本机 Titan V 2026-10-11** |
|---|---|---|---|
| NMT_4layers | 230.52 s | — | 138.30 s |
| GNMT | 237.98 s | 256.67 s (n4) / 166.70 s (n2) | 141.94 s (n2) / **275.00 s (n4)** / 718.10 s (n8) |

**搜索耗时随卡数增长的跨机器核对（GNMT）**：本机 n4 = 275.00s 与 K100 n4 = 256.67s
吻合（差 7%，RL 随机性范围）；4→8 卡 = 275→718s（2.6×，幂指数 ~1.2–1.4）。机制：
两级 RL 的调度级动作空间、事件驱动模拟的设备队列/跨卡通信、PPO 张量规模均随设备数扩大，
且与图规模基本无关（BERT_Huge 35 节点 @n8 = 679.7s ≈ GNMT 1743 节点 @n8 = 718.1s）。
同口径 n2 下本机（141.9s）< K100（166.7s）< 云端（238.0s），排除机器性能因素。
日志：`results_titanv_20261011/hierarchical_GNMT_4devs_check.log`。

---

## 五、FasterTransAware（方法2）为何不在本次对比内

- `build/`（1.7G 编译产物）与 `3rdparty/`（273M，含 Megatron-LM/FasterTransformer 子模块）未入仓；
- 需 cmake 重编译 + 按 `.gitmodules` 拉子模块（经 socks 代理拉 NVIDIA 仓库可行性未知）；
- 属于"内核融合加速"维度，与 Galvatron 的"并行策略搜索"不同维度。
  结论：四方对比 = **Galvatron vs TransAware vs Hierarchical(Trinity) vs HSDAG**（HSDAG 见 §8.5）。

---

## 六、三方对比的口径设计（核心结论）

| 维度 | Galvatron | TransAware | Hierarchical(Trinity) |
|---|---|---|---|
| 搜索空间 | 22 种层级并行策略组合 (pp/tp/dp/fsdp) | 算子→GPU 放置（任意算子任意卡） | 算子→GPU 放置（层级式分组+调度） |
| 搜索算法 | 前向画像 + 成本模型 + DP（确定性） | Graphormer+LSTM+PPO（RL） | 层级 RL（Node2Vec 图嵌入+PPO） |
| 评估方式 | **真实 8×Titan V 训练实测**（HDP 协议计时） | **纯模拟器估计**（不执行） | **grappler 虚拟集群+模拟器估计**（不执行） |
| 搜索耗时（本机） | 41.6–212 s | 21.6–37.3 s | 待测 |
| 结果指标（本机） | r_d 实测步时延 0.88–2.23 s、峰值显存实测、OOM→−10 | run_time 模拟值 0.25–0.42 s | 模拟执行时间/内存/通信 |

**可比项**：
1. **搜索开销**：三方同口径（脚本上报的策略搜索 wall 时间）——直接可比；
2. **方法族定位**：TransAware/Trinity 与 Google HDP 同族（op-level placement，pkl 即 HDP 基准套件，
   模拟器同构），Galvatron 是 layer-level 并行策略搜索——**论文叙事：我们不做 op-level 放置，
   而是在真实硬件上端到端搜索可执行的混合并行方案**；
3. **诚实口径声明**：TransAware/Trinity 的 run_time 是虚拟设备模型上的估计（compute_cost 来自异机
   StepStats 画像、TITAN X 属性写死），与 Galvatron 的 Titan V 实测值**不可直接比大小**；
   对比表需分"模拟估计"与"实测"两栏。

---

## 七、复现命令（本机）

```bash
# ---- TransAware（galvatron env，无需 GPU）----
cd /data/coding/lzm/PaperCode/TransAware/transaware_pytorch
/data/miniconda/envs/galvatron/bin/python strategy_search.py --model_name NMT_4layers --n_devs 8 --debug_placement
/data/miniconda/envs/galvatron/bin/python strategy_search.py --model_name GNMT         --n_devs 8 --debug_placement
# n_devs=4 对齐云端/DCU 口径：--n_devs 4

# ---- hierarchical / Trinity（tf env）----
cd /data/coding/lzm/PaperCode/TransAware/Hierarchical/Trinity
# GNMT、n-devs 2（云端原口径）
/data/miniconda/envs/tf/bin/python Trinity_main_test.py \
    --pickled-inp-file "./datasets/GNMT/gnmt_baseline.pkl" \
    --virtual True --n-devs 2 --verbose True --reset-devs True \
    --is-sim True --is-all True --is-base True --all-steps 140 \
    --num-cpus 0 --output-dir "./output/GNMT/"
# NMT：--pickled-inp-file "./datasets/NMT_4layers/nmt_4layers_baseline.pkl"
# 8 卡口径：--n-devs 8
```

---

## 八、统一模型对比实验：BERT-Huge @ 8×Titan V（2026-10-11）★ 核心实验

> 📖 **可执行手册**：`/data/coding/lzm/docs/统一模型实验执行手册.md`——不熟悉环境也能照着逐步复现
> 全部实验（搜索耗时对比 / placement 真机执行 / 步时延实测 / 原始模型口径 / 故障排查）。

### 8.1 实验设计（同一张图、同一协议、真机执行）

**统一模型** = BERT-Huge（32 层, hidden 1280, seq 512, bsz 8），与 Galvatron 训练评测同款。

**统一计算图**（`galvatron-paper/bert/build_bert_graph_pkl.py` 构建，一个 pkl 喂两家）：
35 节点链 `embeddings → layer_0..31 → pooler → cls`，与 `placement_runner_bert.py` 的执行单元一一对应。

- `compute_cost`(µs)：按 FLOPs 比例分摊**本机实测** naive 均匀放置步时 1.8099s（HDP 协议中位）——
  layer 52841µs × 32, cls 109404µs, embeddings 5284µs, pooler 4403µs；
- `memory`(B)：混合精度 Adam 持久状态 16B/param，总 **10.60GB**（单卡 12GB 装不下 → 硬约束）；
- 边激活：bsz×seq×hidden×2B = 10.49MB；
- pkl 同时携带 `nx_graph`（TransAware 读）与 `Meta_graph_def`+`step_stats`+`op_perf`（Trinity 读），
  step_stats 设备名含 'all'、时长=真机 µs（过 `get_op_costs` 校验）。

**流程**：各方法在统一图上搜索 → 搜索出的 placement 翻译成 35 单元设备表 →
`placement_runner_bert.py` 真机执行（HDP 协议：5 warmup + 5 measure 取中位，OOM→−10）。

### 8.2 结果总表

| 方法 | 搜索耗时 | 自报最优（各自模拟口径） | **真机 r_d（HDP 协议）** | 判定 |
|---|---|---|---|---|
| **Galvatron 8G full** | **174.44 s**（32L 实测） | 预测吞吐 8.53 samples/s | **1.321 s**（7023MB 实测） | ✅ 唯一可行，较均匀放置 **−27.1%** |
| Galvatron pp8（手工纯流水） | — | — | 0.883 s（7127MB） | ✅ −51.2%（本机拓扑下最优） |
| Galvatron tp8 / dp8 / 12G | 41.6–212 s | — | 2.230 s / −10 / −10(OOM) | 参照 |
| **TransAware** | 1.17 s | 30.21 s @ep17（含 epoch 重缩放的构造性假象） | **−10（真机 OOM）** | ❌ 不可行 |
| **Hierarchical（Trinity）** | **679.74 s** | 1.827 s（×1e6 修正后，见 8.4） | **−10（真机 OOM）** | ❌ 不可行 |
| **HSDAG（NeurIPS'24，方案 A 移植）** | **129.38 s**（170 eps 重载配置） | 1.812 s（统一模拟器口径） | **−10（真机 OOM）** | ❌ 不可行 |
| 均匀放置 4L/GPU（无搜索基准） | 0 | — | 1.810 s | placement 族结构上限 |

日志：`results_titanv_20261011/{transaware_BERTHuge_8devs.log, hierarchical_BERTHuge_8devs.log,
trinity_BERTHuge_8devs_real_exec.log, hsdag_BERTHuge_8devs.log, hsdag_BERTHuge_8devs_eps170.log}`；
Galvatron 搜索 `/tmp/galvatron_search_bench/search_32L_8G_full.log`
（并入 `galvatron-paper/bert/search_time_benchmark_results.json`）。

Trinity 最终 placement（`output/BERT_Huge/8/all/output_140_ungrouped_pl.json`，第 140 步已收敛，
GPU 分布 {0:21, 1:2, 2:7, 3:1, 4:3, 6:1}）真机执行命令：

```bash
cd /data/coding/galvatron-paper/bert && export PATH=/data/miniconda/envs/galvatron/bin:$PATH
python placement_runner_bert.py --bsz 8 --placement \
  2,4,0,2,0,1,3,2,2,0,6,1,2,2,0,0,0,0,4,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# → RuntimeError: CUDA out of memory (GPU 0, 10.15 GiB allocated, 42 MiB free) ⇒ r_d = −10
```

### 8.3 两个 op 级放置基线为何在真机翻车

1. **模拟器对模型持久状态失明**：两家的峰值显存都只统计输出激活张量（HDP 模拟器传统），
   看不见 10.6GB 参数/梯度/优化器状态 → "最优解"就是把节点堆到少数卡上：
   TransAware 20 次搜索 19 次 **35/35 全塞一张卡**（GPU2，自报峰值仅 0.021GB 激活）；
   Trinity 收敛到 **GPU0=21/35 节点**（~7GB 状态+激活）。
2. **真机执行即 OOM**：TransAware 单卡 10.12GB 已分配、差 128MB；Trinity GPU0 10.15GB 已分配、
   申请 80MB 只剩 42MB——`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` 重试错误一字不差，
   排除碎片假阳性。按 HDP 协议记 −10。
3. **TransAware 还叠加"根本没学"**：ReinforceModel 优化器 lr=1e-8（忠实保留 TF 原版），
   策略网络权重训练前后≈不变，placement ≈ 随机初始化采样；1.17s 的"搜索耗时"实为空转。
4. **结构上限**：35 节点链无微批流水重叠、无 tp/fsdp 分片，即便完美均衡也 ≥1.81s——
   Trinity 模拟器自己收敛到 1.827s（≈链下界+通信），**搜索空间里根本不存在优于均匀放置的解**；
   Galvatron 靠层级并行分片（8G: dp+tp+pp 混合）突破该上限到 1.32s。

### 8.4 模拟器口径校准发现（诚实披露，均已在代码中定位）

- **Trinity 打印的"模拟环境（执行时间）"根本不是执行时间估计**：`graph_scheduling_base.py:226`
  打印的 `sim_time` = `eval_grouping_and_scheduling_sim` 里的 `slim_time = time.time()-start_time`
  ——**跑模拟器本身耗费的 Python 墙钟**（35 节点 ~1.8ms，GNMT 1743 节点 ~10ms，
  历史日志"模拟环境 0.010s"之谜就此解开）。
- **Trinity 内部 run_time 被二次 /1e6**：`tf_pl_rewarder.py:80` 的 `simulate` 已经 `return r/1e6`，
  `ColorRL_program.py:1026` 又除一次 → `self.run_time`/`best_run_time` 绝对值失真 1e6×；
  但除法是单调变换，RL 搜索排序不受影响。**修正 ×1e6 后 best = 1.827s**。
- **模拟器机制本身是准的**：用我们烘焙的真机成本做隔离验证——`ImportantOpsRewarder`
  全放 GPU0 → 1.810s ≈ 真机实测 1.8099s ✓；5 卡均分 → 1.817s；交错 → 1.818s。
  即"引擎正确、输入有盲区"：喂给它真机成本它就吐真机级估计，坏在 peak_mem 不含模型状态。
- **TransAware 的收敛曲线不可信**：`progressive_placer.py:224` 的
  `run_time = run_time/np.log(ep+3)*50` epoch 重缩放使"best@epoch17"成为构造性必然；
  自报 30.21s 与图上任何放置的真实模拟值（~1.8–1.9s）都不可比。

### 8.5 第四基线 HSDAG（NeurIPS'24）的迁移与结果

**原版机制**：op 级放置——GPN（Graph Parsing Network）图解析出个性化共置簇 → 每簇独立分类到设备；
reward = 1/OpenVINO `HETERO:GPU,CPU` 真执行时间（10 次推理去前 5 warmup 取均值）；全放单一设备罚 1e10。

**本机四个硬阻塞 → 采用方案 A（算法移植）**：OpenVINO GPU 插件仅支持 Intel GPU（Titan V 不可见）；
无内部模拟器（无法消费烘焙成本）；`num_devices=2`；`merge_operations` 在 35 节点纯链上级联合并成 1 组
（策略退化为单动作，必触发 1e10 罚）。故保留 GPN+PPO(REINFORCE) 策略栈原样（`PPO/GPN` 逐行照抄，
含个性化簇反馈），仅换 reward 后端。适配器 `/data/coding/HSDAG/hsdag_unified.py`，四处偏离全部落盘在
结果 json 的 `deviations` 字段：

- **D1** 图源：OpenVINO IR → 统一 pkl（35 节点，TitanV 实测成本烘焙）；
- **D2** 共置分组：级联合并 → 单点组（否则链图上策略无自由度）；
- **D3** 设备数：2（CPU+1GPU）→ 8（GPU）；
- **D4** reward：OpenVINO 真执行 → 统一 HDP 族事件驱动模拟器（与 TransAware/Trinity 同一评估后端）——
  三个 op 级基线由此共享同一 reward，差异只剩搜索算法本身，口径受控。

**验收与结果**（galvatron env，PyG 栈 torch-scatter 2.0.9 + torch-sparse 0.6.13 + torch-geometric 2.0.4，
从 `data.pyg.org/whl/torch-1.10.0+cu113` 装 cp38 wheel）：
- 校准通过（首跑即过）：全放 GPU0 → 模拟器输出 **1.810202s**（期望 ~1.81）✓，交错 1.857s，单设备罚 1e10 ✓；
- 搜索 170 episodes × 20 rollouts，**129.38s**，最优估计 1.81158s ≈ 均匀放置（又一次印证 op 级空间无更优解）；
- 最优 placement GPU 分布 {7:28, 2:7}，真机执行（`placement_runner_bert.py --bsz 8 --placement
  7,×28,2,×7`）→ GPU7 10.12GB 已分配、62MB free、申请 80MB 失败；`max_split_size_mb:128` 重试
  错误一字不差 → 合法 OOM，**r_d = −10**（同 8.3.2 的判定标准）；
- 与 TransAware/Trinity 同病：HSDAG 的 reward 同样只看激活、对 10.6GB 模型持久状态失明，
  于是把 28/35 个单元堆上 GPU7。

**搜索预算口径（诚实披露，与 TransAware 同款机制）**：原版默认 25 eps 搜索仅 20.38s——策略基本没训练。
正式对比统一采用重载配置 `--max_episodes 170`（策略多训练 ~7 倍），墙钟自然落在 **129.38s**
（标定：25 eps=20.38s，170 eps=129.38s，两次最优解完全相同：1.81158s / GPU{7:28, 2:7}，
故真机 −10 判定沿用不变）。可用 `HSDAG_EPS` 环境变量覆盖（同 `TA_ROUNDS`）。

### 8.6 结论：统一模型后谁更好

**Galvatron 完胜，且不是险胜是维度差**：
1. **唯一真机可行**——TransAware/Trinity/HSDAG 三个 op 级基线的搜索最优解在 8×Titan V 上全部
   直接 OOM（−10），Galvatron 8G 实测 1.321s 正常训练；
2. **唯一超过无搜索基准**——均匀放置 1.810s 是 op 级放置方法的结构上限（其搜索空间内无更优解），
   Galvatron 靠 tp/fsdp/pp 分片拿到 1.32s（−27%），手工 pp8 达 0.883s（−51%）；
3. **搜索开销可接受**——174s（确定性 DP，无需 RL 试错）vs Trinity 680s（且搜出不可行解）；
   统一重载口径下 TransAware ~105s、HSDAG ~129s 均快于 Galvatron，但都搜出 OOM 解
   （且 TransAware 原版默认 18 轮仅 1.17s = lr=1e-8 空转，HSDAG 原版默认 25 eps 仅 20.4s）。

**方法论结论**：op 级放置 RL（HDP 家族，含 HSDAG 的 GPN 变体）在"大模型 + 小显存卡"场景系统性失效——
模拟器看不见模型持久状态、搜索空间没有分片/流水维度；Galvatron 的层级并行搜索
以成本模型+显存硬约束直接工作在可行域内。这也正是论文 VLDB'23 的立论。
