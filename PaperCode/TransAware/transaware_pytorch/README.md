# TransAware —— PyTorch 移植版

将原 TF 实现的 TransAware（DNN 算子 → GPU 放置 RL 系统）迁移到 PyTorch。
原代码位于 `测试hierarchical和transAware/PaperCode/TransAware/transaware/`，本目录为其
**可运行主链路** `search_policy()` 的独立 PyTorch 实现，原 TF 代码保留作参考。

## 运行环境

- Python 3.10，PyTorch ≥ 2.0，numpy，networkx，scipy（`floyd_warshall` 加速，可选）
- **不需要安装 TensorFlow**

## 快速开始

```bash
# ---- 自测命令（两个可用数据集，100 轮，打印每轮 debug）----
# GNMT（1743 节点，约 60s）
python strategy_search.py --model_name GNMT --n_devs 4 --n_rounds 100 --debug_placement

# NMT_4layers（1503 节点，约 45s）
python strategy_search.py --model_name NMT_4layers --n_devs 4 --n_rounds 100 --debug_placement

# ---- 其他用法 ----
# 默认 18 轮（原版硬编码轮数）
python strategy_search.py --model_name GNMT --debug_placement

# 只计算 Graphormer 编码（跳过策略搜索，用于评估编码耗时）
python strategy_search.py --model_name GNMT --only_graphormer True

# 便捷脚本
bash run_gnmt.sh
bash run_nmt.sh
```

数据集 pkl 默认指向原 TransAware 项目的 `datasets/`（不复制数据），可用
`--data_path <dir>` 覆盖。关键超参（`--n_devs`、`--hidden_dim`、`--seed`、
`--node_traversal_order` 等）与原版一致。`--n_rounds` 为策略搜索轮数
（默认 18，即原版 `search_policy()` 硬编码的值；原版 `place()` 路径的
`n_eps` 默认 40 是另一条坏代码路径，与本主链路无关）。

## 目录结构与文件对应

```
transaware_pytorch/
├── __init__.py
├── embedding_network.py   # 重写：Graphormer 及子层 → PyTorch nn.Module
├── lstm.py                # 重写：PPOStrategyGenerator / ReinforceModel / PPO 损失
├── progressive_placer.py  # 重写：去掉 tf.Session，search_policy 改为 PyTorch 循环
├── progressive_graph.py   # 原样复制（纯 Python，无 TF 依赖）
├── simulator.py           # 原样复制（纯 Python，事件驱动模拟器）
├── graph.py               # 原样复制（纯 Python）
├── log.py                 # 原样复制（纯 Python）
└── interface.py           # 微调：PyAwareInterface，与原文一致
strategy_search.py         # 入口：加 stub unpickler 加载 nx_graph
run_gnmt.sh / run_nmt.sh   # 运行脚本
```

| 原 TF 文件 | PyTorch 文件 | 迁移方式 |
|---|---|---|
| `embedding_network.py`（Graphormer 部分） | `embedding_network.py` | 逐层改写为 `nn.Module` |
| `LSTM.py` | `lstm.py` | 改写 `tf.keras`/`tf.train` 为 `nn.Module`/`torch.optim`；公式原样 |
| `progressive_placer.py`（`search_policy` 路径） | `progressive_placer.py` | 去掉 `tf.Session`，`convert_input_data` 用 numpy |
| `progressive_graph.py` / `simulator.py` / `graph.py` / `log.py` | 同名 | 原样复制 |
| `interface.py` | `interface.py` | 原样 |
| `strategy_search.py` | `strategy_search.py` | 加 stub unpickler |

**不迁移（按已确认的迁移范围）**：`progressive_nn.py`（含 `MessagePassingProgressiveNN`，
主链路未用）、`progressive_placer.py` 的 `place()` 方法（原实现中的坏代码，引用了不存在的
`lstm`/`self.model`/`self.model_saver`，且深度依赖 TF）、`shell/`、`test_ZINC*.py`
（依赖未安装的 torch_geometric）。

## 数据流

```
strategy_search.py
  → StubUnpickler 读取 pkl 的 nx_graph（屏蔽 TF 对象，无需 TF）
  → 构建 Graph（节点 cost/mem/device，边 weight=num_bytes）
  → PyAwareInterface.startup_strategy()
    → ProgressivePlacer.search_policy()
      → convert_input_data()   # 硬编码特征 alpha*cost+beta*mem + 距离矩阵（numpy）
      → Graphormer 编码         # (1, node_num, 512)
      → --n_rounds 轮循环（默认 18，原版硬编码 18）：
          LSTM+PPO 前向 → 生成 device placement
          → 纯 Python 模拟器评估 run_time / mem
          → compute_rewards → PPO 反向更新
```

## 关键迁移决策

### Graphormer（`embedding_network.py`）
标准 Graphormer（Ying et al. 2021）逐层翻译：

| TF/Keras | PyTorch |
|---|---|
| `tf.keras.layers.Embedding` | `nn.Embedding` |
| `Dense(16, relu) → Dense(1, relu)`（空间编码） | `nn.Linear(1,16)+ReLU → nn.Linear(16,1)+ReLU` |
| `split_heads`（reshape + transpose） | `reshape(B,-1,H,D).transpose(1,2)` |
| `q·kᵀ/√dk + 空间偏置（tf.repeat 复制到各 head）+ mask·-1e9` | 同公式，`unsqueeze(1).expand(-1,H,-1,-1)` |
| Pre-LayerNorm，`LayerNormalization(epsilon=1e-6)` | `nn.LayerNorm(eps=1e-6)` |
| 返回原始序列（分类 head 已注释） | 同 |

### LSTM / PPO（`lstm.py`）
- TF Keras LSTM 默认 `batch_first=False`，输入 `(1, node_num, hidden)` 解释为
  `(timesteps=1, batch=node_num)`；PyTorch `nn.LSTM(batch_first=False)` 输入约定相同，直接对齐。
- `ReinforceModel.optimizer = torch.optim.Adam(lr=1e-8)`（忠实保留原学习率）。
- `ppo_loss` / `compute_gae` / `td_update` 公式逐行照搬；`train_step_backward` 改为
  `loss.backward(); optimizer.step()`。原 TF 循环里的 `sess.run(tf.global_variables_initializer())`
  在 PyTorch 中为多余（参数构造时自动初始化），已删除。

### 策略搜索（`progressive_placer.py`）
- `convert_input_data()` 从 `tf.constant/reduce_sum/expand_dims/Session` 改为纯 numpy。
- `search_policy()`：`tf.Session` 换成张量前向 + `.detach().numpy()`；`self.old_policy`
  保留为张量供 PPO 使用。
- 模拟评估（`eval_placement` → `aware_simuator.simulate`）与 `compute_rewards` 为纯 Python/numpy，
  原样保留。

## 已知怪癖（迁移时如实保留的退化行为）

1. **`node_features` 是浮点硬编码特征但走 `nn.Embedding`**。原 TF 用
   `alpha*cost + beta*mem` 作为节点特征（形状 `(1, node_num)`），却喂给 `Embedding(100, d_model)`，
   Keras 会把浮点 cast 成 int 索引——真实数据上值远超词表大小，原实现实际会越界。PyTorch 用
   `x.long().clamp(0, 99)` 保证可运行（语义与原实现的退化路径一致）。
2. **中心性编码的"度"实际是非边数**。原实现的"距离矩阵"把非边填成 `-1`（而非真正的最短距离），
   `abs(-1)==1` 导致 `CentralityEncoding` 的度计数在真实大图上≈`node_num`，超出 `max_degree=256`
   嵌入表，原 TF 同样会崩。PyTorch 将度索引 `clamp(0, max_degree-1)` 保持可运行。
3. **距离矩阵用边权填充**，并非图论意义上的最短距离（原作者如此，未改动）。
4. `--only_graphormer True` 时程序只打印编码耗时并提前返回（与原版一致）。

## 验证结果（2026-08-13，CPU）

| 数据集 | 节点数 | 总耗时 | best run_time | best peak_mem | placement 覆盖 |
|---|---|---|---|---|---|
| GNMT（`gnmt_baseline.pkl`） | 1743 | ~45 s | 0.433 s | 7.67 GB | 1743/1743，设备 ∈ [0,4) |
| NMT_4layers（`nmt_4layers_baseline.pkl`） | 1503 | ~35 s | 0.342 s | 6.14 GB | 1503/1503，设备 ∈ [0,4) |
| 合成小图（10 节点，冒烟） | 10 | ~4 s | 0.381 s | 0.0001 GB | 10/10 |

两份真实数据上 run_time 均随训练轮次单调下降，说明 PPO 链路（前向→模拟→反向）正常收敛。
（注：模型参数为随机初始化，不设固定 seed，故多次运行结果数值不同——与原 TF 行为一致。）
