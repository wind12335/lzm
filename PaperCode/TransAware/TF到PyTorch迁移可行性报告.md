# TransAware 双项目 TensorFlow → PyTorch 迁移可行性报告

- **报告日期**：2026-08-11
- **分析对象**：
  - `TransAware/transaware`（TransAware 策略搜索系统）
  - `TransAware/Hierarchical`（Trinity 分层 RL 设备放置系统）
- **分析人**：Claude Code（基于对两个项目全部核心源码的逐文件阅读）

---

## 一、结论摘要

| 项目 | 能否迁移 | 难度 | 核心障碍 |
|---|---|---|---|
| **transaware** | ✅ **可以，难度中等** | ★★☆☆☆ | 约 1500 行神经网络/RL 代码需重写；输入数据与模拟器均为纯 Python，**无需修改** |
| **Hierarchical** | ⚠️ **可以但难度较大** | ★★★★☆ | 深度依赖 TF **Grappler** 图优化与 **MetaGraphDef protobuf** 输入格式，**数据层需重新设计** |

**关键判断：**

1. **transaware 数据流与 TF 几乎解耦**。输入是 `networkx` 图（纯 Python），执行评估是纯 Python 事件模拟器（`simulator.py`）。只有 Graphormer 图编码、RL 策略网络、以及 Session 控制流依赖 TF，迁移属于"净重写"，工作量可控，且可用**同一 pkl 输入下 TF 版与 PyTorch 版结果一致**作为验证基准。
2. **Hierarchical 的难点不在神经网络**（就是 LSTM + PPO），而在于它建立在 TF 专有静态图基础设施上：`tf_optimizer.OptimizeGraph`（Grappler）、`cluster.Cluster` / `item.Item`（硬件簇与计算图 item）、`cluster.MeasureCosts`（成本测量）、`MetaGraphDef` 输入格式、`step_stats` 执行统计。**这些在 PyTorch 生态中没有等价物**，必须先重构输入数据层。
3. **当前运行环境**（Python 3.10.12、Torch 2.5.1、无 TensorFlow、无 torch_geometric、networkx 3.4.2）下，部分 pkl 数据集（如 `gnmt_baseline.pkl`、`nmt_4layers_baseline.pkl`）内含 TF `MetaGraphDef` 对象，**无 TensorFlow 时连反序列化都会报错**。这既是迁移的现状阻碍，也印证了迁移的必要性。

---

## 二、项目架构与数据流

### 2.1 transaware —— TransAware 算子放置策略搜索

**定位**：DNN 算子 → 设备（GPU）放置的强化学习系统。用 Graphormer 编码计算图，用 LSTM+PPO 生成放置策略，用纯 Python 模拟器评估放置的运行时/内存峰值。

```
strategy_search.py
   │  读取 pkl 中的 'nx_graph'（networkx 有向图：节点=算子，边=通信量 num_bytes）
   ▼
PyAwareInterface → ProgressivePlacer.search_policy()      ← progressive_placer.py
   │  ① convert_input_data()：硬编码特征(compute_cost×0.5+memory×0.5) + 距离矩阵
   ▼
Graphormer 编码计算图                                     ← embedding_network.py
   │  中心度编码 + 空间编码 + 多头注意力（Graphormer，Ying et al. 2021）
   │  输出节点嵌入 x
   ▼
ReinforceModel（PPOStrategyGenerator）                    ← LSTM.py
   │  LSTM 策略/价值网络；18 轮循环：
   │    前向生成 policy/value/device_indices
   │    → 模拟执行 → 计算 reward
   │    → ppo_loss 反向更新
   ▼
AwareSimulator.simulate(placement)                        ← simulator.py（纯 Python）
   │  返回 run_time / start_time / mem_peak
   ▼
best_placement（最优算子-设备映射）
```

**数据格式**：pkl 内含 `nx_graph`（networkx 图，节点属性 `compute_cost`/`memory`/`device`，边属性 `weight`/`tensor`）。**与 TF 图无关**。

### 2.2 Hierarchical —— Trinity 分层 RL 设备放置

**定位**：基于 Google Hierarchical Controller / Placeto 框架的分层 RL 设备放置系统，含 PPO 改进，并集成了 node2vec 图嵌入。

```
Trinity_main_test.py / Trinity_main_interface.py
   │  TrinityControllerTest._buildCluster()   ← device_properties_pb2 构造虚拟 GPU/CPU 硬件
   │  ReadPickleFile 读 pkl：
   │      'optim_mg'(MetaGraphDef) + 'G'(networkx) + 'ungrouped_mapping'
   ▼
GraphScheduling.__init__                                 ← graph_scheduling.py
   │  tf_optimizer.OptimizeGraph(...)          ← Grappler 图重写优化
   │  gitem.Item(optimized_metagraph)
   │  cluster.MeasureCosts(item)               ← 测量"原始运行成本"基线
   ▼
schedule_graph()：with training.MonitoredSession(...)
   │  每 5 步：
   │    model.update_ppo / partitioning / create_group_embeddings / scheduling
   │    → sim 模拟 or 真实执行 → compute_rewards
   ▼
TrinityProgram（三层 RL 策略）                            ← Trinity_program_norml.py
   │  Grouper 网络（算子分组）+ Placer 网络（组→设备放置）
   │  LSTM encoder/decoder；PPO（old/new 双策略 + clip）
   │  node2vec 图嵌入（ge/，纯 Python）
   ▼
export_placement(metagraph)                 ← 把放置结果写回 MetaGraphDef
```

**数据格式**：pkl 内含 **TF `MetaGraphDef`**（`optim_mg` 键）、networkx 图（`G`）、`op_perf`（Grappler 测量）、`step_stats`（TF profiler 执行统计）。**输入数据本身强依赖 TF protobuf**。

---

## 三、TensorFlow 依赖盘点

### 3.1 transaware（共约 5452 行，需重写约 1500 行）

| 文件 | 行数 | TF 依赖点 | 类别 | 迁移难度 |
|---|---|---|---|---|
| `embedding_network.py` | 611 | `tf.keras.Model/Layer`、`tf.matmul`、`tf.nn.softmax`、`tf.placeholder`、`tf.Variable`、`tf.variable_scope`、`tf.random.uniform`、`tf.cond`、`tf.repeat`、`tf.transpose` | 神经网络 | **困难**（Graphormer 逐行翻译） |
| `LSTM.py` | 121 | `tf.keras.Model`、`tf.keras.layers.LSTM/Softmax`、`tf.train.AdamOptimizer`、`tf.gradients`、`tf.clip_by_value`、`tf.exp`/`tf.log`、`tf.TensorArray` | 神经网络 + 训练 | **中等**（PPO 数学可保留） |
| `progressive_nn.py` | 326 | `tf.train.get_or_create_global_step`、`AdamOptimizer`、`compute_gradients`/`apply_gradients`、`exponential_decay`/`polynomial_decay`、`tf.placeholder_with_default`、`tf.random.categorical`、`tf.nn.sparse_softmax_cross_entropy_with_logits`、`tf.linalg.global_norm`、占位符梯度回填 | 神经网络 + RL 训练 | **中等偏难** |
| `progressive_placer.py` | 530 | `tf.Session`、`tf.constant`、`tf.convert_to_tensor`、`tf.ConfigProto`、`tf.OptimizerOptions`、`sess.run`、`tf.disable_eager_execution` | 图执行模型 | **中等** |
| `shell/accuracy/graphormer.py` | ~600 | 与 embedding_network 同源的 GraphSAGE/Aggregator/Classifier | 神经网络 | **困难** |
| `progressive_graph.py` | 235 | **无**（纯 numpy/networkx） | — | ✅ 无需迁移 |
| `simulator.py` | 258 | **无**（纯 Python 事件驱动模拟器） | — | ✅ 无需迁移 |
| `graph.py` / `merge.py` / `log.py` | ~190 | **无**（纯 networkx） | — | ✅ 无需迁移 |
| `interface.py` | 40 | **无** | — | ✅ 仅微调 import |
| `strategy_search.py` | 210 | **无**（读 pkl 的 nx_graph） | — | ✅ 仅微调入口 |
| `test_ZINC.py` 等 4 个入口 | ~500 | 已混用 `torch_geometric` + TF placeholder/Session | 混合 | 重写为纯 PyTorch |

### 3.2 Hierarchical（共约 2.6 万行，需重写约 8000 行，含三个 RL 程序）

**A. 深度依赖 TF 专有基础设施（PyTorch 无等价物，需重新设计）**

| 依赖 | 位置 | 用途 | 处理方式 |
|---|---|---|---|
| `MetaGraphDef` / `GraphDef` protobuf | 所有入口、`readfile.py`、`programer.py` | 输入图数据 | **重构**：改用 networkx `G` + 节点属性，丢掉 MetaGraphDef |
| `tf_optimizer.OptimizeGraph` | `graph_scheduling.py:124` | Grappler 图优化 | **删除**（非必需） |
| `gitem.Item`：`IdentifyImportantOps` / `GetOpProperties` / `GetColocationGroups` | `graph_scheduling.py`、`programer.py` | 图分析 | 用 networkx 重写 |
| `cluster.Cluster` / `MeasureCosts` / `GetSupportedDevices` / `ListDevices` | `graph_scheduling.py`、`programer.py`、`Trinity_main_*.py` | 硬件簇 + 成本测量 | 用 dict 描述硬件；成本用纯 Python 模拟器替代 |
| `step_stats` / `node_stats` / `tensor_description` | `readfile.py`、`sim/tf_sim/*`、`to_digraph.py` | TF profiler 统计 | 改为从 pkl 字典读取，不依赖 protobuf 结构 |
| `MonitoredSession` / `ChiefSessionCreator` / `tf.summary.FileWriter` | `graph_scheduling.py` | 会话管理 + 日志 | `torch.utils.tensorboard` / 普通循环 |
| `device_properties_pb2` | `Trinity_main_*.py` | 模拟 GPU 硬件规格 | 改为普通 dict |

**B. 神经网络部分（PyTorch 有等价物，可翻译）**

| 文件 | 行数 | TF 依赖 | 迁移难度 |
|---|---|---|---|
| `Trinity_program_norml.py` | 1723 | LSTM encoder/decoder、Grouper/Placer 网络、PPO 损失、`state_ops.assign`、`control_flow_ops.cond`、`array_ops.boolean_mask`、`get_local_variable`、`embedding_ops` | **困难**（最大工作量） |
| `trinity_program_ppo.py` | 1390 | 同上（PPO 变体） | **困难** |
| `Trinity_program_node2vec.py` | 1335 | 同上（node2vec 变体） | **困难** |
| `ColorRL_program.py` | 1211 | 同上（基线方法） | **中等偏难** |
| `graph_scheduling_base.py` | ~370 | 同 graph_scheduling | **中等** |
| `programer.py`（Controller 基类） | 150 | Grappler item 接口 | **中等** |
| `aware_fuse.py` | 527 | **无**（纯 networkx 算子融合） | ✅ 无需迁移 |
| `graph_scheduling.py` | 368 | Grappler + MonitoredSession | **中等偏难** |
| `ge/` 图嵌入 | ~700 | node2vec/struc2vec/deepwalk **纯 Python**；仅 `line.py`、`sdne.py` 依赖 TF（sdne 在 norml 中是注释状态） | ✅ 基本无需迁移 |
| `nets/`（slim 模型库） | ~18000 | 全部 slim/TF | **不需要迁移**（参考模型，非核心路径） |
| `deployment/` / `preprocessing/` | ~1300 | slim/TF | **不需要迁移** |
| `to_digraph.py` | ~600 | Grappler + step_stats | **中等** |

---

## 四、迁移的核心难点

### 4.1 范式差异：TF1 静态图 vs PyTorch 动态图（两项目共有）

TF1 是 **静态图 + Session + placeholder + feed_dict** 模式：先 `build()` 出整张计算图，再通过 `feed_dict` 喂数据、`sess.run()` 取结果。PyTorch 是 **动态图 + 直接前向调用**。

| TF1 代码 | PyTorch 替代 |
|---|---|
| `x = tf.placeholder(tf.float32, shape=[None, d])` | 删掉，直接 `x = torch.tensor(data)` 传入 |
| `feed = self.model.get_feed_dict(...)`；`sess.run(ops, feed_dict=feed)` | `model(x, mask)` 直接调用 |
| `sess.run(tf.global_variables_initializer())` | 删掉（参数随模型构造自动初始化） |
| `optimizer.compute_gradients(loss, vars)` → 占位符回填 → `apply_gradients` | `loss.backward(); optimizer.step()` |
| `tf.train.get_or_create_global_step()` | `self.global_step += 1` 或 `optimizer` 的 step 计数 |
| `tf.train.exponential_decay(...)` | `torch.optim.lr_scheduler.*` 或手动计算 |

**特别说明**：`progressive_nn.py` 中 `_build_train_ops` 用 placeholder 构建 `gradphs_and_vars` 回填梯度、`_clip_grads_and_vars` 用 `global_norm` 截断——这些在 PyTorch 里分别用 `loss.backward()` + `torch.nn.utils.clip_grad_norm_` 即可完整替代，**应整体简化而非逐行翻译**。

### 4.2 transaware 特有：迁移近似"纯翻译"

1. `progressive_graph.py`（embedding 生成）与 `simulator.py`（执行模拟）为**纯 Python**，一行不改。
2. `embedding_network.py` 的 Graphormer 就是标准 Graphormer（Ying et al. 2021）：中心度编码（`nn.Embedding`）、空间编码（`Dense→Dense` bias）、`split_heads`、`scaled_dot_product_attention`（QK^T/√d + 空间编码 bias 按 head 复制 + softmax）、pre-LayerNorm TransformerBlock。可**逐行对照翻译成 `nn.Module`**。
   - 需要重点核对的广播细节：`scaled_attention_logits += spatial_encoding_bias` 中 `tf.repeat(..., num_heads, axis=1)` 对应 PyTorch `unsqueeze(1).expand` 或 `repeat_interleave`。
   - `FNN` 的 `glorot` 初始化范围 `sqrt(6/(d_in+d_out))` 对应 `nn.init.xavier_uniform_`。
3. `LSTM.py` 的 PPO：`ppo_loss`（clip 代理目标 + 价值损失 + 熵正则）、`compute_gae`、`td_update` 的**数学公式原样保留**，仅替换张量后端。

### 4.3 Hierarchical 特有：输入数据层是最大障碍

```python
# readfile.py 现状
j = pickle.load(f)          # ← 需要 tensorflow 才能反序列化（MetaGraphDef 对象）
mg = j['optim_mg']          # ← TF MetaGraphDef
G  = j['G']                 # ← networkx 图（实际算法主要用这个）
op_perf, step_stats = j['op_perf'], j['step_stats']   # TF profiler 输出
```

且 `programer.py` 的 `Controller` 基类直接使用 Grappler item 接口：

```python
self._node = {n.name: n for n in item.metagraph.graph_def.node}
item.IdentifyImportantOps(sort_topologically=True)   # ← Grappler
item.GetOpProperties()
cluster.ListDevices()
item.GetColocationGroups()
```

这些 API **在 PyTorch 中不存在**。迁移必须做两个决策：

**决策 A（推荐）——数据层整体重构**
- 既然 `mg`（MetaGraphDef）实际只用于读取节点名/设备/算子上限，而这些信息 networkx `G` 里已大体具备，可写一个 `load_graph()`：从 pkl 提取 `G` + 节点 cost/mem/device 属性，**丢弃 MetaGraphDef**。
- 用 networkx 重写 `programer.py`：`identify_important_ops`（拓扑排序选重要算子）、`get_node_fanin/fanout`、`colocation_groups`（`aware_fuse.py` 已有现成的同款逻辑可参考）。
- `to_digraph.py` 的 `main_cost(step_stats)` 改为直接读取 pkl 中已解析好的 `op_perf` 字典。
- `graph_scheduling.py` 中 `OptimizeGraph` 与 `MeasureCosts` **直接删除**——它们只是计算"原始运行时间"基线，可用纯 Python 模拟器替代（transaware 的 `simulator.py` 是现成参考）。

**决策 B（保守）——保留 MetaGraphDef**：仅把神经网络层换成 PyTorch。代价是迁移后仍须安装 TensorFlow 才能反序列化 pkl。**不推荐**。

### 4.4 Hierarchical 的 RL 结构
三层策略（Grouper + Placer）+ PPO old/new 双网络。代码中大量 `state_ops.assign`（reward/baseline/ready 变量更新）、`control_flow_ops.cond`（条件分支）、`array_ops.boolean_mask`（掩码选损失）、`get_local_variable`（局部变量）。翻译集中在 `Trinity_program_norml.py` 的 `build_program()`（410–650 行）与 `trinity_program_ppo.py`。PyTorch 里用 `nn.Parameter` + 普通 if/else + `boolean_mask`（`x[mask]`）可直接改写，但需吃透 reward/baseline/loss 的更新时序。

---

## 五、迁移策略建议

### 5.1 推荐路径：先 transaware，后 Hierarchical

**Phase 1 — transaware（预计 2–4 个工作日）**
1. `embedding_network.py` → `nn.Module`：Graphormer（多头注意力、空间/中心度编码、TransformerBlock）、Aggregator/Classifier。
2. `LSTM.py` → `PPOStrategyGenerator(nn.Module)`：LSTM + policy/value head；`ppo_loss`/`compute_gae`/`td_update` 保留数学。
3. `progressive_nn.py` → `ProgressiveNN`：将占位符梯度机制简化为 `loss.backward()`；保留 `_sample`（`torch.multinomial`）、`_get_entropy`。
4. `progressive_placer.py`：`convert_input_data` 用 numpy 直接算；`search_policy()` 18 轮循环改为 `for e in range(18): policy,value,_ = model(x); ...; loss.backward(); opt.step()`。
5. 清理 `shell/accuracy/graphormer.py`、`test_ZINC*.py` 等入口脚本为纯 PyTorch。

**Phase 2 — Hierarchical（预计 1–2 周）**
1. **数据层**：`readfile.py` 的 PyTorch 版解析（只读 `G` + 节点属性 + `op_perf` 字典），删掉 MetaGraphDef 依赖；用 networkx 重写 `programer.py` 的 `Controller`。
2. `graph_scheduling.py`：删 `OptimizeGraph`/`MeasureCosts`/`MonitoredSession`，改为 `for step: model(emb); simulate; loss; backward; step` 循环；日志改用 `tensorboard`。
3. `Trinity_program_norml.py` / `trinity_program_ppo.py` / `ColorRL_program.py`：翻译 LSTM encoder/decoder、Grouper/Placer、PPO 损失（**最大工作量**）。
4. `Trinity_main_*.py`：`device_properties_pb2` 改为普通 dict 描述硬件。
5. `nets/`、`deployment/`、`preprocessing/` **不迁移**；`ge/` 保留 node2vec/struc2vec/deepwalk，`line.py`/`sdne.py` 若未引用可直接忽略。

### 5.2 可以迁移 / 删掉的清单（Hierarchical）
- `nets/`（全部 slim 模型）、`deployment/`、`preprocessing/`：约 1.8 万行，核心路径不 import，**不需要迁移**。
- `datasets/profile/` 下 TF 工具链（`get_forward_graph.py` 等）：属于"生成 pkl 数据集"的上游工具，不属于要跑的算法。
- 三个 RL 程序是平行变体：`Trinity_main_test.py` 默认走 `Trinity_program_norml` + `ColorRL_program` 两个分支，可先只迁移这两个；`Trinity_program_node2vec.py` 按需再迁。

### 5.3 transaware 无需迁移清单
- `progressive_graph.py`、`simulator.py`、`graph.py`、`merge.py`、`log.py`：**全部保留**，纯 Python。

---

## 六、风险提示

1. **数据依赖是硬伤**：Hierarchical 的 pkl（`optim_mg` 键）为 TF protobuf，无 TF 环境直接 `pickle.load` 会报 `No module named 'tensorflow'`（实测 `gnmt_baseline.pkl`、`nmt_4layers_baseline.pkl` 即此错误）。迁移第一步必须先确认数据中哪些字段真的需要、能否用 networkx 版本重新生成。
2. **transaware 存在死代码**：`progressive_placer.py` 的 `place()` 方法（248–365 行）引用了未定义的 `lstm`、`self.model`、`self.model_saver`，属坏掉的遗留代码；真正运行的是 `search_policy()`。迁移时直接保留 `search_policy()` 路径即可。
3. **模拟器一致性可作验证基准**：transaware 的 `simulator.py` 是确定性事件模拟器，不经过 TF。迁移后可用**同一 pkl 输入，对比 TF 版与 PyTorch 版的 `best_placement` 与 `run_time`** 验证正确性。
4. **Hierarchical 的 `step_stats`/`op_perf` 可能缺失**：部分 pkl（如 `nasnet_32.pkl` 为纯 ndarray）没有这些键，`get_outperform()` 会 KeyError，迁移时须做容错（默认值）。
5. **`torch_geometric` 未安装**：transaware 的 `test_ZINC*.py` 引用 PyG，当前环境没有；若只跑主链路 `strategy_search.py` 则不需要。
6. **`datasets/InceptionV3/` 缺 pkl**：该目录下只有 `__pycache__`，`strategy_search.py --model_name=InceptionV3` 会找不到数据文件，迁移验证时应优先用 GNMT / NMT_4layers / NasNet 数据集。

---

## 七、工作量估算

| 项 | 需改写行数 | 预计投入 |
|---|---|---|
| transaware 核心（embedding_network + LSTM + progressive_nn + progressive_placer） | ~1600 行 | 2–4 天 |
| transaware 入口/测试清理 | ~500 行 | 0.5 天 |
| Hierarchical 数据层 + Controller 基类 + graph_scheduling | ~1200 行 | 3–5 天 |
| Hierarchical 三个 RL 程序 | ~4300 行 | 5–8 天 |
| Hierarchical 入口 / cluster 模拟 | ~400 行 | 1 天 |
| **合计** | **约 8000 行** | **约 2–3 周（单人）** |

---

## 八、结论与建议

1. **可行性结论**：两个项目**均可迁移到 PyTorch 运行**。transaware 属于"教科书级"的可迁移案例（数据与模拟器已与 TF 解耦）；Hierarchical 需要先重构数据层，属于"高成本但可行"。
2. **迁移顺序建议**：先 transaware 练手并建立验证基准（同一 pkl 输入下 TF 版与 PyTorch 版结果一致），再攻 Hierarchical 数据层，最后翻译 RL 程序。
3. **与导师对齐目标**：务必确认是要求"**逐位复现原算法**"，还是只需"**在 PyTorch 中能跑出可用的放置策略**"。若是后者，Hierarchical 的 Grappler 依赖可直接砍掉，工作量可省约 1/3。
