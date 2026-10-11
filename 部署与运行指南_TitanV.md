# 部署与运行指南（8× Titan V 目标机）

> 目标机环境：Python 3.8 / torch 1.10（已有），8 张 Titan V。
> 本仓库在目标机上需要**两套独立虚拟环境**：transaware 用 torch 环境，hierarchical 用
> TensorFlow 环境（Trinity 依赖 TF grappler 虚拟集群做模拟，不动 torch 环境）。

## 0. 仓库结构

```
测试hierarchical和transAware/
├── PaperCode/
│   ├── TransAware/
│   │   ├── transaware/            # 论文方法一（TensorFlow 原版，仅作参考）
│   │   ├── transaware_pytorch/    # ★ 方法一 PyTorch 移植版（目标机上跑这个）
│   │   └── Hierarchical/Trinity/  # ★ hierarchical 基线（TensorFlow）
│   └── FasterTransAware/          # 方法二源码（build/、3rdparty/ 未入仓，见 §4）
├── results_local_20261011/        # 海光 DCU 机器上的复现日志与结论
├── 对比测试hierarchical和transAware性能_云端原始结果_202512.md
└── 部署与运行指南_TitanV.md       # 本文件
```

## 1. transAware（PyTorch 版）—— 用机器自带 torch 1.10 / py3.8

```bash
cd PaperCode/TransAware/transaware_pytorch
# 依赖机器已有 torch1.10+py3.8；如缺：
pip install networkx scipy
# 数据集默认自动探测到同级 ../transaware/datasets，无需配置

# 8 卡机跑 8 设备（n_devs 对应虚拟设备数）
python strategy_search.py --model_name NMT_4layers --n_devs 8 --debug_placement
python strategy_search.py --model_name GNMT         --n_devs 8 --debug_placement

# 便捷脚本（默认 n_devs=4，可透传参数）
bash run_nmt.sh --n_devs 8
bash run_gnmt.sh --n_devs 8
```

兼容性已验证（2026-10-11，海光机器静态审查）：
- vermin 逐文件扫描：10 个 .py 最低 Python 要求 ≤3.7（最高为 simulator.py 的 3.7）；
- torch API 面全部为 1.0 时代 API，无 torch2 独有调用；
- 参照运行结果（py3.11/torch2.7，n_devs=4）：NMT_4layers best run_time 0.247s、
  GNMT 0.347s，placement 全覆盖，PPO 收敛正常（详见 results_local_20261011/）。

## 2. hierarchical（Trinity）—— 单独 TF 虚拟环境

```bash
python3.8 -m venv ~/tf_env && ~/tf_env/bin/pip install -U pip
# py3.8 最后一个 TF 大版本是 2.13.x（2.14 起要求 py≥3.9）
~/tf_env/bin/pip install "tensorflow-cpu==2.13.1" networkx pandas gensim fastdtw

cd PaperCode/TransAware/Hierarchical/Trinity
# 脚本内为 GNMT 数据集；跑 NMT 改 --pickled-inp-file ./datasets/NMT_4layers/nmt_4layers_baseline.pkl
~/tf_env/bin/python Trinity_main_test.py \
    --pickled-inp-file "./datasets/GNMT/gnmt_baseline.pkl" \
    --virtual True --n-devs 2 --verbose True --reset-devs True \
    --is-sim True --is-all True --is-base True --all-steps 140 \
    --num-cpus 0 --output-dir "./output/GNMT/"
# 或直接: PATH=~/tf_env/bin:$PATH bash hierarchical_test.sh
```

TF 说明：
- 代码为 TF1 风格（`tf.disable_v2_behavior` + `tensorflow.python.grappler`），
  海光机器已用 **tensorflow-cpu 2.15.1（py3.11）** 验证 import 链与数据集反序列化
  （pkl 内嵌 MetaGraphDef/StepStats 可正常加载），py3.8 用 2.13.1 同链路；
- Trinity 用 TF grappler 的**虚拟 GPU 集群**模拟（伪造 TITAN X 设备属性），
  不需要真 CUDA，装 `tensorflow-cpu` 即可；
- 依赖链会拉 `pandas/gensim/fastdtw`（`Trinity.ge` 的 Node2Vec/DeepWalk 图嵌入需要）。

## 3. 两方法的对比口径（对齐云端 2025-12 原始结果）

| 模型 | hierarchical (Trinity) | transAware |
|---|---|---|
| NMT_4layers | `hierarchical_test.sh` 换 NMT pkl（或 `aware_test_nmt.sh` 配 `--is-base True`） | `strategy_search.py --model_name NMT_4layers --n_devs N` |
| GNMT | `hierarchical_test.sh`（默认 GNMT pkl） | `strategy_search.py --model_name GNMT --n_devs N` |

云端原始数字（230.52s vs 139.51s / 237.98s vs 155.92s）是各脚本上报的策略搜索执行
时间；在 Titan V 上复测时建议同时记录脚本上报值与整进程 wall，便于对齐。

## 4. 未入仓内容（在旧机器上，如需可另行转移）

| 路径 | 大小 | 说明 |
|---|---|---|
| `FasterTransAware/build/` | 1.7G | 编译产物，重编译即得（`mkdir build && cd build && cmake ..`） |
| `FasterTransAware/3rdparty/` | 273M | 第三方子模块（Megatron-LM 等），按 `.gitmodules` 重新拉取 |
| `FasterTransAware/.git_local_history` | ~60M | NVIDIA/FasterTransformer 修改版的独立 git 历史（含未提交修改） |
| `transaware/datasets/profile/.git_local_history` | - | profile 数据集原为独立 git 克隆 |
| `transaware/core.5414` | 2.5G | 核心 dump，垃圾文件 |
| `测试hierarchical和transAware.zip` | 688M | 仓库初始 zip 包（旧机器 lzm/ 下） |
