# 3rdparty/ —— 统一对比实验所依赖的上游基线代码快照

本目录收纳两个**外部方法**的代码快照，使仓库自包含（克隆即可看到统一实验用到的全部代码）。
两个快照都是本机工作副本的完整拷贝（排除 `__pycache__`/`.git`）；
**在本机上运行请继续使用原位置**（路径已写入各文档的命令中）。

| 目录 | 上游 | 快照版本 | 在统一实验中的角色 | 本机改动（相对上游） |
|---|---|---|---|---|
| `Galvatron/` | [PKU-DAIR/Hetu-Galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron)（VLDB'23, arXiv 2211.13878） | Hetu 仓库 commit `3757771` 时代的 paper 代码（非 git 拷贝） | 参与对比的第 2 方法 | ① `env_configs/` 按本机 8×Titan V 重新 profile（paper 原始预设在 `env_configs_paper_preset_backup/`）；② `bert/` 下**新增**本实验脚本：`build_bert_graph_pkl.py`（统一图构建）、`placement_runner_bert.py`（HDP 协议真机执行器）、`benchmark_step_latency.py`（步时延基准）及计时结果 json |
| `HSDAG/` | [hping666/HSDAG](https://github.com/hping666/HSDAG)（NeurIPS'24, arXiv 2405.14185, 官方实现） | commit `9d5caf7770cd50f975e014d19920873c112df095`（2025-01-09） | 参与对比的第 3 方法（第四基线） | **新增** `hsdag_unified.py` 适配器（方案 A 算法移植：保留 GPN+PPO，reward 后端换统一 HDP 模拟器，偏离 D1–D4 见文件头与 `HSDAG_output/best_result_unified.json` 的 deviations 字段）；其余文件未改动 |

其余两个方法的代码在仓库本体内，不在此目录：
- **TransAware（Ours）**：`PaperCode/TransAware/transaware_pytorch/`（PyTorch 移植版，本机运行的版本）；
- **Hierarchical（Trinity）**：`PaperCode/TransAware/Hierarchical/Trinity/`（含本机加的 TF2.13 兼容垫片，集中在 `Trinity_main_test.py` 头部）。

## 环境与版本（复现统一实验所需）

**机器**：8× NVIDIA Titan V（Volta sm_70, 12GB, PCIe），driver 550.67，CUDA toolkit 11.6，64 核 / 251GB RAM。

两套 conda 环境（复现手册 `docs/统一模型实验执行手册.md` §2/§9 有完整细节与坑）：

| 环境 | 路径 | 关键版本 | 跑什么 |
|---|---|---|---|
| `galvatron` | `/data/miniconda/envs/galvatron` | Python 3.8.20, torch 1.10.2+cu113, torchvision 0.11.3, transformers 4.16.2（须 pin tokenizers 0.11.6 + huggingface_hub 0.4.0）, numpy 1.23.5（**必须 ≤1.23**）, fairscale 0.4.5, timm 0.5.4, networkx, scipy | TransAware、Galvatron、HSDAG、真机执行器 |
| `galvatron`（HSDAG 追加） | 同上 | torch-scatter 2.0.9 + torch-sparse 0.6.13（cp38 wheel，源自 `data.pyg.org/whl/torch-1.10.0+cu113`）+ torch-geometric 2.0.4（最后含 dropout_adj/segregate_self_loops 的版本） | HSDAG 适配器 |
| `tf` | `/data/miniconda/envs/tf` | Python 3.8, tensorflow-cpu 2.13.1, pandas, gensim, fastdtw, joblib, tqdm, networkx | Hierarchical/Trinity |

注意：
- 系统 `python` 是旧的 torch 2.1 环境，**永远用全路径**（所有文档命令已写全路径）；
- 联网（pip/装包）需 socks 代理：`export http_proxy=socks5h://127.0.0.1:10808 https_proxy=socks5h://127.0.0.1:10808`；
- 向本仓库推送用的 SSH deploy key 配置：`~/.ssh/config`（github.com → ssh.github.com:443 + PySocks 中继脚本）。
