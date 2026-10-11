#!/usr/bin/env bash
# 在真实 NMT_4layers 数据集上运行 TransAware 策略搜索（PyTorch 版）
set -euo pipefail
cd "$(dirname "$0")"
python strategy_search.py --model_name NMT_4layers --debug_placement "$@"
