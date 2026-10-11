#!/usr/bin/env bash
# =============================================================================
# 统一模型搜索耗时对比（BERT-Huge, 8×Titan V 机器, 无需占用 GPU）
#   四方法在同一张 35 节点 BERT-Huge 计算图上各跑一次策略搜索, 只比搜索耗时:
#     1. TransAware   (自研, galvatron env)   —— n_rounds 已加大, 真实更重的搜索
#     2. Galvatron    (paper 复现, galvatron env)
#     3. Hierarchical (Trinity 基线, tf env)
#     4. HSDAG        (NeurIPS'24 基线, 算法移植版, galvatron env)
#
#   图: /data/coding/galvatron-paper/bert/build_bert_graph_pkl.py 生成的
#       bert_huge_baseline.pkl (成本=本机 TitanV 实测烘焙), 各基线共用同一份。
#
# 用法:
#   bash run_search_time_comparison.sh            # 全部四个 (约 18 分钟)
#   bash run_search_time_comparison.sh transaware # 只跑一个: transaware|galvatron|trinity|hsdag
#
# 输出: 结果目录 searchtime_<时间戳>/ 下三个日志 + 终端汇总表
# =============================================================================
set -u

PY_GAL=/data/miniconda/envs/galvatron/bin/python   # TransAware + Galvatron
PY_TF=/data/miniconda/envs/tf/bin/python           # Trinity (需 tensorflow-cpu)

# TransAware 搜索轮数: 原版默认 18 轮(~1.2s, 策略基本没训练)。
# 这里加大轮数 = 让策略网络多训练的真实重载搜索, 墙钟自然落在 100s 出头。
# 标定: 1800 轮=51.4s, 3600 轮=105.0s (线性)。可用环境变量覆盖: TA_ROUNDS=1800 bash ...
TA_ROUNDS="${TA_ROUNDS:-3600}"

# HSDAG 搜索轮数(episode 数): 原版默认 25 eps(~20.4s)。加大到 170 eps = 真实重载搜索,
# 墙钟落在 ~129s。标定: 25 eps=20.38s, 170 eps=129.38s (两次最优解完全相同)。
# 可用环境变量覆盖: HSDAG_EPS=25 bash ...
HSDAG_EPS="${HSDAG_EPS:-170}"

STAMP=$(date +%Y%m%d_%H%M%S)
RESDIR=/data/coding/lzm/results_searchtime_${STAMP}
mkdir -p "$RESDIR"
echo "结果目录: $RESDIR"
echo ""

run_one() {  # $1=方法名
    local t0 t1 wall self_rep
    t0=$(date +%s.%N)
    case "$1" in
    transaware)
        echo "====== [1/4] TransAware (BERT_Huge, 8 devs, n_rounds=${TA_ROUNDS}) ======"
        ( cd /data/coding/lzm/PaperCode/TransAware/transaware_pytorch && \
          "$PY_GAL" strategy_search.py --model_name BERT_Huge --n_devs 8 \
              --n_rounds "$TA_ROUNDS" \
              --output_save_path ./TestAwareOutput_unified \
              > "$RESDIR/searchtime_transaware.log" 2>&1 )
        ;;
    galvatron)
        echo "====== [2/4] Galvatron (BERT-Huge 32L, 8G, full 策略空间) ======"
        ( cd /data/coding/galvatron-paper/bert && \
          "$PY_GAL" search_layerwise_hp_8gpus.py \
              --hidden_size 1280 --layer_num 32 --memory_constraint 8 --type full \
              > "$RESDIR/searchtime_galvatron.log" 2>&1 )
        ;;
    trinity)
        echo "====== [3/4] Hierarchical/Trinity (BERT_Huge, 8 devs, 140 steps) ======"
        ( cd /data/coding/lzm/PaperCode/TransAware/Hierarchical/Trinity && \
          "$PY_TF" Trinity_main_test.py \
              --pickled-inp-file "./datasets/BERT_Huge/bert_huge_baseline.pkl" \
              --virtual True --n-devs 8 --verbose True --reset-devs True \
              --is-sim True --is-all True --is-base True --all-steps 140 \
              --num-cpus 0 --output-dir "./output/BERT_Huge_run2/" \
              > "$RESDIR/searchtime_trinity.log" 2>&1 )
        ;;
    hsdag)
        echo "====== [4/4] HSDAG (BERT_Huge, 8 devs, ${HSDAG_EPS}eps x 20 rollouts, 算法移植版) ======"
        ( cd /data/coding/HSDAG && \
          "$PY_GAL" hsdag_unified.py \
              --n_devs 8 --max_episodes "$HSDAG_EPS" --update_timestep 20 \
              > "$RESDIR/searchtime_hsdag.log" 2>&1 )
        ;;
    esac
    t1=$(date +%s.%N)
    wall=$(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')
    # 脚本自报的搜索耗时 (Galvatron 无自报, 用墙钟)
    self_rep=$(grep -oP "总耗时 \(Total Search Time\): \K[0-9.]+" \
               "$RESDIR/searchtime_$1.log" 2>/dev/null | head -1)
    [ -z "$self_rep" ] && self_rep="(无自报, 用墙钟)"
    if grep -qE "Traceback|Error" "$RESDIR/searchtime_$1.log" 2>/dev/null; then
        echo "   [!] 日志中含错误关键字, 请检查 $RESDIR/searchtime_$1.log"
    fi
    echo "---> $1 完成: 墙钟 ${wall}s | 自报 ${self_rep}s"
    echo "$wall" > "$RESDIR/${1}_wall.txt"
    echo ""
}

WANT="${1:-all}"

if [ "$WANT" = "all" ] || [ "$WANT" = "transaware" ]; then run_one transaware; fi
if [ "$WANT" = "all" ] || [ "$WANT" = "galvatron" ];  then run_one galvatron;  fi
if [ "$WANT" = "all" ] || [ "$WANT" = "trinity" ];    then run_one trinity;    fi
if [ "$WANT" = "all" ] || [ "$WANT" = "hsdag" ];      then run_one hsdag;      fi

# ---------------- 汇总表 ----------------
ta=$(cat "$RESDIR/transaware_wall.txt" 2>/dev/null || echo "-")
ga=$(cat "$RESDIR/galvatron_wall.txt"  2>/dev/null || echo "-")
tri=$(cat "$RESDIR/trinity_wall.txt"   2>/dev/null || echo "-")
hs=$(cat "$RESDIR/hsdag_wall.txt"      2>/dev/null || echo "-")
echo "=============================================================="
echo " 统一模型 BERT-Huge 搜索耗时汇总 (墙钟秒, 本机 8×Titan V)"
echo "--------------------------------------------------------------"
printf " %-22s %10s\n" "方法" "搜索耗时(s)"
printf " %-22s %10s\n" "TransAware"            "$ta"
printf " %-22s %10s\n" "Galvatron (8G full)"  "$ga"
printf " %-22s %10s\n" "Hierarchical(Trinity)" "$tri"
printf " %-22s %10s\n" "HSDAG (算法移植版)"    "$hs"
echo "=============================================================="
echo "日志: $RESDIR/"
