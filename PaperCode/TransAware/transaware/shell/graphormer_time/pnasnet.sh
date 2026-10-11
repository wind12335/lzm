echo "threshold 8"
python ../../strategy_search.py --model_name "PnasNet" --only_graphormer True --threshold 8
echo "threshold 16"
python ../../strategy_search.py --model_name "PnasNet" --only_graphormer True --threshold 16
echo "threshold 32"
python ../../strategy_search.py --model_name "PnasNet" --only_graphormer True --threshold 32