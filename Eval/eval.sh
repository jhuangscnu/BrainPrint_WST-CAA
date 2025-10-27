# 调用 ../Baseline 目录下的 eval.py（无参数）
cd ../Baseline 
python ../Baseline/eval.py
cd ../Train
# 调用 ../Train 目录下的 eval.py，带 --full 参数
python ../Train/eval.py --full

# 调用 ../Train 目录下的 eval.py，带 --ablation 参数
python ../Train/eval.py --ablation