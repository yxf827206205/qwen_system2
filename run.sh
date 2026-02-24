#!/bin/bash

# 打印提示
echo "🚀 正在启动 4 个并行的 MCTS 收割机进程..."

# 使用 nohup 和 & 将 4 个独立的进程同时挂在后台运行！
nohup python run_gsm8k_star_parallel.py --total_shards 4 --shard_idx 0 > log_shard_0.txt 2>&1 &
nohup python run_gsm8k_star_parallel.py --total_shards 4 --shard_idx 1 > log_shard_1.txt 2>&1 &
nohup python run_gsm8k_star_parallel.py --total_shards 4 --shard_idx 2 > log_shard_2.txt 2>&1 &
nohup python run_gsm8k_star_parallel.py --total_shards 4 --shard_idx 3 > log_shard_3.txt 2>&1 &

echo "✅ 4 个进程已全部在后台启动！"
echo "你可以使用 'tail -f log_shard_0.txt' 等命令查看实时进度。"