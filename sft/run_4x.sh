#!/bin/bash

export PYTHONPATH=".:./grpo"


mkdir -p /root/autodl-tmp/data/chunks

echo "启动 4 个并发进程，开始榨干显卡..."


python select_value_data.py --start_idx 0 --end_idx 125 --output_path /root/autodl-tmp/data/chunks/chunk_1.jsonl &


python select_value_data.py --start_idx 125 --end_idx 250 --output_path /root/autodl-tmp/data/chunks/chunk_2.jsonl &


python select_value_data.py --start_idx 250 --end_idx 375 --output_path /root/autodl-tmp/data/chunks/chunk_3.jsonl &


python select_value_data.py --start_idx 375 --end_idx 500 --output_path /root/autodl-tmp/data/chunks/chunk_4.jsonl &


wait

echo " 4 个进程全部完成！正在合并数据..."


cat /root/autodl-tmp/data/chunks/chunk_*.jsonl > /root/autodl-tmp/data/value_data_full.jsonl

echo "最终数据位于 /root/autodl-tmp/data/value_data_full.jsonl"