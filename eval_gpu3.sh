#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

# nohup bash eval_gpu3.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_minicpm-o_unimodal_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &
