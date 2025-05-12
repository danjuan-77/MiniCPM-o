#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval_gpu7.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_minicpm-o_unimodal_gpu7_$(date +%Y%m%d%H%M%S).log 2>&1 &
