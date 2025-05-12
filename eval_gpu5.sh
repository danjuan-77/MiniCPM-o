#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

# nohup bash eval_gpu5.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_minicpm-o_unimodal_gpu5_$(date +%Y%m%d%H%M%S).log 2>&1 &
