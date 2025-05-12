#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

# nohup bash eval_gpu4.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_minicpm-o_unimodal_gpu4_$(date +%Y%m%d%H%M%S).log 2>&1 &
