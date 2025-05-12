#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

# python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

python eval.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_minicpm-o_unimodal_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &
