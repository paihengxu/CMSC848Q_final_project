#!/bin/bash
srun --pty --qos=gpu-medium --time=8:00:00 --partition=gpu --gres=gpu:gtxtitanx:1 --exclude=materialgpu00 \
--mem=64g /fs/clip-emoji/tonyzhou/anaconda3/envs/q_pain/bin/python Q_Pain.py --medical_context_file "data_acute_cancer.csv"