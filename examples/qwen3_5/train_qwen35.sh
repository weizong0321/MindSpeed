#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

python train.py \
  --config config/qwen3_5_0.8b.yaml \
  --data_path ../../data/ecommerce_multimodal/train.json