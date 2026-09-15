#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH

python mindspeed_mm/fsdp/models/qwen3_5/weight_convert.py \
  --input_dir /workspace/mnt/share/weights/Qwen3.5-0.8B-VL \
  --output_dir /workspace/mnt/share/weights/Qwen3.5-0.8B-VL-dcp