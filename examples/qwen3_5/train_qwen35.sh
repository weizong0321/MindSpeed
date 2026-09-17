#!/bin/bash
# Qwen3.5-0.8B 微调启动脚本（FSDP2 后端）
#
# 用法：
#   source /workspace/mindspeed_env.sh          # 先激活环境（CANN + venv）
#   bash examples/qwen3_5/train_qwen35.sh
#
# 可用环境变量覆盖：
#   NPUS_PER_NODE  使用多少张卡（默认 1）
#   CONFIG         配置文件（默认 examples/qwen3_5/qwen3_5_0.8B_config.yaml）
#
# 注意：改前版本调用的是 `python train.py --config config/qwen3_5_0.8b.yaml`，
#       而仓库根目录既没有 train.py 也没有 config/ 目录，无法运行；
#       正确的入口是 mindspeed_mm/fsdp/train/trainer.py <配置文件>。

source /usr/local/Ascend/cann/set_env.sh

# FSDP2 后端标志
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH

NPUS_PER_NODE=${NPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
CONFIG=${CONFIG:-examples/qwen3_5/qwen3_5_0.8B_config.yaml}
WORLD_SIZE=$((NPUS_PER_NODE*NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py "$CONFIG" \
    2>&1 | tee logs/train_${logfile}.log
