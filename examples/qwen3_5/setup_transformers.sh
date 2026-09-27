#!/bin/bash
# 拉取本项目所需的 transformers 源码版本。
#
# 为什么要单独做这一步
#   本项目依赖 transformers 5.2.0.dev0（含 Qwen3.5 / Qwen3VL-MoE 架构支持），
#   pip 上的正式版不包含这些模型，直接安装会因"不认识 model_type"而失败。
#   该源码树约 95 MB（不含 .git），不适合放进 git 仓库，故用脚本获取。
#
# commit 已锁定，保证可复现。
set -e

TF_COMMIT=fc9137225880a9d03f130634c20f9dbe36a7b8bf
TF_URL=https://github.com/huggingface/transformers.git
DEST=${1:-/workspace/third_party/transformers_src}

if [ -d "$DEST/.git" ]; then
  echo "[setup] 已存在，切到 $TF_COMMIT"
  git -C "$DEST" fetch --all --quiet || true
  git -C "$DEST" checkout -q "$TF_COMMIT"
else
  echo "[setup] 克隆到 $DEST"
  mkdir -p "$(dirname "$DEST")"
  git clone --quiet "$TF_URL" "$DEST"
  git -C "$DEST" checkout -q "$TF_COMMIT"
fi

echo "[setup] 安装（可编辑模式）"
pip install -e "$DEST" --no-deps -q

python -c "import transformers; print('[setup] transformers', transformers.__version__)"
