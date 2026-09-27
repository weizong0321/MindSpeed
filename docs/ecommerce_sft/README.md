# Qwen3.5-0.8B 多模态 SFT（电商商品问答）

在 Ascend 910B2 单卡上，用 MindSpeed-MM 的 FSDP2 后端对 `Qwen3.5-0.8B` 做全参微调，
让模型能基于**商品图**回答用户问题，并且**只断言图上能可靠读出的客观结构属性**。

本 README 说明环境、启动方式、代码结构与 PR 状态。

---

## 环境版本

| 组件 | 版本 | 备注 |
|---|---|---|
| **CANN 版本** | **8.5.0** | `/usr/local/Ascend/cann → cann-8.5.0`；`ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0` |
| **torch_npu 版本** | **2.7.1** | — |
| PyTorch | 2.7.1+cpu | — |
| Python | 3.10.12 | — |
| transformers | 5.2.0.dev0 | 本地源码版（`/workspace/third_party/transformers_src`） |
| tokenizers | 0.22.2 | — |
| accelerate | 1.2.0 | — |
| datasets | 5.0.1 | — |
| safetensors | 0.8.0 | — |
| numpy | 1.26.0 | — |
| ATB（nnal） | 9.1.0 | `/usr/local/Ascend/nnal/atb/9.1.0` |
| 驱动 / npu-smi | 25.5.2 | ascendhal 7.35.23 |
| 硬件 | Ascend910B2 × 1 | HBM 64 GB，CPU 256 核，内存 2014 GB |

> 仓库自带兼容性矩阵（`README.md`）标注 MindSpeed-MM 2.3.0 对应
> CANN 8.5.0 / PyTorch 2.6.0,2.7.1 / Python 3.10 —— 与本环境一致。
> 环境中另有 `cann-9.1.0` 目录，但**未生效**（`latest` 软链接指向 8.5.0）。

---

## 运行启动脚本

### 0. 环境准备

```bash
source /workspace/mindspeed_env.sh
```

该脚本会：设置驱动 `LD_LIBRARY_PATH` → source CANN `set_env.sh` → source ATB `set_env.sh`
→ 激活 `/workspace/.venv` → `cd /workspace/MindSpeed`。

> ⚠️ 注意：它会切换工作目录。脚本里若依赖相对路径，务必**先 source 再 cd**。

### 1. 训练（单卡 FSDP2）

```bash
cd /workspace/MindSpeed
source /workspace/mindspeed_env.sh

CONFIG=examples/qwen3_5/qwen3_5_0.8B_sample8b_config.yaml \
NPUS_PER_NODE=1 \
  bash examples/qwen3_5/train_qwen35.sh
```

`train_qwen35.sh` 会设好 FSDP2 所需环境变量
（`NON_MEGATRON` / `MULTI_STREAM_MEMORY_REUSE` / `TASK_QUEUE_ENABLE` /
`PYTORCH_NPU_ALLOC_CONF` 等），并以 `torchrun` 调起
`mindspeed_mm/fsdp/train/trainer.py <配置文件>`。

### 2. DCP 权重转 HF

```bash
cd /workspace/MindSpeed
export PYTHONPATH=$PWD:$PYTHONPATH

python checkpoint/common/merge_dcp_to_hf.py \
  --load-dir  /workspace/user_data/output/qwen3_5_0.8B_sample8b/iter_0001816 \
  --save-dir  /workspace/user_data/output/qwen3_5_0.8B_sample8b_hf \
  --model-assets-dir /workspace/user_data/base_hf/Qwen3.5-0.8B-base
```

> ⚠️ `--load-dir` 必须指向**直接包含 `.metadata`** 的目录（训练输出是 `iter_XXXX`）；
> `--model-assets-dir` 必须包含 `model.safetensors.index.json`。
> 转换后该工具写出的分片名与它拷来的 index 不一致，需重命名对齐：
> `model-00001-of-00001.safetensors` → `model.safetensors-00001-of-00001.safetensors`

### 3. 评测

```bash
cd /workspace/user_data
python3 eval_v8.py \
  --model /workspace/user_data/output/qwen3_5_0.8B_sample8b_hf \
  --data sample8b --images sample5 --per-sku 2 \
  --tag v8_s8b --out /workspace/user_data/eval8_s8b.json
```

### 4. 线上形态（属性抽取 + 代码渲染）

```bash
python3 extract_and_render.py \
  --model /workspace/user_data/output/qwen3_5_0.8B_sample8b_hf \
  --data sample8b --images sample5 --limit 20
```

模型只生成第一句属性陈述（`max_new_tokens=48`），正文由确定性模板渲染。

### 便捷启动脚本

| 脚本 | 作用 |
|---|---|
| `examples/qwen3_5/train_qwen35.sh` | 训练入口（通用，接受 `CONFIG` 环境变量） |
| `start_train8b.sh` | 训练 sample8b（约 29 分钟） |
| `convert7.sh` | DCP → HF + 分片名修正 |
| `run_v8_after_train.sh` | 训练完成后自动转权重并评测 |

---

## 代码结构说明

```
MindSpeed/                                  # 仓库根
├── examples/qwen3_5/
│   ├── README.md                           # 本文件
│   ├── train_qwen35.sh                     # 训练入口（FSDP2）
│   ├── convert_weight.sh                   # 官方权重转换脚本
│   ├── qwen3_5_0.8B_config.yaml            # 基础配置
│   ├── qwen3_5_0.8B_sample3_config.yaml    # ┐
│   ├── qwen3_5_0.8B_sample4_config.yaml    # │
│   ├── qwen3_5_0.8B_sample4_e1_config.yaml # │ 各轮数据集配置
│   ├── qwen3_5_0.8B_sample5_config.yaml    # │（逐步演进，见下）
│   ├── qwen3_5_0.8B_sample6_config.yaml    # │
│   ├── qwen3_5_0.8B_sample7_config.yaml    # │
│   └── qwen3_5_0.8B_sample8b_config.yaml   # ┘ 当前交付配置
│
├── mindspeed_mm/                            # 框架本体
│   ├── fsdp/
│   │   ├── train/trainer.py                # 训练入口实现
│   │   ├── models/qwen3_5/                 # Qwen3.5 FSDP2 适配
│   │   └── data/datasets/huggingface/      # 数据集适配（含 image token 处理）
│   └── ...
├── checkpoint/common/merge_dcp_to_hf.py    # DCP → HF 合并工具
│
└── data/ecommerce_multimodal/              # 数据集（图片 + 标签）
    ├── images/                             # 141,931 张原始商品图（gitignore）
    ├── sample5/                            # 原始子样式划分（含 answers.json / questions.json）
    ├── sample6/                            # 颜色 + 自由文本部件
    ├── sample7/                            # 去颜色，自由文本部件走白名单
    └── sample8b/                           # 【交付】闭词表属性 + 散文去色
```

**工作脚本**（本地 `_tunnel/`，同份部署到服务器 `/workspace/user_data/`）：

| 分组 | 文件 | 说明 |
|---|---|---|
| 属性定义 | `attr_spec.py` | 各品类的闭词表互斥判断题（鞋帮高度、开合方式…） |
| | `attr_render.py` | 属性值 ⇄ 自然短语双向映射，保证可精确反解 |
| | `part_vocab.py` | 品类结构词白名单（挡掉背景词、归一化说法） |
| 数据合成 | `compose_v7.py` | 生成 sample7（自由文本部件 + 白名单） |
| | `compose_v8.py` | 生成 sample8（闭词表属性） |
| | `strip_prose_colors.py` | 删除散文里的颜色断言句 → sample8b |
| 评测 | `eval_v7.py` | 换图测试（喂自己的图 vs 喂同子样式另一张） |
| | `eval_v8.py` | 分属性平衡准确率 + 换图测试 |
| | `report_ci.py` | 置信区间与对照基线 |
| 线上 | `extract_and_render.py` | 最小可用流水线 |
| | `render_answer.py` | 确定性答案渲染器 |
| 审计 | `check_full_color.py` / `check_whitelist2.py` / `validate_v8.py` / `usability_audit.py` | 数据与输出质量检查 |
| 数值 | `precision_compare.py` / `diag_repro.py` | 逐步 loss 比对与可复现性诊断 |
| 出图 | `plot_report.py` / `plot_precision.py` / `plot_early.py` | 报告图 |

**报告产物**（本地 `_tunnel/`）：

| 文件 | 内容 |
|---|---|
| `ACCURACY_REPORT.md` | 模型精度报告（含置信区间、对照基线、三个不可引用的数字） |
| `EXP5_gate_report.md` | 完整实验链（门槛实验 → 30B 验证 → 多色方案证伪 → 闭词表路线） |
| `PRECISION_FINDING.md` | 数值精度结论（一致性窗口 + 训练不可复现的证据链） |
| `PERFORMANCE_REPORT.md` | 性能报告（模型侧改动） |

### 数据集演进（为什么需要多轮）

| 数据集 | 视觉断言 | 关键结论 |
|---|---|---|
| sample5 | 子样式共用一段答案 | 不看图也能把 loss 压到 0.03 |
| sample6 | 颜色 + 自由文本部件 | "逐图"答案形成；但目标是欠定的 |
| sample7 | 去颜色 + 白名单部件 | 颜色断言 95.7% → 19.1% |
| **sample8b** | **闭词表属性** + 散文去色 | 颜色断言 **0**；属性目标可精确评测 |

---

## PR 链接

> **状态：待创建。** 本次工作的新增文件仍在工作区**未提交**，因此尚无 PR。

- 仓库：`https://github.com/weizong0321/MindSpeed.git`
- 目标分支：`26.0.0`（当前 HEAD `c1556f4e`，追踪 `origin/26.0.0`）
- 创建 PR：`https://github.com/weizong0321/MindSpeed/pull/new/26.0.0`

**待提交内容**（`git status`）：

```
 M examples/qwen3_5/eval_step1.py
?? examples/qwen3_5/caption_images.py
?? examples/qwen3_5/compose_from_attrs.py
?? examples/qwen3_5/eval_attrs.py
?? data/ecommerce_multimodal/sample5/ … sample8b/
```

提交前请确认 `data/ecommerce_multimodal/images/`、`*.npy`、`assignments.csv`、
`montages/` 等大文件已被 `.gitignore` 排除（当前已排除）。
