# 让远端仓库「clone 即可用」需要补齐的文件清单

> 依据：在已对齐的服务器仓库（`/workspace/MindSpeed`，HEAD = `668d4d44`）上实测，
> 逐个路径检查 `git ls-files --error-unmatch` 与磁盘体积。

## 一、结论速览

| 层级 | 内容 | 体积 | 不补的后果 |
|---|---|---|---|
| **第 1 层（必做）** | 数据集标签 + 人工真值 + 训练配置 | **~3.9 MB** | **训练/评测直接失败** |
| **第 2 层（必做）** | 工作脚本（数据流水线/评测/流水线/数值诊断） | **~250 KB** | 无法重建数据集、无法复核精度 |
| **第 3 层（必做）** | 报告与文档 | **~70 KB** | 无结论可读，等于只给了一堆代码 |
| **第 4 层（需决策）** | `sample5/images` 855 张 | **89 MB** | 训练脚本找不到图片 |
| **第 0 层（最严重）** | **transformers 5.2.0.dev0 本地源码版** | **~数百 MB** | **框架层就缺，跑不起来** |
| 外部（不该进 git） | 基座权重 / 训练产物 / 老师模型 | 20 M + 1.7 G + 2.2 G + 3.2 G + 8.8 G | 需 Release 或对象存储 |

---

## 二、第 0 层：比图片更严重的隐藏依赖（框架层）

**`transformers 5.2.0.dev0` 是本地源码构建，且不在仓库里。**

```
实际使用: 5.2.0.dev0
位置:     /workspace/third_party/transformers_src/src/transformers/
仓库跟踪: git ls-files third_party  ->  0 个文件
```

Qwen3.5 的 `model_type: qwen3_5`、`Qwen3_5ForConditionalGeneration`、
以及 `Qwen3VLMoeForConditionalGeneration`（30B）**都依赖这个版本**。
用 pip 上的正式版 transformers 会因为不认识这些架构而直接失败。

**必须补**（三选一）：
1. 把该源码作为 submodule 或子目录入库（含构建所需文件，**排除 `.git`**）；
2. 或记录确切 commit / 分发地址，写进环境文档，由使用者自行构建；
3. 或打成 wheel 发布到 Release。

> 这是**当前克隆后最致命的缺口**——比缺图片更早触发失败。

---

## 三、第 1 层：数据集与配置（~3.9 MB，必做）

### 3.1 人工真值（14 KB）—— 精度报告的唯一依据

```
✗ data/ecommerce_multimodal/sample5/gate_truth.json            9.0 KB
✗ data/ecommerce_multimodal/sample5/gate_images.json           4.0 KB
✗ data/ecommerce_multimodal/sample5/gate_product_colors.json   1.0 KB
```

这三份是我逐张肉眼标注的 30 张真值。**没有它们，`ACCURACY_REPORT.md` 里
"75.6% vs 51.3%" 就无法复核**，报告变成不可验证的断言。

### 3.2 各轮数据集标签（3.86 MB）

```
✗ data/ecommerce_multimodal/sample6/    1.16 MB（5 个 json）
✗ data/ecommerce_multimodal/sample7/    1.04 MB（5 个 json）
✗ data/ecommerce_multimodal/sample8/    0.85 MB（5 个 json）
✗ data/ecommerce_multimodal/sample8b/   0.82 MB（5 个 json）  ← 交付数据集
```

**`sample8b/` 是最关键的**：交付配置 `qwen3_5_0.8B_sample8b_config.yaml`
里写着 `dataset: ./data/ecommerce_multimodal/sample8b/train.json`，
该文件不存在则训练在数据加载阶段就报错。

另外 `sample5/` 已有 4 个 json 入库（`train/eval/questions/answers`），
但缺上述 3 个 gate 真值。

### 3.3 训练配置（21.7 KB）

```
✗ examples/qwen3_5/qwen3_5_0.8B_sample7_config.yaml          4.5 KB
✗ examples/qwen3_5/qwen3_5_0.8B_sample8b_config.yaml         4.3 KB  ← 交付配置
✗ examples/qwen3_5/qwen3_5_0.8B_sample8b_rep_config.yaml     3.4 KB
✗ examples/qwen3_5/qwen3_5_0.8B_sample8b_eager_config.yaml   3.2 KB
✗ examples/qwen3_5/qwen3_5_0.8B_nostream_config.yaml         3.1 KB
✗ examples/qwen3_5/qwen3_5_0.8B_dettest_config.yaml          3.2 KB
```

`sample3/4/4_e1/5/6` 的配置已在库里，`sample7/8b` 没有——**恰好缺的是交付那一版**。
后 4 个是精度验证用的配置，可选但建议一并入库（否则数值结论无法复现）。

---

## 四、第 2 层：工作脚本（~250 KB，必做）

这些目前**只存在于 `/workspace/user_data/`**（74 个 `.py`，353 KB），
仓库里除 `skill/` 下 4 个副本外没有。需要移入 `examples/qwen3_5/` 后提交：

**数据流水线**
```
attr_spec.py  attr_render.py  part_vocab.py
compose_v7.py  compose_v8.py  strip_prose_colors.py
make_attr_list.py  sample_attr_images.py  caption_images2.py
```

**老师探询与门槛评测**
```
probe_attrs.py  prompt_probe.py  prompt_multi.py  probe_30b.py  gate_eval.py
```

**评测**
```
eval_v7.py  eval_v8.py  attr_agreement.py  report_ci.py  validate_v8.py
usability_audit.py  select_attrs2.py  retrieval_test.py
check_full_color.py  check_whitelist2.py  label_noise2.py  cmp_exact.py
```

**推理流水线与数值诊断**
```
extract_and_render.py  pipeline_audit.py
precision_compare.py  diag_repro.py  extract_loss.py  analyze_divergence.py
plot_report.py  plot_precision.py  plot_early.py
```

> 注意：`attr_spec.py` / `attr_render.py` / `render_answer.py` /
> `extract_and_render_model.py` 已随 `skill/` 入库（4 个副本）。
> 若在 `examples/qwen3_5/` 再放一份会产生**重复代码**，
> 建议改为从 skill 目录引用，或反向让 skill 引用 examples。

---

## 五、第 3 层：报告与文档（~70 KB，必做）

```
✗ docs/ecommerce_sft/README.md              环境/启动/结构/PR（就是前面写的那份）
✗ docs/ecommerce_sft/ACCURACY_REPORT.md     模型精度报告（含置信区间与三个不可引用数字）
✗ docs/ecommerce_sft/EXP5_gate_report.md    完整实验链与结论
✗ docs/ecommerce_sft/PRECISION_FINDING.md   数值精度结论与可复现性证据链
✗ docs/ecommerce_sft/PERFORMANCE_REPORT.md  性能报告
✗ docs/ecommerce_sft/DELIVERY_INDEX.md      交付索引
```

---

## 六、第 4 层：图片（89 MB，需你决策）

```
✗ data/ecommerce_multimodal/sample5/images   855 张   89 MB
```

训练配置里 `dataset_dir: ./data/ecommerce_multimodal/sample5`，
**图片不在仓库里 → 克隆后训练找不到图**。

四个方案：

| 方案 | 仓库增量 | 优点 | 缺点 |
|---|---|---|---|
| **A. 直接入库** | +89 MB（含历史膨胀） | 克隆即可用，最省心 | `.git` 已 744 MB，会到 ~830 MB+ |
| **B. Git LFS** | 指针文件很小 | 仓库干净 | 使用者需装 LFS，且 GitHub LFS 有配额 |
| **C. 提供下载脚本** | 几 KB | 仓库最轻 | 克隆后需联网跑一次脚本 |
| **D. 不放，文档说明** | 0 | — | **克隆不能直接用**，与目标冲突 |

**建议 A 或 B**。考虑到只 89 MB、且目标是"clone 即可用"，A 最直接。

> 附带发现：仓库里还留着 `data/ecommerce_multimodal/images/` 的 **2000 张图（312 MB）**，
> 而本地工作区已把它们标记为删除。这是历史遗留的死重量，
> 建议顺手清理（可省 ~300 MB 仓库体积）。

---

## 七、不该进 git、但克隆后必须另行获取（权重）

| 用途 | 路径 | 体积 | 建议 |
|---|---|---|---|
| 训练元信息 | `/workspace/user_data/base_meta` | 20 MB | 随仓库（体积小） |
| 基座 HF | `/workspace/user_data/base_hf` | **1.7 GB** | Release / 对象存储 |
| 基座 DCP | `/workspace/user_data/dcp` | **2.2 GB** | Release / 对象存储 |
| **交付模型** | `output/qwen3_5_0.8B_sample8b_hf` | **3.2 GB** | **Release**（使用者要跑 Skill 必需） |
| 老师模型 | `shared_assets/.../Qwen3.5-4B` | 8.8 GB | 外部，仅重建数据集时需要 |

建议在 README 里写清"从哪拿、放到哪个路径"，并给一条校验命令。

---

## 八、建议的补齐方式

```bash
cd /workspace/MindSpeed
source /workspace/mindspeed_env.sh

# --- 第 1 层：数据集 + 真值 + 配置 ---
git add data/ecommerce_multimodal/sample5/gate_truth.json \
        data/ecommerce_multimodal/sample5/gate_images.json \
        data/ecommerce_multimodal/sample5/gate_product_colors.json
git add data/ecommerce_multimodal/sample6 data/ecommerce_multimodal/sample7 \
        data/ecommerce_multimodal/sample8 data/ecommerce_multimodal/sample8b
git add examples/qwen3_5/qwen3_5_0.8B_sample7_config.yaml \
        examples/qwen3_5/qwen3_5_0.8B_sample8b_config.yaml \
        examples/qwen3_5/qwen3_5_0.8B_sample8b_rep_config.yaml \
        examples/qwen3_5/qwen3_5_0.8B_sample8b_eager_config.yaml \
        examples/qwen3_5/qwen3_5_0.8B_nostream_config.yaml \
        examples/qwen3_5/qwen3_5_0.8B_dettest_config.yaml

# --- 第 2 层：脚本（先从 user_data 移入 examples/qwen3_5/）---
# （见第四节清单，建议先统一改掉硬编码的 /workspace/user_data 路径）

# --- 第 3 层：文档 ---
git add docs/ecommerce_sft/

# --- 第 4 层（二选一）---
git add data/ecommerce_multimodal/sample5/images          # 方案 A
# 或 git lfs track "data/ecommerce_multimodal/sample5/images/**" && git add ...  # 方案 B

git commit -m "feat(ecommerce-sft): 补齐数据集/配置/脚本/文档，使 clone 即可复现"
git push
```

---

## 九、克隆后自检清单（建议写进 README）

```bash
git clone https://github.com/weizong0321/MindSpeed.git && cd MindSpeed

# 1) 框架版本（最易踩坑）
python -c "import transformers; print(transformers.__version__)"   # 需 5.2.0.dev0
python -c "import torch_npu; print(torch_npu.__version__)"         # 需 2.7.1

# 2) 数据集完整性
ls data/ecommerce_multimodal/sample8b/{train,eval,attrs_truth}.json
ls data/ecommerce_multimodal/sample5/images | wc -l                # 需 855

# 3) 人工真值
ls data/ecommerce_multimodal/sample5/gate_truth.json

# 4) 权重（需另行下载）
ls /workspace/user_data/base_meta/Qwen3.5-0.8B
ls /workspace/user_data/dcp/Qwen3.5-0.8B

# 5) 冒烟：一次训练能否起步
CONFIG=examples/qwen3_5/qwen3_5_0.8B_sample8b_config.yaml NPUS_PER_NODE=1 \
  bash examples/qwen3_5/train_qwen35.sh
```

**当前状态下，第 1 步（框架版本）和第 2、3、4 步都会失败。**
补齐第 1–3 层（约 4.2 MB）+ 决策图片后，才能真正做到 `git clone` 即可用。
