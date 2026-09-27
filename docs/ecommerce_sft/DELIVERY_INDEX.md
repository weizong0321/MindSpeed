# 交付件索引

| 编号 | 交付项 | 文件 | 规格 |
|---|---|---|---|
| 5 | 原始日志（.log 压缩包） | `5_logs.zip` | 68 条目 / 1.62 MB；含索引 `INVENTORY.md` |
| 5 | 原始日志（PDF 文档） | `5_logs.pdf` | 141 页 / 0.47 MB |
| 6 | Skill 文件夹 | `6_skill.zip` | 7 条目 / 16.7 KB |
| 7 | 源代码（Python） | `7_source.zip` | 80 条目 / 174 KB |

---

## 5. 原始日志

- **时间跨度**：从最早的环境初始化日志到本次最终实验（`train_sample8b_rep` / `nostream_C`），共 67 个 `.log`。
- **压缩包内容**：`logs/` 下保留全部原始文本，**未做任何删减**；`logs/INVENTORY.md` 为按修改时间排序的索引，并附「日志 ↔ 实验环节」对应表。
- **PDF 的取舍**（封面已注明）：
  - 删除纯进度噪声行（tqdm 的 `it/s`、`Materializing param=` 等）
  - 超长日志保留首 90 行 + 末 60 行，中间标注省略行数
  - 每段日志标注实际过滤行数，便于与 zip 中的原文核对
- 覆盖的实验链：权重分片补齐 → 30B 加载修复 → 4B/30B 属性标注 → 门槛实验 → 换问法实验 → 多色方案证伪 → sample7/sample8b 训练 → 换图测试 → 精度比对与可复现性诊断。

## 6. Skill 文件夹

`skill/ecommerce_guide_skill/`

```
SKILL.md                        能力说明、实测精度、能力边界与已知失败项
skill.yaml                      Skill 元信息（含 metrics / limitations）
scripts/run_chat.py             交互式入口（真正加载模型；旧的占位实现已替换）
scripts/extract_and_render_model.py   属性读取器（加载 → 生成 → 反解 → 渲染）
scripts/attr_spec.py            各品类闭词表属性定义
scripts/attr_render.py          属性值 ⇄ 自然短语双向映射
scripts/render_answer.py        确定性答案渲染器
```

核心设计：**模型只读属性，文案由代码渲染**。依据是本项目的实测结论——
颜色跨标注者一致率仅 0.67–0.73（换 30B 无改善），自由文本特征仅 0.133（目标欠定），
而闭词表互斥属性达 0.85–1.00。因此 Skill 不对颜色/材质/尺码等做断言。

## 7. 源代码

`source/` 按功能分组，含 `README.md`（环境、运行前置、关键实现要点）：

| 目录 | 内容 |
|---|---|
| `01_attribute_schema/` | 闭词表属性定义、可逆渲染、结构词白名单 |
| `02_dataset_build/` | 答案合成（sample7 / sample8b）、散文去色、标注清单 |
| `03_teacher_probing/` | 老师模型属性探询、换问法实验、门槛评测 |
| `04_evaluation/` | 平衡准确率、置信区间、换图测试、数据与输出审计（25 个脚本） |
| `05_inference_pipeline/` | 两段式推理流水线、渲染器、可用性审计 |
| `06_precision/` | 逐步 loss 比对、可复现性诊断、日志曲线抽取 |
| `07_plotting/` | 报告出图脚本 |
| `08_configs/` | 训练配置（含交付配置 `qwen3_5_0.8B_sample8b_config.yaml`）与启动脚本 |

---

## 交付环境

| 项 | 值 |
|---|---|
| CANN | 8.5.0 |
| torch_npu | 2.7.1 |
| PyTorch | 2.7.1+cpu |
| Python | 3.10.12 |
| transformers | 5.2.0.dev0 |
| 硬件 | Ascend 910B2 × 1（HBM 64 GB） |

## 交付模型

`qwen3_5_0.8B_sample8b_hf`（由 `merge_dcp_to_hf.py` 从 `iter_0001816` 转换）

- 训练数据：570 张商品图 / 908 条问答 / 闭词表属性目标
- 整组属性全对 **75.6%**（95% CI 65–84%），对照不看图 **51.3%**（40–62%）
- 颜色断言率 **0**
- 覆盖 11 / 21 个品类
