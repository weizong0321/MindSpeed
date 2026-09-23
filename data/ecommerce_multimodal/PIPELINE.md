# 数据产线说明（142k 无标注电商图 → 可训练数据集）

> 本文件记录从「14 万张无标注商品图」到「可训练 SFT 数据集」的完整流程、脚本清单、
> 每一步的实测数字，以及已经踩过的坑。

---

## 0. 一句话概览

```
141,931 张无标注图（文件名纯序号 1.jpg…141931.jpg）
    ↓ 审计 + 去重
133,174 张可用（93.8%）
    ↓ ResNet50 特征 + k=200 聚类 + 逐簇命名
33 个品类（其中 7 个是「杂项桶」）
    ↓ 按保留清单过滤
85,575 张 / 26 个干净品类（60.3%）
    ↓ 品类内再聚类
3,457 个「视觉子样式」
    ↓ 取 3 个品类 × 10 个子样式 × ~13 张
样板：30 个子样式 / 386 张 / 628 训练条 + 144 验证条
```

---

## 1. 关键发现（会影响后续所有决策）

1. **这是一批泛电商杂货图**，不是单一品类：服装、鞋、箱包、手表、玩具、零食、
   个护美妆、清洁纸品、药品保健、手机数码、家电、家纺、珠宝。
2. **文件名零信息**，类别只能靠模型或人工得出。
3. **同一个品类散落在多个簇里**（裤子跨 3 簇、鞋跨 6 簇、零食跨 6 簇）。
   只看单个簇的规模会严重低估品类规模 —— 必须按品类合并簇。
4. **聚类给出的是「视觉子样式」，不是「单一商品」**。
   ResNet50(ImageNet) 特征能把「黑色数字运动表」聚成一堆（纯度 0.89），
   但也会把「各种盒子形状的零食」聚成一堆（纯度 0.685）。
   要切出真正的「同款」需要实例级检索能力（DINOv2 / CLIP）。
5. **按款全量标注不现实**：85k 张图 → 1,987（per-sku 40）到 3,457（per-sku 15）个款，
   即 2000–3500 段答案。可行的是**取子集**：
   - 每品类 10 个子样式 × 15 张 ≈ 260 段答案 / 3,900 张图 / ~22 小时
   - 或先做 3 个品类样板：30 段答案 / 386 张图 / ~2.5 小时

---

## 2. 脚本清单（按执行顺序）

| # | 脚本 | 作用 | 实测耗时 |
|---|---|---|---|
| 1 | `scan_dataset.py` | 数据审计：尺寸分布、可用率、感知哈希去重 | 约 12 分钟 / 142k |
| 2 | `embed_cluster.py` | 视觉嵌入 + 聚类（GPU） | 提特征 11.8 分钟（200 张/秒）+ 聚类 8.7 分钟 |
| 3 | `make_index_sheet.py` | 把聚类结果汇成总览图，供命名 | 数十秒 |
| 4 | `build_cluster_labels.py` | 标注 → 标签文件，并对 clusters.csv 做独立校验 | 秒级 |
| 5 | `summarize_classes.py` | 按品类合并簇、算真实规模、出保留清单 | 秒级 |
| 6 | `select_dataset.py` | 按 `keep_categories.json` 过滤图片 | 秒级 |
| 7 | `cluster_within.py` | 品类内再聚类，分出「款/子样式」 | 约 4 分钟 / 85k |
| 8 | `build_sample_dataset.py` | 构建 3 品类样板（选样式→复制→评审图→骨架） | 约 1 分钟 |
| 9 | `generate_dataset.py` | 由 questions.json + answers.json 生成 train/eval，带质检 | 秒级 |
| 10 | `classify_tool.html` + `apply_labels.py` | 人工归类工具与落地脚本（品类/款两级） | — |

**回归测试**：`node tool_test.js`（归类工具，17 项检查）

### 典型执行序列

```bash
# ① 审计
python scan_dataset.py --root <图片目录> --report scan_report.json --csv image_sizes.csv

# ② 提特征 + 聚类（特征会缓存，后续换 k 重聚只要几十秒）
python embed_cluster.py --root <图片目录> --k 200 --pca 192 \
    --cache embeddings.npy --montage-dir montages --montage-clusters 60

# ③ 出总览图并命名（人工或模型读图）
python make_index_sheet.py --csv clusters.csv --out sheet_top20.png --from-rank 1 --to-rank 20
#   命名结果写入 cluster_categories.json 后：
python build_cluster_labels.py          # 校验 + 转格式
python summarize_classes.py --min-images 2000

# ④ 过滤
python select_dataset.py --manifest kept_images.csv

# ⑤ 品类内分款
python cluster_within.py --manifest kept_images.csv --per-sku 15

# ⑥ 构建样板
python build_sample_dataset.py --src <图片目录> --dst sample3 --per-sku 15 --copy

# ⑦ 生成训练数据
cd sample3 && python ../generate_dataset.py --root . --no-size-check
```

---

## 3. 文件说明

### 中间产物（可重新生成，不必入库）

| 文件 | 大小 | 说明 |
|---|---|---|
| `embeddings.npy` | 582 MB | 142k × 2048 的 fp16 特征，**换 k 重聚的关键缓存** |
| `clusters.csv` | — | 逐图聚类结果（path, cluster, rank, purity），**行序 == embeddings 行序** |
| `image_sizes.csv` | — | 每张图的尺寸明细 |
| `montages/`、`sku_montages/` | — | 簇/款拼图 |
| `sheet_*.png` | — | 品类总览图 |

### 结论性产物（应入库）

| 文件 | 说明 |
|---|---|
| `scan_report.json` | 审计结论 |
| `cluster_categories.json` | 200 个簇的原始标注（rank/cluster/size/品类/置信度） |
| `cluster_labels.json` | 规范化后的簇标签（按 cluster id 索引 + 品类级中文名） |
| `keep_categories.json` | **保留清单（26 品类）+ 丢弃清单（7 杂项桶）+ 备注** |
| `kept_images.csv` | 保留的 85,575 张（path, cluster, category, purity） |
| `dropped_images.csv` | 丢弃的 56,356 张 + 原因 |
| `assignments.csv` | 逐图 → (品类, 子样式) 分配 |
| `sku_summary.json` | 3,457 个子样式的规模/纯度/代表图 |
| `DATA_SPEC.md` | **标注规范**（图片要求、标注四步法、答案写作规范、质检项） |
| `sample3/` | **3 品类样板**（386 张图 + questions/answers + train/eval） |

---

## 4. 数据质量约束（来自训练配置）

`examples/qwen3_5/qwen3_5_0.8B_config.yaml`：`image_max_pixels: 262144`、`image_min_pixels: 1024`，
配合 `patch_size=14 × merge_size=2 = 28`：

- 面积 > 262144（=512×512）会被**等比压缩** → 多传的分辨率无效（本数据集 91.4% 属此列）
- 短边 < 336 会被**放大** → 细节丢失，建议淘汰（本数据集 5.3% 属此列）

---

## 5. 已修掉的数据缺陷（踩坑记录）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 品类级聚类把女单鞋、高跟鞋分进了「运动鞋」 | 样板构建时加 `exclude` 名单并替换 |
| 2 | 「背包」品类里混进 5 个斜挎包/手提包 | 同上，替换为双肩包 |
| 3 | `clusters.csv` 的簇编号靠肉眼看图抄会错 | 改为从 CSV 精确统计并独立校验（200 簇 / 张数之和 141,931 全部吻合） |
| 4 | `generate_dataset.py` 数字质检假阳性（问句里的 "15.6寸"、"300"、"5公里"） | 从待核对集合中剔除问句里出现过的数字 |
| 5 | Windows 控制台 GBK 导致打印崩溃/乱码 | 脚本统一 `sys.stdout.reconfigure(encoding="utf-8")` |
| 6 | torch 权重下载被沙箱拦（写到工作区外） | `TORCH_HOME` 指到工作区内 |

---

## 6. 已知限制

1. **聚类 ≠ 同款切分**。要实例级精度需换 DINOv2/CLIP 重提特征（约 10–20 分钟）。
2. **`unclear` 簇 13,766 张（9.7%）** 是纯度 0.46–0.61 的杂图，已丢弃。
3. **30 个簇置信度 < 0.5**（19,869 张）未人工复核。
4. **样板答案的 `visible` 是依据每个子样式 2 张代表图起草的**，
   每个子样式实际有 8–15 张图，特征不一定全部成立 → `needs_review: true` 标记待复核。

---

## 7. 当前状态

- ✅ 全量数据已审计、已聚类、已命名、已按保留清单过滤
- ✅ 26 个品类 / 85,575 张 可用；7 个杂项桶 / 42,590 张 与 unclear / 13,766 张 已丢弃
- ✅ 3 品类样板已生成并可训练（`sample3/`，628 训练条 / 144 验证条）
- ⬜ 样板尚未推送、尚未在昇腾机器上训练验证
- ⬜ 26 品类的完整答案（约 260 段）尚未撰写
- ⬜ 待定：是否换 DINOv2 重跑特征以获得更好的「同款」切分
