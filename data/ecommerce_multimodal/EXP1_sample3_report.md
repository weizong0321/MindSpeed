# 实验一：3 品类样本 → 昇腾训练 → 推理对照

**结论先行：管线全通；格式与"承认看不到"学到位；但"看图"只学会了很浅的一层 ——
差异大的子样式（银表 vs 黑 G-SHOCK）能正确区分，差异小的（白跑鞋 vs 黑帆布鞋）
被合并成同一个答案，且跨品类词汇会串味。**

---

## 1. 实验设置

| 项 | 内容 |
|---|---|
| 数据 | `data/ecommerce_multimodal/sample3`：3 品类 / 30 子样式 / 386 张图 |
| 训练集 / 验证集 | 628 条 / 144 条（按款整款划分） |
| 起点权重 | 原始 base 的 DCP（`user_data/dcp/Qwen3.5-0.8B`，由原始 0.8B 权重转换而来） |
| 配置 | `examples/qwen3_5/qwen3_5_0.8B_sample3_config.yaml` |
| 超参 | mbs=1、lr=1e-5 cosine、train_iters=1260（约 2 epoch）、freeze=model.visual、bf16 |
| 硬件 | 单卡 Ascend 910B2，约 620 ms/iter，总耗时约 20 分钟 |

> 注意：原始 Qwen3.5-0.8B 的 HF 权重已从 `shared_assets` 移除。
> 本次用原始 base 的 DCP 作为起点，并用只含 config/tokenizer/processor 的
> `base_meta` 目录做 meta init，避免误加载此前微调过的权重。

## 2. 训练曲线（值得注意）

| 阶段 | 现象 |
|---|---|
| epoch 1（iter 1–628） | loss 从 4 降到 **5e-3**，grad norm ~1–5 |
| epoch 1/2 交界（iter 629+） | loss 突跳到 2.8，**grad norm 从 ~2 飙到 60–120** |
| epoch 2（iter 629–1260） | loss 稳定在 0.3–1.0，grad norm 维持 60–100 |

epoch 1 就把 loss 压到 5e-3 = **已经背下来了**。原因有两个：
628 条样本中，同一子样式的 10–15 张图共用**完全相同的一段答案**，
所以模型不需要看图也能把 loss 压到接近 0。这正是设计上最大的隐患。

## 3. 推理对照（12 个用例，BASE vs FT）

### 3.1 BASE（原始 base）表现

- 输出又长又自信，**大量幻觉**：把白色系带运动鞋认成 "Adidas Superstar（美极星）"、
  "Nike Air Max 1"，编造缓震技术名称。
- 给出荒唐建议，例如对一双本来就全白的鞋说"**建议将鞋面改为白色或浅色**"。
- 无固定格式，像聊天而不像导购。
- 组内字符重合度 0.27–0.43。

### 3.2 FT 表现

**学到的（真实、可复现）**

1. **固定输出结构**：`结论 → 图中为… → 图中看不到、以商品页标注为准 → 短板是…`
2. **不再编造品牌与参数**。BASE 会编 "Adidas Superstar"，FT 一律说
   "缓震配置、鞋重与耐磨性图中看不到，需以商品页标注为准"。
   **这是本次实验最有价值的收获** —— 正是针对旧数据集"背参数"失败模式设计的机制。
3. **对手表能真正看图**：
   - `watch_127`（银色米兰尼斯带、白盘、条形时标、无日期窗）→
     "图中为银色金属表带配银色表盘，条形时标、三针……图中没有可见的日期窗或夜光标记"
   - `watch_086`（黑色树脂壳、蓝紫双显、多按键）→
     "图中为黑色树脂表壳配蓝紫色双显表盘，可同时看指针和数字，表壳两侧多按键、金属表圈"
   两张图 → 两段准确且不同的描述。
4. 生成长度可控（修正 eos 之后不再拼接多份答案）。

**没学到的（同样是真实、可复现）**

1. **跨品类词汇串味**：
   - 背包答案里出现"**深色表盘**不耐脏"（手表词）
   - 背包答案里出现"低帮""鞋面平整""好搭配校服""鞋重"（鞋词）
   - 手表答案里出现"正面有拉链袋方便分装小物"（背包词）
   - 运动鞋答案开头是"**日跑可以背**"（"背"用于鞋）
2. **差异小的子样式被合并**：`shoe_031`（白色 Adidas）与 `shoe_080`（黑白帆布鞋）
   的回答字符重合度 **0.91** —— 黑白帆布鞋被描述成"白色系带运动鞋"。
3. 组内重合度整体比 BASE **升高**（0.27–0.43 → 0.45–0.91），答案更同质化。

## 4. 根因分析

| # | 根因 | 说明 |
|---|---|---|
| 1 | **答案复用过度** | 一个子样式的 10–15 张图共用一段答案 → loss 可以在不看图的情况下压到 0 |
| 2 | **答案总数太少** | 全数据集只有 30 段答案、628 条样本，0.8B 模型足以记住骨架 + 少量名词 |
| 3 | **答案骨架完全共享** | 所有答案都是同一句式，模型学到"骨架 + 名词池"，名词按问法/品类而非图片选择 |
| 4 | **子样式视觉差异不足** | 选样时按聚类纯度取 top10，导致部分子样式彼此很像（白跑鞋 vs 黑帆布鞋） |

## 5. 下一步建议（按性价比排序）

1. **每张图给独立答案，而不是每个子样式一段**
   哪怕只是把"颜色/有无鞋带/鞋底厚薄"这些图上可见项按图写出来，也能强迫模型看图。
   代价：答案数从 30 涨到几百，但这是**唯一能根治"不看图"的办法**。
2. **扩大答案总量到 200–300 段**（26 品类 × 8–12 子样式），并让相邻子样式的答案差异更大。
3. **选子样式时不要只看纯度**，要用 review 图人工剔除"彼此太像"的。
4. **提高图片分辨率上限**（当前 262144 像素 = 512×512，很多细节如品牌标识、材质纹理看不清），
   或对需要细看的品类单独提高 `image_max_pixels`。
5. 训练侧可试：加 `--per-sku` 更小粒度、增加 epoch 数、或对同图不同问法做更强的配对对比。

## 6. 复现命令

```bash
# 训练
CONFIG=examples/qwen3_5/qwen3_5_0.8B_sample3_config.yaml bash examples/qwen3_5/train_qwen35.sh

# DCP -> HF
mm-convert Qwen35Converter dcp_to_hf \
  --dcp_dir /workspace/user_data/output/qwen3_5_0.8B_sample3/iter_0001260 \
  --save_hf_dir /workspace/user_data/output/qwen3_5_0.8B_sample3_hf \
  --origin_hf_dir /workspace/user_data/output/qwen3_5_0.8B_hf

# 原始 base 的 DCP -> HF（对照用）
mm-convert Qwen35Converter dcp_to_hf \
  --dcp_dir /workspace/user_data/dcp/Qwen3.5-0.8B/release \
  --save_hf_dir /workspace/user_data/base_hf/Qwen3.5-0.8B-base \
  --origin_hf_dir /workspace/user_data/output/qwen3_5_0.8B_hf

# 推理对照
python examples/qwen3_5/infer_sample3.py --model <HF目录> --tag <base|ft> --out <json>
```

### 踩坑记录
- `dcp_to_hf --dcp_dir` 必须指向**直接含 `.metadata` 的目录**（base 是 `.../release`，
  训练产出是 `.../iter_0001260`）。
- `--origin_hf_dir` 只用于 config/tokenizer/资产，但**必须有 `model.safetensors.index.json`**
  （它决定写哪些 key 到哪个分片），所以只放 config 的 `base_meta` 会报错。
- 推理时必须把 `<|im_end|>`(248046) 加进 `eos_token_id`，否则 tokenizer 默认 eos(248044)
  不匹配，模型生成完不停止、把多份答案拼在一起。
