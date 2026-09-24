# 交接说明：Step 2 训练已完成，评测待跑

> 状态时间：2026-09-24 06:25（服务器时间）
> 本文件记录当前进度、恢复方式、以及已知的坑。**评测还没跑** —— 用户要求训练结束后先收尾。

---

## 1. 已完成

| 项 | 状态 |
|---|---|
| 环境 | 新容器 `640e06bb92f2` / IP `179.42.54.110`；**NPU 910B4，HBM 32GB**（此前 910B2/64GB） |
| 隧道 | 本机 `127.0.0.1:2299`（后台任务 `pwsh-3`），带 `tar_put` 直传能力，**不依赖 GitHub** |
| 你的人工判读 | 200 个候选里 **57 个"纯"**（28.5%） |
| 子代理复核 | 发现其中 **10 个仍混了多款商品**（最严重：`shoes_leather_008` 实为童鞋归到皮鞋类） |
| 最终数据集 | `sample5`：**47 个子样式 / 21 个品类 / 705 张图 / 1128 训练 + 282 验证**，全部带答案 |
| 训练 | **已完成**，2736 iter（2 epoch），0.69 s/iter，峰值 HBM **23.9/32 GB**，无 OOM |
| 检查点 | `/workspace/user_data/output/qwen3_5_0.8B_sample5/iter_0001368`、`iter_0002736` |

### 训练损失轨迹（与 sample3/sample4 同一模式）

| 阶段 | loss | grad norm |
|---|---|---|
| iter 1 | 3.98 | 80 |
| epoch 1 末（1128） | ~0.03 | ~10 |
| epoch 2 末（2736） | 0.40 | 92 |

epoch 1 就把训练集背下来了（loss→0.03），epoch 2 有轻微退化。
但 sample4 的实验已证明**1 个 epoch 不够**（1-epoch 模型串味 14 条、输出语无伦次），
所以这次仍用 2 epoch。

---

## 2. 待办：只剩一步（约 15 分钟）

### 2.1 转 HF

```bash
source /workspace/mindspeed_env.sh && cd /workspace/MindSpeed
mm-convert Qwen35Converter dcp_to_hf \
  --dcp_dir /workspace/user_data/output/qwen3_5_0.8B_sample5/iter_0002736 \
  --save_hf_dir /workspace/user_data/output/qwen3_5_0.8B_sample5_hf \
  --origin_hf_dir /workspace/user_data/output/qwen3_5_0.8B_hf
```

注意：`--dcp_dir` 必须指向**直接含 `.metadata` 的目录**（训练产出就是 `iter_xxxx`）。
`--origin_hf_dir` 必须有 `model.safetensors.index.json`（它决定写哪些 key 到哪个分片），
所以只能用 `qwen3_5_0.8B_hf`，不能用只有 config 的 `base_meta`。

### 2.2 评测（BASE 与 FT 各一轮）

```bash
# BASE 对照
python examples/qwen3_5/eval_step1.py \
  --model /workspace/user_data/base_hf/Qwen3.5-0.8B-base \
  --data sample5 --tag base5 --out /workspace/user_data/step2_base.json

# sample5 微调
python examples/qwen3_5/eval_step1.py \
  --model /workspace/user_data/output/qwen3_5_0.8B_sample5_hf \
  --data sample5 --tag ft5 --out /workspace/user_data/step2_ft.json
```

**必须后台跑**：282 条验证样本 × 2 个模型，每条生成约 1.2 s，单模型约 6 分钟，
而工具调用有 10 分钟上限。用 `nohup ... > log 2>&1 &`，然后
`tr '\r' '\n' < log | grep -av "Loading weights" | tail -20` 看结果。

### 2.3 对照表（要看的结论）

| 模型 | A 组间 | B 组内 | **B−A** | C 串味 | D 归属 |
|---|---|---|---|---|---|
| BASE（sample4 上） | 0.147 | 0.172 | +0.025 | 0 | 0.214 |
| sample4 FT（14 子样式） | 0.350 | 0.573 | **+0.223** | **6** | 0.417 |
| **sample5 FT（47 子样式）** | 待测 | 待测 | 待测 | 待测 | 待测 |

**判据**：如果 **D 明显上升、C 降到 0**，说明"纯度 + 数量"这条路成立，可以往 26 品类铺开。
如果 D 没上去，说明瓶颈不在数量，应转向**大模型蒸馏**（见第 4 节）。

---

## 3. 恢复工作时的重要提醒

### 3.1 隧道

- 隧道进程是本机后台任务 `pwsh-3`，监听 `127.0.0.1:2299`。
- 用法：`python rsh.py exec --file <脚本>` / `python rsh.py tar_put <本地目录> <远端父目录>`
- **token 的 direct-tcpip 授权次数有限（每 token 2–5 次），但已建立的通道不受 token 过期影响。**
  重启隧道需要新 token：让用户给新的 `ssh -J jt_xxx:TOKEN@113.47.8.48:2234 root@<IP>` 和密码，
  写进 `_tunnel/creds.json` 即可（其他不用改）。
- 若要彻底清理：结束后台任务 `pwsh-3`。

### 3.2 服务器到 GitHub 的 `git pull` 不稳定

实测超时 129 s。**小文件改 heredoc 直投**（见 `gen_deploy*.py` 的写法），
大数据用 `tar_put`。`eval_step1.py` 和 sample5 的配置文件都是这么投递的。

### 3.3 致命坑：`pkill -f` 会杀掉承载脚本自身

脚本内容就在 bash 命令行里，所以 `pkill -f "eval_step1[.]py"` 会匹配到自己，
实测把评测进程和它一起杀了（返回 `EXIT -1`）。
**要么按 PID 杀，要么用 `$(printf 'eval_step%s' '_1.py')` 运行时拼 pattern。**

### 3.4 评测脚本已改进

- 新增 `--data <数据集名>`，可对 sample4/sample5 复用同一套判据。
- 串味检测（判据 C）已从硬编码 3 个品类改成**自动检测**：从每类参考答案抽"专属 2-gram"，
  输出里若大量命中别的品类的专属短语就判串味。**对任意品类数都有效。**

---

## 4. 如果 Step 2 结论不理想，下一步是「大模型蒸馏」

服务器上有 **`Qwen3-VL-30B-A3B-Instruct`**（`/workspace/shared_assets/models/Qwen/Qwen3-VL-30B-A3B-Instruct`）。
思路：用它给每张图生成"只描述图上可见内容、不编造参数"的标注 → 人工抽检 50 张 →
蒸馏到 0.8B。**这条路能一次性解决"逐图标注"的成本问题**，也是我们认为的最终解法。

为什么需要它：本项目的核心矛盾是"一个子样式的 10–15 张图共用一段答案，
模型不看图也能把 loss 压到 0"。要根治就得**逐图不同的答案**，而人工逐图写不可扩展。

---

## 5. 本地与服务器上的资产位置

| 内容 | 位置 |
|---|---|
| 本地工作区（DSH） | `data/ecommerce_multimodal/`：sample3、sample4、sample5、全部脚本、`cand_sheets/`、`ans_sheets/` |
| 本地 git 仓库 | `C:\Users\wqy\Downloads\MindSpeed-MM`（sample4 及之前已推 GitHub；**sample5 与本次报告未入库**） |
| 服务器 | `/workspace/MindSpeed/data/ecommerce_multimodal/sample5/`（705 张图 + 4 个 JSON，已校验 0 缺失） |
| 服务器权重 | DCP `/workspace/user_data/dcp/Qwen3.5-0.8B`；base HF `/workspace/user_data/base_hf/Qwen3.5-0.8B-base`；产物 `/workspace/user_data/output/` |

### 数据集演进一览

| 数据集 | 子样式 | 品类 | 图 | 训练条 | 关键变化 |
|---|---|---|---|---|---|
| sample3 | 30（按聚类纯度选） | 3 | 386 | 628 | 子样式混多款商品 → 标签只对部分图成立 |
| sample4 | 14（人工核验） | 3 | 175 | 266 | 纯度修复 → B−A 从 +0.025 升到 +0.223 |
| sample5 | 47（人工核验+子代理复核） | 21 | 705 | 1128 | 子样式数量 ×3.4 → **待测** |

## 6. 需要人工复核的 10 个子样式

这些是子代理判定"仍混多款商品"而剔除的（`needs_review=true`），
建议人工确认后再决定是否拆分或丢弃：

| 子样式 | 问题 |
|---|---|
| `bag_backpack_007` | 白底商品图款与深蓝/黑色实拍款不一致 |
| `bag_handbag_007` | 棕色花纹托特、黑色手提、玫红小包三种包型 |
| `purse_wallet_008` | 前两张是钱包，第三张是金属铆钉单肩小包 |
| `shoes_leather_006` | 白帆布鞋 vs 绿色后跟皮质小白鞋 |
| `shoes_leather_007` | 系带德比 vs 一脚蹬乐福 |
| `shoes_leather_008` | **实为儿童魔术贴运动鞋，品类错配到皮鞋类** |
| `watch_004` | 简洁三针金属带 vs 计时小表盘赛车风款 |
| `luggage_007` | 亮黄绿 vs 暗红两款硬壳箱 |
| `cosmetics_006` | 盒装瓶身但印刷/盒型不同，按图更像香氛或护肤盒 |
| `health_supplement_007` | 第一张是按压泵瓶（印 MOISTURE BODY LOTION），后两张是软胶囊瓶 |
