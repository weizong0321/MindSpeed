#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample8 的评测：闭词表属性能不能从图里读出来。

为什么不能用普通准确率
    上一轮发现平均多数类基线高达 0.807 —— 全答最常见值就能拿 80%。
    所以这里同时报：
      原始准确率      容易虚高
      平衡准确率      每个取值算 recall 再取平均，不受类别偏斜影响
      多数类基线      作为对照
    只有"平衡准确率明显高于基线"才说明模型真的在读属性。

另外做换图测试（和 eval_v7 一致）：喂自己的图 vs 喂同子样式的另一张，
看解析出的属性更匹配哪一张。这直接回答"有没有看图"。
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from attr_render import parse_values  # noqa: E402

import torch
from PIL import Image
from transformers import AutoProcessor
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


BASE = str(MM_DATA)


def clause_of(answer):
    m = re.search(r"图中为(.*?)。", answer)
    return m.group(1).strip() if m else None


def load_model(model_dir, device):
    from transformers import AutoConfig
    import transformers
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    cls = getattr(transformers, arch, None) or \
        transformers.AutoModelForImageTextToText
    big = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    if big > 10e9:
        return cls.from_pretrained(model_dir, trust_remote_code=True,
                                   dtype=torch.bfloat16,
                                   low_cpu_mem_usage=True,
                                   device_map="auto").eval()
    try:
        m = cls.from_pretrained(model_dir, dtype=torch.bfloat16,
                                trust_remote_code=True)
    except TypeError:
        m = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                trust_remote_code=True)
    return m.eval().to(device)


@torch.no_grad()
def ask(model, proc, eos, path, question, device, max_new=200):
    img = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": question}]}]
    try:
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
    except TypeError:
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt")
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=eos, pad_token_id=eos[0])
    return proc.decode(out[0][inp["input_ids"].shape[1]:],
                       skip_special_tokens=True).strip()


def spec_for(cat, selected_spec):
    """把 attr_selected_855.json 的 {cat: [attr,...]} 转成 parse_values 需要的格式"""
    return [{"key": k, "options": None} for k in (selected_spec.get(cat) or [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="sample8")
    ap.add_argument("--images", default="sample5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sku", type=int, default=2)
    a = ap.parse_args()

    d = Path(BASE) / a.data
    img_root = Path(BASE) / a.images
    evalset = json.loads((d / "eval.json").read_text("utf-8"))
    truth = json.loads((d / "attrs_truth.json").read_text("utf-8"))
    selected = json.loads((d / "attr_spec.json").read_text("utf-8"))

    # parse_values 需要选项列表；从 attr_spec.ATTRS 取全量选项
    import attr_spec
    full_spec = {}
    for cat, attrs in selected.items():
        m = {k: o for k, _, o in attr_spec.ATTRS.get(cat, [])}
        full_spec[cat] = [{"key": k, "options": m.get(k, [])} for k in attrs]

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")

    print(f"[{a.tag}] {a.model} -> {device}", flush=True)
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = proc.tokenizer
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                  tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos = [i for i in eos if isinstance(i, int) and i >= 0] or [tok.eos_token_id]
    model = load_model(a.model, device)

    by_sku = defaultdict(list)
    for s in evalset:
        by_sku["/".join(s["image"].split("/")[1:3])].append(s)

    rows, t0 = [], time.time()
    for sku, items in sorted(by_sku.items()):
        cat = sku.split("/")[0]
        spec = full_spec.get(cat)
        if not spec:
            continue
        imgs = sorted({s["image"] for s in items})
        if len(imgs) < 2:
            continue
        sel = imgs[:a.per_sku]
        for t, img in enumerate(sel):
            other = imgs[(t + 1) % len(imgs)]
            q = next(s["conversations"][0]["value"] for s in items
                     if s["image"] == img)
            a_own = ask(model, proc, eos, img_root / img, q, device)
            a_swp = ask(model, proc, eos, img_root / other, q, device)
            c_own, c_swp = clause_of(a_own), clause_of(a_swp)
            p_own = parse_values(cat, c_own or "", spec)
            p_swp = parse_values(cat, c_swp or "", spec)
            g_own = (truth.get(img) or {}).get("attrs") or {}
            g_oth = (truth.get(other) or {}).get("attrs") or {}
            rows.append({
                "sku": sku, "cat": cat, "image": img, "other": other,
                "a_own": a_own, "a_swap": a_swp,
                "pred_own": p_own, "pred_swap": p_swp,
                "gold_own": g_own, "gold_other": g_oth,
                "clause_own": c_own, "clause_swap": c_swp,
                "clause_exact": (c_own == (truth.get(img) or {}).get("clause")),
            })
            if len(rows) % 5 == 0:
                el = time.time() - t0
                print(f"  [{len(rows)}] {el / len(rows):.1f}s/图", flush=True)

    # ---- 指标 ----
    per_attr = defaultdict(lambda: {"n": 0, "correct": 0,
                                    "cls": Counter(), "cls_correct": Counter()})
    for r in rows:
        cat = r["cat"]
        for k, gold in r["gold_own"].items():
            pred = r["pred_own"].get(k)
            d2 = per_attr[f"{cat}.{k}"]
            d2["n"] += 1
            d2["cls"][gold] += 1
            if pred == gold:
                d2["correct"] += 1
                d2["cls_correct"][gold] += 1

    print("\n" + "=" * 96)
    print(f"{'属性':<32s}{'n':>4s}{'准确率':>8s}{'平衡准确':>9s}{'多数基线':>9s}"
          f"{'覆盖':>7s}")
    print("-" * 96)
    accs, bals, bases, covs = [], [], [], []
    for k in sorted(per_attr):
        v = per_attr[k]
        n = v["n"]
        acc = v["correct"] / n if n else 0
        recs = [v["cls_correct"][c] / v["cls"][c] for c in v["cls"]]
        bal = sum(recs) / len(recs) if recs else 0
        base = max(v["cls"].values()) / n if n else 0
        cov = len(v["cls_correct"]) / len(v["cls"]) if v["cls"] else 0
        accs.append(acc)
        bals.append(bal)
        bases.append(base)
        covs.append(cov)
        print(f"{k:<32s}{n:>4d}{acc:>8.3f}{bal:>9.3f}{base:>9.3f}{cov:>7.2f}")
    avg = lambda x: sum(x) / len(x) if x else 0.0                   # noqa: E731
    print("-" * 96)
    print(f"{'宏平均':<32s}{'':>4s}{avg(accs):>8.3f}{avg(bals):>9.3f}"
          f"{avg(bases):>9.3f}{'':>7s}")

    n_exact = sum(1 for r in rows if r["clause_exact"])
    # 换图测试（属性层面）
    disc = tot = 0
    for r in rows:
        cat = r["cat"]
        spec = full_spec.get(cat, [])
        m_own = sum(1 for k in r["gold_own"]
                    if r["pred_own"].get(k) == r["gold_own"][k])
        m_oth = sum(1 for k in r["gold_own"]
                    if r["pred_own"].get(k) == r["gold_other"].get(k))
        if r["gold_own"] != r["gold_other"]:
            tot += 1
            if m_own > m_oth:
                disc += 1

    print("=" * 96)
    print(f"  从句逐字命中率        {n_exact}/{len(rows)} = "
          f"{n_exact / max(len(rows),1):.3f}")
    print(f"  换图测试（更像自己）  {disc}/{tot} = "
          f"{disc / max(tot,1):.3f}   ← 0.5=瞎猜")
    print("=" * 96)

    summary = {
        "tag": a.tag, "model": a.model, "n": len(rows),
        "macro_acc": round(avg(accs), 3), "macro_balanced": round(avg(bals), 3),
        "macro_baseline": round(avg(bases), 3),
        "clause_exact": round(n_exact / max(len(rows), 1), 3),
        "swap_discrimination": round(disc / max(tot, 1), 3),
        "n_swap": tot, "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    Path(a.out).write_text(json.dumps(
        {"summary": summary, "per_attr": {k: dict(v) for k, v in per_attr.items()},
         "rows": rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"→ {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
