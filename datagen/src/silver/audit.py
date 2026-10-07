"""Step 5: audit the labels against the NER layer and prepare manual-QA samples."""
from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict

from silver.io import SILVER, read_jsonl, step_path
from silver.ner import _cache_path

log = logging.getLogger("silver")
AUDIT = SILVER / "audit"
JUDGE = re.compile(r"судд|головуюч", re.I)
LAWYER = re.compile(r"адвокат", re.I)


def _iou(a, b) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    return inter / (max(a["end"], b["end"]) - min(a["start"], b["start"]))


def _ctx(text: str, s: int, e: int, w: int = 100) -> str:
    left = text[max(0, s - w): s]
    right = text[e: e + w]
    if s - w > 0 and left and not left[0].isspace():
        left = left.split(" ", 1)[1] if " " in left else left
    if e + w < len(text) and right and not right[-1].isspace():
        right = right.rsplit(" ", 1)[0] if " " in right else right
    return (left + "[[" + text[s:e] + "]]" + right).replace("\n", " ")


def _stratified(docs: list[dict], n: int, rng: random.Random) -> list[dict]:
    groups = defaultdict(list)
    for d in docs:
        t = d["text"][:1500]
        key = "negative" if d.get("is_negative") else "judge" if JUDGE.search(t) else "lawyer" if LAWYER.search(d["text"]) else "other"
        groups[key].append(d)
    out = []
    per = max(1, n // len(groups))
    for k, g in groups.items():
        rng.shuffle(g)
        out += g[:per]
    return out[:n]


def run_audit(seed: int = 42, src: str = "02_labeled", n: int = 1000) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    docs = [d for d in read_jsonl(step_path(src)) if _cache_path(d["id"], d["text"]).exists()]
    if not docs:
        log.warning("audit: no documents with NER cache yet")
        return
    sample = _stratified(docs, n, rng)
    status = Counter()
    by_source = defaultdict(Counter)
    holes, contexts = [], Counter()
    for d in sample:
        text = d["text"]
        persons = [l for l in d["labels"] if l["type"] == "ОСОБА"]
        placeholders = [l for l in d["labels"] if l["source"] == "placeholder"]
        ents = json.loads(_cache_path(d["id"], d["text"]).read_text(encoding="utf-8"))
        for e in ents:
            if any(_iou(e, l) >= 0.5 or (l["start"] <= e["start"] and e["end"] <= l["end"]) for l in persons + placeholders):
                status["labeled"] += 1
            elif any(_iou(e, m) > 0 for m in d["masks"]):
                status["masked"] += 1
            else:
                status["hole"] += 1
                left = text[max(0, e["start"] - 40): e["start"]].split()
                contexts[" ".join(left[-2:]).lower()] += 1
                holes.append({"id": d["id"], "text": e["text"], "context": _ctx(text, e["start"], e["end"])})
        for l in persons:
            confirmed = any(_iou(e, l) >= 0.5 for e in ents)
            by_source[l["source"]]["confirmed" if confirmed else "unconfirmed"] += 1
    total = sum(status.values()) or 1
    metrics = {
        "docs": len(sample),
        "pers_total": total,
        "holes_rate": status["hole"] / total,
        "masked_rate": status["masked"] / total,
        "labeled_rate": status["labeled"] / total,
        "unconfirmed_rate_by_source": {s: c["unconfirmed"] / (sum(c.values()) or 1) for s, c in by_source.items()},
        "counts_by_source": {s: dict(c) for s, c in by_source.items()},
    }
    (AUDIT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(AUDIT / "holes.jsonl", "w", encoding="utf-8") as f:
        for h in holes:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    lines = ["# Топ контекстів «дірок» (2 слова ліворуч від PERS без лейбла)\n", "| n | контекст |", "|---|---|"]
    lines += [f"| {n} | {c} |" for c, n in contexts.most_common(30)]
    lines += ["", "## Приклади", ""] + [f"- `{h['id']}` {h['context']}" for h in holes[:60]]
    (AUDIT / "top_contexts.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("audit on %d docs: %s", len(sample), json.dumps({k: v for k, v in metrics.items() if k != "counts_by_source"}, ensure_ascii=False))


SAMPLE_LAYERS = ["role_regex", "header", "ini_any", "full_pib", "signature", "ner", "rnokpp", "passport", "iban", "card",
                 "email", "dob", "address", "placeholder_person"]


def run_samples(seed: int = 42, src: str | None = None, per_layer: int = 50) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    src = src or ("03_ner" if step_path("03_ner").exists() else "02_labeled")
    pools: dict[str, list] = defaultdict(list)
    for d in read_jsonl(step_path(src)):
        for l in d["labels"]:
            key = l["source"]
            if key == "placeholder":
                if l["type"] != "ОСОБА":
                    continue
                key = "placeholder_person"
            pools[key].append((d["id"], _ctx(d["text"], l["start"], l["end"])))
    for layer in SAMPLE_LAYERS:
        items = pools.get(layer, [])
        rng.shuffle(items)
        items = items[:per_layer]
        lines = [f"# {layer} — {len(items)} випадкових спанів (всього {len(pools.get(layer, []))})", "",
                 "Постав у колонці ok/bad: `ok` якщо спан — коректна персональна дата потрібного типу з правильними межами.", "",
                 "| # | id | контекст | ok/bad |", "|---|---|---|---|"]
        lines += [f"| {i + 1} | {doc_id} | {ctx.replace('|', '¦')} |  |" for i, (doc_id, ctx) in enumerate(items)]
        (AUDIT / f"samples_{layer}.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("samples written to %s", AUDIT)


def score_samples() -> None:
    rows = []
    for p in sorted(AUDIT.glob("samples_*.md")):
        marks = Counter()
        bad = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| ") or line.startswith("| #"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4:
                continue
            mark = cells[-1].lower()
            if mark in ("ok", "+", "1", "так"):
                marks["ok"] += 1
            elif mark in ("bad", "-", "0", "ні"):
                marks["bad"] += 1
                bad.append(cells[2])
        n = marks["ok"] + marks["bad"]
        if n:
            rows.append((p.stem.replace("samples_", ""), marks["ok"] / n, n, bad))
    for layer, prec, n, bad in rows:
        print(f"{layer:20s} precision {prec:.3f}  (n={n})")
        for b in bad[:10]:
            print("    bad:", b)
