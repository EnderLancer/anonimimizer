"""Бенчмарк образу: час на документ (NER-модель окремо, правила окремо) і ресурси.

    docker run --rm --entrypoint python -v "$PWD":/data pseudonymizer /app/bench.py input.jsonl --out bench.json
    python docker/bench.py cases.jsonl --ner onnx:models/ner --out bench.json     # поза Docker

Пише JSON (усі вимірювання, час кожного документа) і друкує дві Markdown-таблиці: ресурси та розподіл
часу на документ (p50 / p95 / p99 / avg / max). Пам'ять — пік RSS процесу (ru_maxrss) і ліміт cgroup,
якщо контейнер запущено з --memory.
"""
import argparse
import json
import os
import resource
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import ner_adapters as N  # noqa: E402
import ner_postprocess as NP  # noqa: E402
import run  # noqa: E402


def rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if sys.platform != "darwin" else kb / 1024 / 1024


def cgroup_limit_mb():
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v.isdigit() and int(v) < 1 << 60:
                return int(v) / 1024 / 1024
        except OSError:
            pass
    return None


def cgroup_cpus():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return int(q) / int(p)
    except (OSError, ValueError):
        pass
    return None


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(xs):
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99), "avg": statistics.fmean(xs), "max": max(xs),
            "min": min(xs), "sum": sum(xs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--ner", default=os.environ.get("NER_SPEC", run.DEFAULT_NER))
    ap.add_argument("--mode", default=os.environ.get("MODE", "model_regex"), choices=["model", "regex", "model_regex"])
    ap.add_argument("--out", help="JSON з усіма вимірюваннями")
    ap.add_argument("--submission", help="заодно записати анотації (як run.py)")
    a = ap.parse_args()
    src = a.src if os.path.isabs(a.src) or os.path.exists(a.src) else os.path.join("/data", a.src)
    rows = [json.loads(l) for l in open(src, encoding="utf8") if l.strip()]
    t_start = time.perf_counter()
    rss0 = rss_mb()
    adapter = N.load_adapter(a.ner) if a.mode != "regex" else None
    t_load = time.perf_counter() - t_start
    docs, subs = [], []
    for r in rows:
        text = r["original"]
        t0 = time.perf_counter()
        spans = NP.process(N.map_labels(adapter.predict([text])[0]), text) if adapter else []
        t1 = time.perf_counter()
        ann = run.annotate(text, spans, a.mode)
        t2 = time.perf_counter()
        docs.append({"id": r["id"], "chars": len(text), "ner_s": t1 - t0, "rules_s": t2 - t1, "total_s": t2 - t0,
                     "entities": len(ann)})
        subs.append({"id": r["id"], "annotations": ann})
        print(f"{len(docs)}/{len(rows)} {sum(d['total_s'] for d in docs):.0f}s", file=sys.stderr, end="\r", flush=True)
    print(file=sys.stderr)
    wall = time.perf_counter() - t_start
    chars = sum(d["chars"] for d in docs)
    res = {
        "ner": a.ner, "mode": a.mode, "docs": len(docs), "chars": chars,
        "cpus": os.cpu_count(), "cpu_limit": cgroup_cpus(), "memory_limit_mb": cgroup_limit_mb(),
        "rss_peak_mb": rss_mb(), "rss_before_model_mb": rss0, "model_load_s": t_load, "wall_s": wall,
        "per_doc": {k: stats([d[k] for d in docs]) for k in ("total_s", "ner_s", "rules_s")},
        "per_1k_chars_ms": stats([d["total_s"] / d["chars"] * 1e6 for d in docs if d["chars"]]),
        "chars_per_s": chars / sum(d["total_s"] for d in docs),
        "documents": docs,
    }
    if a.out:
        json.dump(res, open(a.out if os.path.isabs(a.out) else os.path.join("/data", a.out) if os.path.isdir("/data") else a.out,
                            "w", encoding="utf8"), ensure_ascii=False, indent=1)
    if a.submission:
        with open(a.submission, "w", encoding="utf8") as g:
            for s in subs:
                g.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(markdown(res))


def markdown(r):
    lim = r["memory_limit_mb"]
    cpu = r["cpu_limit"] or r["cpus"]
    lines = ["| Ресурс | Значення |", "|---|---|",
             f"| CPU | {cpu:g}{' (ліміт контейнера)' if r['cpu_limit'] else ' (без ліміту)'}, без GPU |",
             f"| Ліміт пам'яті | {lim / 1024:.1f} GB |" if lim else "| Ліміт пам'яті | немає |",
             f"| Пік RSS процесу | {r['rss_peak_mb'] / 1024:.2f} GB |",
             f"| Завантаження моделі | {r['model_load_s']:.1f} с |",
             f"| Документів / символів | {r['docs']} / {r['chars']:,} |".replace(",", " "),
             f"| Загальний час | {r['wall_s']:.0f} с ({r['wall_s'] / 60:.1f} хв) |",
             f"| Пропускна здатність | {r['docs'] / r['wall_s'] * 60:.1f} док/хв, {r['chars_per_s'] / 1000:.1f} тис. символів/с |", ""]
    lines += ["| Час, с | p50 | p95 | p99 | avg | max |", "|---|---|---|---|---|---|"]
    for k, name in (("total_s", "Документ разом"), ("ner_s", "— NER-модель"), ("rules_s", "— правила і злиття")):
        s = r["per_doc"][k]
        lines.append(f"| {name} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | {s['avg']:.2f} | {s['max']:.2f} |")
    s = r["per_1k_chars_ms"]
    lines.append(f"| На 1 000 символів, мс | {s['p50']:.0f} | {s['p95']:.0f} | {s['p99']:.0f} | {s['avg']:.0f} | {s['max']:.0f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
