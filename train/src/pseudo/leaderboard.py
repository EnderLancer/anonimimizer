"""Test-set leaderboard with run history.

Every evaluation of a run on the organizers' test set is one line of `runs/test_history.jsonl` (append-only).
The Markdown file is a view of that history: the current leaderboard, averages over seeds, how the best score moved,
each run across its retrains, every evaluation with what changed, and per-type details.

Scores are comparable only on the same version of the test file with the same decoder, so every record carries both.
That pair is the scope of an evaluation: the leaderboard, the seed averages, the progress table and `top_models` use
the current scope only, and a score is never compared with a score of another scope.

Until 2026-10-08 --window / --stride / --no-masks were recorded but not applied in training (`legacy_settings`): such
runs are marked in the leaderboard and are not used to choose the settings of the next training.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from pathlib import Path

from pseudo import TYPES

HISTORY = "test_history.jsonl"
LEGACY_LOG = "test_runs.log"
PARAMS = ("epochs", "lr", "batch", "grad_accum", "window", "stride", "use_masks", "oversample_number", "n_docs", "seed")
LEGACY_TEST = "d78e4da942a8"  # md5 of the first test file: every evaluation recorded before the `test` field existed
_SEED = re.compile(r"_s\d+$")
BAR_LO, BAR_HI, BAR_W = 0.5, 1.0, 20
_EIGHTHS = " ▏▎▍▌▋▊▉"
_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) (\S+) (.*?)(\d\.\d{4})\s*$")
# training settings that make two runs the same model variant (the seed and the data do not)
SETTINGS = {"epochs": None, "lr": None, "batch": None, "grad_accum": 1, "window": 512, "stride": 128, "use_masks": True,
            "oversample_number": 1.0}
SHORT_NAMES = {
    "benjamin/roberta-base-wechsel-ukrainian": "wechsel_base", "benjamin/roberta-large-wechsel-ukrainian": "wechsel_large",
    "FacebookAI/xlm-roberta-base": "xlmr_base", "FacebookAI/xlm-roberta-large": "xlmr_large",
    "microsoft/mdeberta-v3-base": "mdeberta_base", "Goader/liberta-large-v2": "liberta_large",
    "Goader/modern-liberta-large": "modern_liberta_large",
    "Wismut/nym-pii-multilingual": "nym_pii", "markussagen/xlm-roberta-longformer-base-4096": "longformer",
}


def test_version(path: Path) -> str:
    """Version of a test file: scores are comparable only between evaluations that carry the same one."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]


def record(run: str, cfg: dict, res: dict, key: str, ts: str) -> dict:
    """One history line: what was evaluated (run config) and how it scored (`evaluate.py --json` output)."""
    rel, strict = res["ner_relaxed"], res["ner_strict"]
    return {
        "ts": ts, "run": run, "model": cfg.get("model"), "key": key, "data": cfg.get("data_version"),
        "test": res.get("test"), "decode": res.get("decode"), "settings_applied": bool(cfg.get("settings_applied")),
        "params": {k: cfg.get(k) for k in PARAMS}, "train_minutes": cfg.get("train_minutes"),
        "score": res["final_score"], "strict": strict["macro"]["f1"], "precision": rel["macro"]["precision"],
        "recall": rel["macro"]["recall"], "risk": res["deanonymization_risk_by_name"], "b3": res["consistency_b3"]["f1"],
        "per_type": {t: {**{k: rel["per_type"][t][k] for k in ("f1", "precision", "recall", "tp", "fp", "fn")},
                         "strict_f1": strict["per_type"][t]["f1"]} for t in TYPES},
    }


def parse_legacy_log(text: str) -> list[dict]:
    """`test_runs.log` lines (date, run, optional model and data version, macro F1 last) -> history records without per-type metrics."""
    out = []
    for line in text.splitlines():
        m = _LOG_LINE.match(line.strip())
        if not m:
            continue
        ts, run, rest, score = m.groups()
        words = rest.split()
        d = re.search(r"data=(\{.*?\})", rest)
        try:
            data = ast.literal_eval(d.group(1)) if d else None
        except (ValueError, SyntaxError):
            data = None
        out.append({"ts": ts, "run": run, "model": words[0] if words and "/" in words[0] else None, "key": None,
                    "data": data, "score": float(score)})
    return out


def load_history(runs_dir: Path) -> list[dict]:
    """All evaluations, oldest first. The first call imports the old `test_runs.log` so earlier results are kept."""
    path = runs_dir / HISTORY
    if not path.exists():
        log = runs_dir / LEGACY_LOG
        rows = parse_legacy_log(log.read_text(encoding="utf-8")) if log.exists() else []
        if rows:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return rows
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["ts"])


def append_history(runs_dir: Path, rec: dict) -> None:
    with open(runs_dir / HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _same_evaluation(a: dict, b: dict) -> bool:
    if a["run"] != b["run"] or round(a["score"], 4) != round(b["score"], 4):
        return False
    if scope_of(a) != scope_of(b):  # an equal score with another test file or decoder is still another evaluation
        return False
    if a.get("key") and b.get("key"):
        return a["key"] == b["key"]
    return a.get("data") is None or b.get("data") is None or a["data"] == b["data"]


def collapse(history: list[dict]) -> list[dict]:
    """Repeated evaluations of the same trained model count once: the earliest date with the richest metrics."""
    out, last = [], {}
    for r in history:
        prev = last.get(r["run"])
        if prev is not None and _same_evaluation(prev, r):
            if "per_type" not in prev and "per_type" in r:
                prev.update({k: v for k, v in r.items() if k != "ts"})
            continue
        r = dict(r)
        out.append(r)
        last[r["run"]] = r
    return out


def _data_key(data: dict | None) -> str | None:
    return json.dumps(data, sort_keys=True) if data else None


def bar(score: float) -> str:
    x = max(0.0, min(1.0, (score - BAR_LO) / (BAR_HI - BAR_LO))) * BAR_W
    full, part = int(x), int((x - int(x)) * 8)
    blocks = "█" * full + (_EIGHTHS[part] if part else "")
    return "`" + blocks + "░" * (BAR_W - len(blocks)) + "`"


def delta(new: float, old: float | None) -> str:
    if old is None:
        return "-"
    d = new - old
    if abs(d) < 5e-5:
        return "= 0.0000"
    return f"▲ +{d:.4f}" if d > 0 else f"▼ -{-d:.4f}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def _bold_best(values: list[float | None], fmt: str, lowest: bool = False) -> list[str]:
    known = [v for v in values if v is not None]
    top = (min(known) if lowest else max(known)) if known else None
    return ["-" if v is None else f"**{v:{fmt}}**" if v == top and len(known) > 1 else f"{v:{fmt}}" for v in values]


def _model(r: dict) -> str:
    return r["model"].split("/")[-1] if r.get("model") else "-"


def test_of(r: dict) -> str:
    return r.get("test") or LEGACY_TEST


def decode_of(r: dict) -> int:
    return r.get("decode") or 1


def scope_of(r: dict) -> tuple[str, int]:
    """Test file and decoder of an evaluation: two scores are comparable only when both are the same."""
    return test_of(r), decode_of(r)


def legacy_settings(r: dict) -> bool:
    """True for a run of the time when window / stride / masks were recorded but training ignored them (until 2026-10-08).

    Such a run was trained with the defaults and predicted with the recorded window, so its score is evidence neither
    for the recorded settings nor for the defaults. Runs trained with the given values carry `settings_applied`."""
    p = r.get("params") or {}
    return not r.get("settings_applied") and any(p.get(k) is not None and p[k] != SETTINGS[k] for k in ("window", "stride", "use_masks"))


def _current_scope(h: list[dict], test: str | None, decode: int | None) -> tuple[str | None, int]:
    """The scope to show: by default the test file of the newest evaluation and the decoder of the newest evaluation on it."""
    by_time = sorted(h, key=lambda r: r["ts"])
    if test is None and by_time:
        test = test_of(by_time[-1])
    if decode is None:
        decode = next((decode_of(r) for r in reversed(by_time) if test_of(r) == test), 1)
    return test, decode


def _trained(r: dict) -> str | None:
    """The trained model an evaluation belongs to: the part of its key before the test file and the decoder."""
    return r["key"].split(":")[0] if r.get("key") else None


def _in_scope(seq: list[dict], scope: tuple[str | None, int]) -> dict | None:
    """The latest evaluation in `scope` of the model a run holds now; `seq` = the evaluations of the run, oldest first.

    Not simply the last evaluation: a session with an older checkout can score a run with an older decoder after the
    current one did, and that must not hide the score the run already has in the current scope. An evaluation made
    before the run was retrained (another fingerprint, or none recorded) belongs to another model and is not shown."""
    for r in reversed(seq):
        if r is not seq[-1] and (_trained(r) is None or _trained(r) != _trained(seq[-1])):
            return None
        if scope_of(r) == scope:
            return r
    return None


def seed_group(run: str) -> str:
    """`wechsel_base_v3_s43` -> `wechsel_base_v3`: runs that differ only by the seed share a group."""
    return _SEED.sub("", run)


def _changes(r: dict, prev: dict | None, label, tlabel) -> str:
    """What differs from the previous evaluation of the same run: test version, decoder, data version, training settings."""
    if prev is None:
        return "first evaluation"
    out = []
    if test_of(prev) != test_of(r):
        out.append(f"test {tlabel(prev)} → {tlabel(r)}")
    if decode_of(prev) != decode_of(r):
        out.append(f"decoder {decode_of(prev)} → {decode_of(r)}")
    if label(prev) != label(r):
        out.append(f"data {label(prev)} → {label(r)}")
    a, b = prev.get("params") or {}, r.get("params") or {}
    out += [f"{k} {a[k]} → {b[k]}" for k in PARAMS if k in a and k in b and a[k] != b[k]]
    return "; ".join(out) or "retrained"


def render(history: list[dict], runs: list[str], name: str, baseline: str | None = None, cpu: dict | None = None,
           test: str | None = None, decode: int | None = None) -> str:
    """Markdown view of the history. `runs` are the rows of the leaderboard; the history sections cover every run.
    `test` and `decode` are the current scope: the version of the current test file (default: the one of the newest
    evaluation) and of the decoder (default: the one of the newest evaluation on that test file)."""
    cpu = cpu or {}
    h = collapse(history)
    cur = _current_scope(h, test, decode)
    labels: dict[str, str] = {}
    tests: dict[str, str] = {}
    scopes: list[tuple[str, int]] = []
    for r in h:
        k = _data_key(r.get("data"))
        if k and k not in labels:
            labels[k] = f"D{len(labels) + 1}"
        tests.setdefault(test_of(r), f"T{len(tests) + 1}")
        if scope_of(r) not in scopes:
            scopes.append(scope_of(r))
    if cur[0]:
        tests.setdefault(cur[0], f"T{len(tests) + 1}")
        if cur not in scopes:
            scopes.append(cur)
    many = len(scopes) > 1
    decoders = {t: {d for s, d in scopes if s == t} for t in tests}
    # the decoder is named only where one test file was scored with several: other labels stay T1, T2, …
    slabel = lambda s: tests[s[0]] + (f"·d{s[1]}" if len(decoders[s[0]]) > 1 else "")
    label = lambda r: labels.get(_data_key(r.get("data")), "?")
    tlabel = lambda r: tests[test_of(r)]
    version = lambda r: label(r) + (f"·{slabel(scope_of(r))}" if many else "")
    by_run: dict[str, list[dict]] = {}
    best: dict[str, float] = {}
    for r in h:
        seq = by_run.setdefault(r["run"], [])
        r["_prev"] = seq[-1] if seq else None
        # a score is compared only with scores of the same scope (test version and decoder)
        r["_prev_score"] = r["_prev"]["score"] if r["_prev"] and scope_of(r["_prev"]) == scope_of(r) else None
        top_before = best.get(scope_of(r))
        r["_gain"] = None if top_before is None else r["score"] - top_before
        r["_record"] = top_before is None or r["score"] > top_before
        if r["_record"]:
            best[scope_of(r)] = r["score"]
        seq.append(r)
    runs = list(dict.fromkeys(runs))
    on_cur = lambda r: scope_of(r) == cur
    shown = {x: _in_scope(seq, cur) for x, seq in by_run.items()}
    current = sorted((shown[x] for x in runs if shown.get(x)), key=lambda r: -r["score"])
    f1 = lambda r, t: r["per_type"][t]["f1"] if "per_type" in r else None
    h_cur = [r for r in h if on_cur(r)]
    top = max(h_cur, key=lambda r: r["score"]) if h_cur else None

    L = [f"# {name} - organizers' test set", ""]
    if top:
        L += [f"_updated {time.strftime('%Y-%m-%d %H:%M')} · {len(by_run)} runs · {len(h)} evaluations · "
              f"best **{top['score']:.4f}** `{top['run']}`_", ""]
    L += ["Score = test macro F1 (official `evaluate.py`, IoU ≥ 0.5). Bars run from 0.50 to 1.00. ▲ / ▼ = change of the score. "
          "D1, D2, … = data versions, listed at the end.", ""]
    if many:
        suffix = "; «·d1», «·d2», … is the decoder version of a test file that was scored with more than one" \
            if any(len(d) > 1 for d in decoders.values()) else ""
        L += [f"Scores are comparable only on the same test file scored with the same decoder. T1, T2, … are the versions of the "
              f"test file{suffix}; all are listed at the end. The leaderboard, the seed averages and the progress table show the "
              f"current scope **{slabel(cur)}** only; scores of different scopes are not compared.", ""]

    L += ["## Leaderboard", "", f"Latest evaluation of each run{' in the current scope' if many else ''}, best first; the best value of a column is bold.", ""]
    base = shown.get(baseline)
    head = ["#", "run", "model", "test macro F1", "0.50 … 1.00", "vs previous"] + ([f"vs {baseline}"] if base else []) \
        + [t[:3] for t in TYPES] + ["data", "evaluated"]
    cols = [_bold_best([r["score"] for r in current], ".4f")] + [_bold_best([f1(r, t) for r in current], ".3f") for t in TYPES]
    rows = []
    mark = lambda r: r["run"] + (" †" if legacy_settings(r) else "")
    for i, r in enumerate(current):
        rows.append([str(i + 1), mark(r), _model(r), cols[0][i], bar(r["score"]), delta(r["score"], r["_prev_score"])]
                    + ([delta(r["score"], base["score"]) if r is not base else "baseline"] if base else [])
                    + [c[i] for c in cols[1:]] + [label(r), r["ts"]])
    for x in runs:
        if x not in by_run:
            rows.append(["-", x, "no result"] + ["-"] * (len(head) - 3))
        elif not shown[x]:
            old = by_run[x][-1]
            rows.append(["-", mark(old), f"not scored on {slabel(cur)} ({slabel(scope_of(old))}: {old['score']:.4f})"]
                        + ["-"] * (len(head) - 3))
    L += _table(head, rows)
    if any(legacy_settings(by_run[x][-1]) for x in runs if x in by_run):
        L += [f"† trained before 2026-10-08, when --window / --stride / --no-masks were recorded but not applied: the model was "
              f"trained with windows of {SETTINGS['window']} tokens, stride {SETTINGS['stride']} and masks, and predicted with the "
              f"recorded window. The score is evidence neither for the recorded settings nor for the defaults, so the run is not "
              f"used to choose the models of the next training.", ""]

    groups: dict[str, list[dict]] = {}
    for r in current:
        groups.setdefault(seed_group(r["run"]), []).append(r)
    groups = {g: rs for g, rs in groups.items() if len(rs) > 1}
    if groups:
        mean = lambda xs: sum(xs) / len(xs)
        L += ["## Averages over seeds", "",
              "Runs that differ only by the seed (`<name>`, `<name>_s43`, …). A difference between two groups that is smaller than "
              "their spread (max − min) is noise.", ""]
        stat = sorted(((g, rs, [r["score"] for r in rs]) for g, rs in groups.items()), key=lambda x: -mean(x[2]))
        means = _bold_best([mean(sc) for _, _, sc in stat], ".4f")
        rows = []
        for i, (g, rs, sc) in enumerate(stat):
            per = [[f1(r, t) for r in rs if f1(r, t) is not None] for t in TYPES]
            rows.append([g, _model(rs[0]), str(len(rs)), means[i], bar(mean(sc)), f"{min(sc):.4f}", f"{max(sc):.4f}", f"{max(sc) - min(sc):.4f}"]
                        + [f"{mean(v):.3f}" if v else "-" for v in per] + [label(rs[0])])
        L += _table(["group", "model", "seeds", "mean macro F1", "0.50 … 1.00", "min", "max", "spread"] + [t[:3] for t in TYPES] + ["data"], rows)

    L += ["## Progress", "", "Every evaluation that set a new best score, oldest first.", ""]
    L += _table(["date", "run", "model", "test macro F1", "0.50 … 1.00", "gain"],
                [[r["ts"], r["run"], _model(r), f"**{r['score']:.4f}**", bar(r["score"]),
                  "first" if r["_gain"] is None else delta(r["score"], r["score"] - r["_gain"])] for r in h_cur if r["_record"]])

    L += ["## Runs over time", "", "Each run across its retrains (new data or settings), oldest → latest; the last 6 evaluations.", ""]
    rows = []
    for seq in sorted(by_run.values(), key=lambda s: -s[-1]["score"]):
        pts = [f"{p['score']:.4f} ({version(p)})" for p in seq[-6:]]
        pts[-1] = f"**{pts[-1]}**"
        same = len(seq) > 1 and scope_of(seq[0]) == scope_of(seq[-1])
        rows.append([seq[-1]["run"], _model(seq[-1]), str(len(seq)), ("… → " if len(seq) > 6 else "") + " → ".join(pts),
                     delta(seq[-1]["score"], seq[0]["score"]) if same else "-"])
    L += _table(["run", "model", "evaluations", "test macro F1 (data)", "first → latest"], rows)

    L += ["## History", "", "Every evaluation, newest first. ★ = best score at that moment" + (" on its test file and decoder." if many else "."), ""]
    rows = []
    for r in reversed(h):
        rows.append([r["ts"], r["run"], version(r), f"{r['score']:.4f}", delta(r["score"], r["_prev_score"])]
                    + ["-" if f1(r, t) is None else f"{f1(r, t):.3f}" for t in TYPES]
                    + [f"{r['strict']:.4f}" if "strict" in r else "-",
                       ("★ " if r["_record"] else "") + _changes(r, r["_prev"], label, tlabel)])
    L += _table(["date", "run", "data", "test macro F1", "vs previous"] + [t[:3] for t in TYPES] + ["strict", "changed since the run's previous evaluation"], rows)

    full = [r for r in current if "per_type" in r]
    if full:
        L += ["## Details of the leaderboard runs", ""]
        cols = [_bold_best([r[k] for r in full], fmt, lowest) for k, fmt, lowest in
                (("strict", ".4f", False), ("precision", ".3f", False), ("recall", ".3f", False), ("risk", ".3f", True), ("b3", ".3f", False))]
        L += _table(["run", "strict F1", "precision", "recall", "deanon risk", "consistency B³", "CPU s/doc", "train min"],
                    [[r["run"]] + [c[i] for c in cols]
                     + [f"{cpu[r['run']]:.2f}" if r["run"] in cpu else "-", f"{r['train_minutes']:.0f}" if r.get("train_minutes") else "-"]
                     for i, r in enumerate(full)])
        L += ["Precision / recall per type:", ""]
        L += _table(["run"] + list(TYPES),
                    [[r["run"]] + [f"{r['per_type'][t]['precision']:.3f} / {r['per_type'][t]['recall']:.3f}" for t in TYPES] for r in full])

    if labels:
        L += ["## Data versions", ""]
        rows = []
        for k, lab in labels.items():
            seen = [r for r in h if _data_key(r.get("data")) == k]
            d = json.loads(k)
            rows.append([lab, f"`{d.get('train.parquet', '-')}`", f"`{d.get('dev.parquet', '-')}`", seen[0]["ts"], str(len(seen))])
        L += _table(["data", "train.parquet", "dev.parquet", "first evaluated", "evaluations"], rows)
    if many:
        L += ["## Test versions", ""]
        rows = []
        for s in scopes:
            seen = [r for r in h if scope_of(r) == s]
            rows.append([slabel(s) + (" (current)" if s == cur else ""), f"`{s[0]}`", str(s[1]), seen[0]["ts"] if seen else "-",
                         str(len(seen))])
        L += _table(["test", "md5 of court_cases.jsonl", "decoder", "first evaluated", "evaluations"], rows)
    return "\n".join(L)


def short_name(model: str, settings: dict) -> str:
    """Run name of a model variant: `wechsel_base`, `nym_pii_w1024`; an unknown model gets its Hub name."""
    name = SHORT_NAMES.get(model) or re.sub(r"[^a-z0-9]+", "_", model.split("/")[-1].lower()).strip("_")
    if settings["window"] != SETTINGS["window"]:
        name += f"_w{settings['window']}"
    if not settings["use_masks"]:
        name += "_nomasks"
    if settings["oversample_number"] != SETTINGS["oversample_number"]:
        name += f"_os{settings['oversample_number']:g}"
    return name


def cli_args(settings: dict) -> str:
    """Arguments of scripts/train.py that reproduce the settings which differ from its defaults."""
    out = []
    if (settings["window"], settings["stride"]) != (SETTINGS["window"], SETTINGS["stride"]):
        out.append(f"--window {settings['window']} --stride {settings['stride']}")
    if settings["window"] >= 2048:
        out.append("--grad-checkpointing")  # not recorded in the history; long windows do not fit in memory without it
    if not settings["use_masks"]:
        out.append("--no-masks")
    if settings["oversample_number"] != SETTINGS["oversample_number"]:
        out.append(f"--oversample-number {settings['oversample_number']:g}")
    return " ".join(out)


def top_models(history: list[dict], n: int = 3, test: str | None = None, decode: int | None = None) -> list[dict]:
    """The `n` best models of the history, best first, each with the training settings it scored best with.

    Runs that share the model and the settings (other seeds, other data) form one variant; its score is the mean of the
    latest evaluation of each run in the current scope: the test version `test` (default: the one of the newest
    evaluation) scored with the decoder `decode` (default: the one of the newest evaluation on that test version).
    A later evaluation of the same trained model in another scope does not hide it; a run that was retrained since
    and not scored in the current scope again is left out (`_in_scope`). A model takes one place: its best variant. Left out: runs on a part of the data
    (`n_docs`), records without settings, and runs whose recorded window / stride / masks were not applied in training
    (`legacy_settings`): those would be merged with runs that did use the settings and offered for retraining with
    settings nobody has tested."""
    h = collapse(history)
    if not h:
        return []
    cur = _current_scope(h, test, decode)
    by_run: dict[str, list[dict]] = {}
    for r in h:
        by_run.setdefault(r["run"], []).append(r)
    variants: dict[tuple[str, str], list[dict]] = {}
    for r in filter(None, (_in_scope(seq, cur) for seq in by_run.values())):
        p = r.get("params") or {}
        settings = {k: d if p.get(k) is None else p[k] for k, d in SETTINGS.items()}
        if legacy_settings(r) or not r.get("model") or p.get("n_docs") or None in settings.values():
            continue
        variants.setdefault((r["model"], json.dumps(settings, sort_keys=True)), []).append(r)
    mean = lambda rs: sum(r["score"] for r in rs) / len(rs)
    out, seen = [], set()
    for (model, key), rs in sorted(variants.items(), key=lambda kv: (-mean(kv[1]), kv[0])):
        if model in seen:
            continue
        seen.add(model)
        settings = json.loads(key)
        out.append({"name": short_name(model, settings), "model": model, "extra": cli_args(settings), "lr": settings["lr"],
                    "batch": settings["batch"], "accum": settings["grad_accum"], "epochs": settings["epochs"],
                    "score": mean(rs), "best": max(r["score"] for r in rs), "runs": sorted(r["run"] for r in rs)})
        if len(out) == n:
            break
    return out
