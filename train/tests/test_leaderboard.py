from pseudo import TYPES
from pseudo.leaderboard import bar, collapse, delta, load_history, parse_legacy_log, record, render

D1 = {"train.parquet": "aaa", "dev.parquet": "bbb"}
D2 = {"train.parquet": "ccc", "dev.parquet": "bbb"}


def result(score: float) -> dict:
    per = {t: {"f1": score, "precision": score, "recall": score, "tp": 1, "fp": 0, "fn": 0} for t in TYPES}
    block = {"per_type": per, "macro": {"f1": score, "precision": score, "recall": score}}
    return {"final_score": score, "ner_relaxed": block, "ner_strict": block, "deanonymization_risk_by_name": 0.01,
            "consistency_b3": {"f1": 0.5}}


def rec(run: str, score: float, ts: str, data: dict, key: str, **params) -> dict:
    cfg = {"model": "org/model", "data_version": data, "train_minutes": 10, "lr": 4e-5, "epochs": 3, **params}
    return record(run, cfg, result(score), key, ts)


def test_parse_legacy_log_reads_both_line_formats():
    rows = parse_legacy_log(
        "2026-10-07 18:03 wechsel_base_full Підсумковий бал                 0.8240\n"
        "2026-10-08 08:26 wechsel_base benjamin/roberta-base-wechsel-ukrainian data={'train.parquet': 'aaa', 'dev.parquet': 'bbb'} "
        "dev=0.9906 | test macro F1 0.8530\n"
        "not a log line\n")
    assert [(r["run"], r["score"], r["model"], r["data"]) for r in rows] == [
        ("wechsel_base_full", 0.8240, None, None),
        ("wechsel_base", 0.8530, "benjamin/roberta-base-wechsel-ukrainian", D1)]


def test_load_history_imports_the_old_log_once(tmp_path):
    (tmp_path / "test_runs.log").write_text("2026-10-07 18:03 a Підсумковий бал 0.8240\n", encoding="utf-8")
    assert [r["run"] for r in load_history(tmp_path)] == ["a"]
    (tmp_path / "test_runs.log").write_text("", encoding="utf-8")
    assert [r["run"] for r in load_history(tmp_path)] == ["a"]  # now read from test_history.jsonl


def test_collapse_merges_a_repeated_evaluation_of_the_same_model():
    log = {"ts": "2026-10-08 08:26", "run": "a", "model": None, "key": None, "data": D1, "score": 0.853}
    again = {**log, "ts": "2026-10-08 08:34"}
    full = rec("a", 0.853, "2026-10-08 17:00", D1, "k1")
    retrained = rec("a", 0.861, "2026-10-09 10:00", D2, "k2")
    out = collapse([log, again, full, retrained])
    assert [(r["ts"], r["key"]) for r in out] == [("2026-10-08 08:26", "k1"), ("2026-10-09 10:00", "k2")]
    assert "per_type" in out[0]


def test_render_shows_progress_and_what_changed():
    history = [rec("a", 0.80, "2026-10-07 10:00", D1, "k1"), rec("b", 0.85, "2026-10-07 11:00", D1, "k2"),
               rec("a", 0.87, "2026-10-08 10:00", D2, "k3", lr=2e-5)]
    md = render(history, ["a", "b", "missing"], "leaderboard", baseline="b")
    board = md.split("## Progress")[0]
    assert board.index("| 1 | a |") < board.index("| 2 | b |")
    assert "▲ +0.0700" in board and "▲ +0.0200" in board and "baseline" in board
    assert "| - | missing | no result |" in board
    progress = md.split("## Progress")[1].split("## Runs over time")[0]
    assert progress.count("\n| 2026") == 3 and "first" in progress
    assert "0.8000 (D1) → **0.8700 (D2)**" in md
    assert "★ data D1 → D2; lr 4e-05 → 2e-05" in md


def test_bar_and_delta():
    assert bar(1.0) == "`" + "█" * 20 + "`" and bar(0.4) == "`" + "░" * 20 + "`" and len(bar(0.853)) == 22
    assert delta(0.85, None) == "-" and delta(0.85, 0.85) == "= 0.0000" and delta(0.84, 0.85) == "▼ -0.0100"
