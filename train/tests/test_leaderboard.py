from pseudo import TYPES
from pseudo.leaderboard import (bar, collapse, decode_of, delta, legacy_settings, load_history, parse_legacy_log, record, render,
                                scope_of, top_models)

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


def rec2(run: str, score: float, ts: str, test: str | None = None, decode: int | None = None, **params) -> dict:
    cfg = {"model": "org/model", "data_version": D1, "train_minutes": 10, "lr": 4e-5, "epochs": 3, "seed": 42, **params}
    return record(run, cfg, {**result(score), "test": test, "decode": decode}, f"{run}:{test}:{decode}:{params}", ts)


def test_scores_on_another_test_version_are_not_compared():
    history = [rec2("a", 0.85, "2026-10-07 10:00"), rec2("b", 0.90, "2026-10-07 11:00"),  # the first test file, no `test` field
               rec2("a", 0.87, "2026-10-09 10:00", test="955fcb0f9240", decode=2)]
    md = render(history, ["a", "b"], "leaderboard", test="955fcb0f9240")
    board = md.split("## Leaderboard")[1].split("## Progress")[0]
    assert "| 1 | a |" in board and "| - | b | not scored on T2 (T1: 0.9000) |" in board
    assert "▲" not in board  # 0.87 on T2 is not an improvement over 0.85 on T1
    assert "best **0.8700** `a`" in md
    progress = md.split("## Progress")[1].split("## Runs over time")[0]
    assert progress.count("\n| 2026") == 1
    assert "0.8500 (D1·T1) → **0.8700 (D1·T2)**" in md
    assert "★ test T1 → T2; decoder 1 → 2" in md
    assert "| T2 (current) | `955fcb0f9240` |" in md and f"| T1 | `d78e4da942a8` |" in md


def test_seed_runs_are_averaged():
    history = [rec2("a", 0.80, "2026-10-09 10:00"), rec2("a_s43", 0.84, "2026-10-09 11:00", seed=43),
               rec2("a_s44", 0.82, "2026-10-09 12:00", seed=44), rec2("b_v3", 0.90, "2026-10-09 13:00"),
               rec2("b_v3_s43", 0.92, "2026-10-09 14:00", seed=43), rec2("c", 0.70, "2026-10-09 15:00")]
    md = render(history, ["a", "a_s43", "a_s44", "b_v3", "b_v3_s43", "c"], "data")
    seeds = md.split("## Averages over seeds")[1].split("## Progress")[0]
    assert seeds.index("| b_v3 |") < seeds.index("| a |") and "| c |" not in seeds
    assert "| a | model | 3 | 0.8200 |" in seeds and "| 0.8000 | 0.8400 | 0.0400 |" in seeds
    assert "| b_v3 | model | 2 | **0.9100** |" in seeds
    assert "## Averages over seeds" not in render(history[:1] + history[3:4], ["a", "b_v3"], "data")


def test_top_models_ranks_variants_by_mean_and_gives_a_model_one_place():
    base = dict(epochs=3, lr=4e-5, batch=64, grad_accum=1, window=512, stride=128, use_masks=True, oversample_number=1.0, n_docs=None)
    def r(run, model, score, test="t2", **over):
        return {"ts": "2026-10-08 10:00", "run": run, "model": model, "key": run, "data": {"train.parquet": "a"}, "test": test,
                "score": score, "params": {**base, "seed": 42, **over}}
    wechsel, nym, xlmr, large = ("benjamin/roberta-base-wechsel-ukrainian", "Wismut/nym-pii-multilingual",
                                 "FacebookAI/xlm-roberta-base", "some-org/New-Model.v2")
    history = [
        r("wechsel_base_v3", wechsel, 0.86), r("wechsel_base_v3_s43", wechsel, 0.82, seed=43),  # mean 0.84
        r("H3_no_oversample", wechsel, 0.83, oversample_number=0.0),                            # same model, worse variant
        # trained with the recorded window: without the marker the run would be left out (see the legacy settings tests)
        {**r("H4", nym, 0.87, window=1024, stride=256, batch=32, lr=3e-5), "settings_applied": True},
        r("xlmr_base", xlmr, 0.80), r("new", large, 0.79),
        r("smoke", xlmr, 0.99, n_docs=2000),          # trained on a part of the data
        r("old_test", large, 0.99, test="t1"),        # scored on another test version
        {"ts": "2026-10-07 10:00", "run": "legacy", "model": large, "key": None, "data": None, "test": "t2", "score": 0.99},
    ]
    top = top_models(history, 3, test="t2")
    assert [(m["name"], round(m["score"], 4)) for m in top] == [("nym_pii_w1024", 0.87), ("wechsel_base", 0.84), ("xlmr_base", 0.80)]
    assert top[0]["extra"] == "--window 1024 --stride 256" and top[0]["batch"] == 32 and top[1]["extra"] == ""
    assert top[1]["runs"] == ["wechsel_base_v3", "wechsel_base_v3_s43"] and top[1]["best"] == 0.86
    assert top_models(history, 9, test="t2")[-1]["name"] == "new_model_v2"
    assert top_models([], 3) == []


T2 = "955fcb0f9240"


def test_scores_of_another_decoder_are_not_compared():
    history = [rec2("a", 0.85, "2026-10-07 10:00"),  # the first test file, scored with one decoder only: its label stays T1
               rec2("a", 0.90, "2026-10-09 10:00", test=T2, decode=2), rec2("b", 0.95, "2026-10-09 11:00", test=T2, decode=2),
               rec2("a", 0.80, "2026-10-10 10:00", test=T2, decode=3), rec2("c", 0.70, "2026-10-10 11:00", test=T2, decode=3)]
    assert [scope_of(r) for r in history[:4:3]] == [("d78e4da942a8", 1), (T2, 3)] and decode_of({"run": "x"}) == 1
    md = render(history, ["a", "b", "c"], "leaderboard", baseline="b", test=T2, decode=3)
    board = md.split("## Leaderboard")[1].split("## Progress")[0]
    assert "| 1 | a |" in board and "| 2 | c |" in board
    assert "▼" not in board and "▲" not in board  # 0.80 with decoder 3 is not a drop from 0.90 with decoder 2
    assert "| - | b | not scored on T2·d3 (T2·d2: 0.9500) |" in board
    assert "vs b" not in board  # the baseline has no score in the current scope
    assert "best **0.8000** `a`" in md
    progress = md.split("## Progress")[1].split("## Runs over time")[0]
    assert progress.count("\n| 2026") == 1 and "0.8000" in progress and "0.9" not in progress
    over_time = md.split("## Runs over time")[1].split("## History")[0]
    assert "0.8500 (D1·T1) → 0.9000 (D1·T2·d2) → **0.8000 (D1·T2·d3)** | - |" in over_time
    # neither «first → latest» nor «vs previous» of the history crosses the decoder change
    assert "▼" not in md.split("## Leaderboard")[1] and "▲ +0.0500" not in md
    assert "★ decoder 2 → 3" in md
    assert "same test file scored with the same decoder" in md and "**T2·d3** only" in md
    versions = md.split("## Test versions")[1]
    assert "| test | md5 of court_cases.jsonl | decoder | first evaluated | evaluations |" in versions
    assert "| T1 | `d78e4da942a8` | 1 | 2026-10-07 10:00 | 1 |" in versions
    assert f"| T2·d2 | `{T2}` | 2 | 2026-10-09 10:00 | 2 |" in versions
    assert f"| T2·d3 (current) | `{T2}` | 3 | 2026-10-10 10:00 | 2 |" in versions


def test_render_defaults_to_the_scope_of_the_newest_evaluation():
    history = [rec2("a", 0.85, "2026-10-07 10:00"),
               rec2("a", 0.90, "2026-10-09 10:00", test=T2, decode=2), rec2("b", 0.95, "2026-10-09 11:00", test=T2, decode=2),
               rec2("a", 0.80, "2026-10-10 10:00", test=T2, decode=3)]
    md = render(history, ["a", "b"], "leaderboard")
    assert "best **0.8000** `a`" in md and "| - | b | not scored on T2·d3 (T2·d2: 0.9500) |" in md
    # a decoder nothing was scored with yet: every run waits for its evaluation
    md = render(history, ["a", "b"], "leaderboard", test=T2, decode=4)
    assert "| - | a | not scored on T2·d4 (T2·d3: 0.8000) |" in md and "| T2·d4 (current) |" in md and "best **" not in md
    # one decoder per test file: the labels stay as they were
    md = render(history[:3], ["a", "b"], "leaderboard")
    assert "·d" not in md and "| T2 (current) |" in md and "| 1 | b |" in md and "| 2 | a |" in md


def test_seed_averages_use_the_current_scope_only():
    history = [rec2("a", 0.80, "2026-10-09 10:00", test=T2, decode=2), rec2("a_s43", 0.90, "2026-10-09 11:00", test=T2, decode=2, seed=43),
               rec2("a", 0.70, "2026-10-10 10:00", test=T2, decode=3)]
    assert "## Averages over seeds" in render(history[:2], ["a", "a_s43"], "leaderboard")
    assert "## Averages over seeds" not in render(history, ["a", "a_s43"], "leaderboard")  # a_s43 was not scored with decoder 3


def hist(run: str, model: str, score: float, ts: str = "2026-10-08 10:00", decode: int | None = None, applied: bool | None = None,
         **over) -> dict:
    params = {"epochs": 3, "lr": 4e-5, "batch": 64, "grad_accum": 1, "window": 512, "stride": 128, "use_masks": True,
              "oversample_number": 1.0, "n_docs": None, "seed": 42, **over}
    r = {"ts": ts, "run": run, "model": model, "key": f"{run}:{decode}", "data": D1, "test": T2, "score": score, "params": params}
    return {**r, **({} if decode is None else {"decode": decode}), **({} if applied is None else {"settings_applied": applied})}


WECHSEL, XLMR, NYM = "benjamin/roberta-base-wechsel-ukrainian", "FacebookAI/xlm-roberta-base", "Wismut/nym-pii-multilingual"


def test_top_models_ranks_the_current_decoder_only():
    history = [hist("old_star", XLMR, 0.99, "2026-10-08 10:00", decode=2),  # its only score comes from the old decoder
               hist("wechsel_base", WECHSEL, 0.95, "2026-10-08 11:00", decode=2),
               hist("wechsel_base", WECHSEL, 0.80, "2026-10-09 10:00", decode=3),
               hist("nym", NYM, 0.70, "2026-10-09 11:00", decode=3)]
    for top in (top_models(history, 3, test=T2, decode=3), top_models(history, 3)):  # the default: the newest evaluation
        assert [(m["name"], m["score"], m["runs"]) for m in top] == [("wechsel_base", 0.80, ["wechsel_base"]), ("nym_pii", 0.70, ["nym"])]
    # the same trained wechsel_base was scored with decoder 3 since: its decoder 2 score still stands in that scope
    old = top_models(history, 3, test=T2, decode=2)
    assert [(m["name"], m["score"]) for m in old] == [("xlmr_base", 0.99), ("wechsel_base", 0.95)]
    assert top_models(history, 3, test=T2, decode=4) == []


def test_a_later_evaluation_in_another_scope_does_not_hide_the_current_one():
    # a session with an older checkout scored `a` with decoder 2 after it was scored with the current decoder 3
    history = [rec2("a", 0.80, "2026-10-10 10:00", test=T2, decode=3), rec2("b", 0.70, "2026-10-10 11:00", test=T2, decode=3),
               rec2("a", 0.90, "2026-10-11 10:00", test=T2, decode=2)]
    md = render(history, ["a", "b"], "leaderboard", baseline="a", test=T2, decode=3)
    board = md.split("## Leaderboard")[1].split("## Progress")[0]
    assert "| 1 | a | model | **0.8000** |" in board and "| 2 | b |" in board and "not scored" not in board
    assert "| baseline |" in board and "▼ -0.1000" in board and "▲" not in board  # b against the baseline a; no 0.80 → 0.90
    assert "best **0.8000** `a`" in md
    # with the older decoder as the current scope the same history shows the other score of `a`, and b waits
    old = render(history, ["a", "b"], "leaderboard", test=T2, decode=2).split("## Leaderboard")[1].split("## Progress")[0]
    assert "| 1 | a | model | 0.9000 |" in old and "| - | b | not scored on T1·d2 (T1·d3: 0.7000) |" in old

    top = [hist("a", WECHSEL, 0.80, "2026-10-10 10:00", decode=3), hist("b", XLMR, 0.70, "2026-10-10 11:00", decode=3),
           hist("a", WECHSEL, 0.90, "2026-10-11 10:00", decode=2)]
    assert [(m["name"], m["score"], m["runs"]) for m in top_models(top, 3, test=T2, decode=3)] == [
        ("wechsel_base", 0.80, ["a"]), ("xlmr_base", 0.70, ["b"])]


def test_a_score_of_the_model_before_its_retrain_is_not_shown():
    # `a` was retrained (another fingerprint in the key) and the new model has a score with the old decoder only
    retrained = {**hist("a", WECHSEL, 0.60, "2026-10-11 10:00", decode=2), "key": f"fp2:{T2}:d2"}
    history = [{**hist("a", WECHSEL, 0.80, "2026-10-10 10:00", decode=3), "key": f"fp1:{T2}:d3"},
               hist("b", XLMR, 0.70, "2026-10-10 11:00", decode=3), retrained]
    assert [(m["name"], m["score"]) for m in top_models(history, 3, test=T2, decode=3)] == [("xlmr_base", 0.70)]
    board = render(history, ["a", "b"], "leaderboard", test=T2, decode=3).split("## Leaderboard")[1].split("## Progress")[0]
    assert "| 1 | b |" in board and "| - | a | not scored on T1·d3 (T1·d2: 0.6000) |" in board
    # scored with the current decoder again: the retrained model is back with its own score
    again = history + [{**hist("a", WECHSEL, 0.65, "2026-10-12 10:00", decode=3), "key": f"fp2:{T2}:d3"}]
    assert [(m["name"], m["score"]) for m in top_models(again, 3, test=T2, decode=3)] == [("xlmr_base", 0.70), ("wechsel_base", 0.65)]
    # evaluations without a key (the old log) cannot be attributed to the model the run holds now
    keyless = [{**history[0], "key": None}, history[1], {**retrained, "key": None}]
    assert [(m["name"], m["score"]) for m in top_models(keyless, 3, test=T2, decode=3)] == [("xlmr_base", 0.70)]


def test_top_models_leaves_out_runs_whose_settings_were_not_applied():
    old = hist("H4_nym_pii_mmbert", NYM, 0.90, window=1024, stride=256)
    new = hist("nym_pii_w1024", NYM, 0.50, "2026-10-09 10:00", applied=True, window=1024, stride=256)
    top = top_models([old, new], 3)
    assert [(m["name"], m["score"], m["runs"], m["extra"]) for m in top] == [("nym_pii_w1024", 0.50, ["nym_pii_w1024"], "--window 1024 --stride 256")]
    assert top_models([old], 3) == []
    # the real case of 2026-10-08: the old run scores higher, the model is still offered with the settings that were tested
    top = top_models([hist("H4_nym_pii_mmbert", NYM, 0.7935, window=1024, stride=256), hist("nym_pii_w512", NYM, 0.7911)], 3)
    assert [(m["name"], m["extra"], m["score"], m["runs"]) for m in top] == [("nym_pii", "", 0.7911, ["nym_pii_w512"])]
    for other in ({"stride": 256}, {"use_masks": False}):
        assert top_models([hist("x", NYM, 0.90, **other)], 3) == []
        assert len(top_models([hist("x", NYM, 0.90, applied=True, **other)], 3)) == 1


def test_legacy_settings_needs_a_non_default_setting_without_the_marker():
    assert legacy_settings(hist("a", NYM, 0.9, window=1024)) and legacy_settings(hist("a", NYM, 0.9, stride=256))
    assert legacy_settings(hist("a", NYM, 0.9, use_masks=False)) and legacy_settings(hist("a", NYM, 0.9, applied=False, window=1024))
    assert not legacy_settings(hist("a", NYM, 0.9, applied=True, window=1024, stride=256, use_masks=False))
    assert not legacy_settings(hist("a", NYM, 0.9)) and not legacy_settings(hist("a", NYM, 0.9, applied=False))  # default settings
    assert not legacy_settings(hist("a", NYM, 0.9, oversample_number=0.0, lr=1e-5))  # these settings were always applied
    assert not legacy_settings({"run": "a", "score": 0.9}) and not legacy_settings(hist("a", NYM, 0.9, window=None, stride=None))


def test_record_carries_the_settings_marker():
    assert rec("a", 0.8, "2026-10-09 10:00", D1, "k", window=1024, stride=256, settings_applied=True)["settings_applied"] is True
    old = rec("a", 0.8, "2026-10-09 10:00", D1, "k", window=1024, stride=256)
    assert old["settings_applied"] is False and legacy_settings(old)
    assert rec("a", 0.8, "2026-10-09 10:00", D1, "k")["settings_applied"] is False
    assert not legacy_settings(rec("a", 0.8, "2026-10-09 10:00", D1, "k", window=1024, stride=256, settings_applied=True))


def test_render_marks_runs_whose_settings_were_not_applied():
    history = [rec2("old", 0.90, "2026-10-09 10:00", window=1024, stride=256), rec2("plain", 0.85, "2026-10-09 11:00"),
               rec2("new", 0.80, "2026-10-09 12:00", window=1024, stride=256, settings_applied=True)]
    md = render(history, ["old", "plain", "new"], "leaderboard")
    board = md.split("## Leaderboard")[1].split("## Progress")[0]
    assert "| 1 | old † |" in board and "| 2 | plain |" in board and "| 3 | new |" in board
    assert board.count("†") == 2 and "\n† trained before 2026-10-08" in board and "not applied" in board
    assert "†" not in render(history, ["plain", "new"], "leaderboard")  # the old run is not on this leaderboard
    # a scope nothing was scored in: the mark stays on the «not scored on …» row, so the explanation has a run to refer to
    waiting = render(history, ["old", "plain", "new"], "leaderboard", decode=9).split("## Leaderboard")[1].split("## Progress")[0]
    assert "| - | old † | not scored on" in waiting and "| - | plain | not scored on" in waiting and waiting.count("†") == 2
    assert "†" not in render(history[1:], ["old", "plain", "new"], "leaderboard")
