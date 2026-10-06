# AGENTS.md

Guidance for coding agents. README.md describes the models, modes and results.

## Rules

- **Do not modify `data/`.** `data/evaluate.py` is the official scorer and `data/test/` is the expected result.
- **The pseudonymizer never sees the expected result.** It reads `out/inputs.jsonl` (`id` + `original`, made by
  `scripts/split_test_data.py`). Do not use `annotations`/`pseudonymized` from `data/test/` for training or tuning.
- **Keep the stages separate:**
  1. `solution/detect.py` — candidates without pseudonyms: `start`, `end`, `type`, `text`, `metadata.source`
     (`ner`, `regex_persons`, `rules`, `crf`).
  2. `solution/combine.py` — merge + numbering (`pseudonymizer.pipeline.combine`); keeps metadata, adds `pseudonym`
     and `pseudonymized`.
  3. `scripts/to_submission.py` — strips everything except `id`, `start`, `end`, `pseudonym`.
  4. `data/evaluate.py` — scores the stripped file only.
- **`predict.py` and the staged flow must agree.** `build_annotations` is `strip(combine(candidates(...)))`; change the
  merge policy only in `combine`. CI fails if the staged regex-mode output differs from `predict.py --mode regex`.
- Offsets are into `original`, `end` exclusive. Code comments and docstrings are in Ukrainian.

## Checking a change

Fast loop (mode 1, only regex, a few seconds, no dependencies):

```bash
mkdir -p out
python scripts/split_test_data.py data/test/court_cases.jsonl out/inputs.jsonl
python solution/detect.py out/inputs.jsonl out/candidates.jsonl --mode regex
python solution/combine.py out/candidates.jsonl out/linked.jsonl
python scripts/to_submission.py out/linked.jsonl out/submission.jsonl
python data/evaluate.py data/test/court_cases.jsonl out/submission.jsonl
```

Mode 2 (model + regex) drops `--mode regex`, needs `solution/requirements.txt`, ~7 GB RAM and ~14 min on CPU.
Report the before/after score and per-type table; compare with `results/results.json`.
