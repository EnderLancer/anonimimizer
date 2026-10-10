"""Step dispatcher for scripts/make_silver.py."""
from __future__ import annotations

import logging
import time

log = logging.getLogger("silver")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def run(a) -> None:
    steps = [a.step] if a.step != "all" else ["filter", "fill", "label"] + ([] if a.skip_ner else ["ner"]) + ["excerpt", "audit", "samples", "split", "export", "report"]
    for step in steps:
        t0 = time.time()
        log.info("== %s ==", step)
        if step == "filter":
            from silver.filter import run_filter
            run_filter(limit=a.limit, negatives=a.negatives)
        elif step == "fill":
            from silver.fill.run import run_fill
            run_fill(seed=a.seed, limit=a.limit)
        elif step == "label":
            from silver.label import run_label
            run_label()
        elif step == "ner":
            from silver.ner import run_ner
            run_ner(batch=a.batch, limit=a.limit)
        elif step == "excerpt":
            from silver.excerpt import run_excerpt
            run_excerpt()
        elif step == "audit":
            from silver.audit import run_audit
            run_audit(seed=a.seed)
        elif step == "samples":
            from silver.audit import run_samples, score_samples
            score_samples() if a.score else run_samples(seed=a.seed)
        elif step == "split":
            from silver.split import run_split
            run_split()
        elif step == "export":
            from silver.export import run_export
            run_export()
        elif step == "report":
            from silver.report import run_report
            run_report()
        elif step == "smoke":
            from silver.smoke import run_smoke
            run_smoke()
        log.info("== %s done in %.1fs ==", step, time.time() - t0)
