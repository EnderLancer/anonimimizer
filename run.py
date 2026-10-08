"""Єдина точка запуску чотирьох конфігурацій.

python run.py cases.jsonl out.jsonl --mode model        # лише NER-модель (ОСОБА), без правил і CRF
python run.py cases.jsonl out.jsonl --mode regex        # лише правила (без нейромережі і CRF)
python run.py cases.jsonl out.jsonl --mode model_regex  # NER-модель + правила
python run.py cases.jsonl out.jsonl --mode full         # NER-модель + правила + CRF (найкраще)

Інша NER-модель (див. ner_adapters.py), наприклад з Hugging Face:
python run.py cases.jsonl out.jsonl --ner "hf:dslim/bert-base-NER?aggregation=first" --ner-label LOC=АДРЕСА

Формат вхідного рядка: {"id": ..., "original": "текст"}. Вихід: {"id", "annotations": [{start, end, pseudonym}]}.
"""
import argparse, json, os
import predict_trf as P
import ner_adapters as N

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NER = "spacy:" + P.MODEL


def ner_spans(rows, spec=DEFAULT_NER, label_map=None):
    """Фрагменти моделі, зведені до типів челенджу; результат кешується (для моделі за замовчуванням — у ner_cache.json)."""
    raw = N.run_ner(rows, spec, N.cache_path(spec, DEFAULT_NER, P.CACHE))
    return {str(r["id"]): N.map_labels(raw[str(r["id"])], label_map, r["original"]) for r in rows}


def parse_label_map(items):
    """["LOC=АДРЕСА", "ORG="] -> DEFAULT_LABEL_MAP з доповненнями; порожнє значення вимикає мітку."""
    m = dict(N.DEFAULT_LABEL_MAP)
    for it in items:
        k, _, v = it.partition("=")
        v = v.strip().upper().replace("I", "І")
        if v and v not in N.TYPES:
            raise SystemExit(f"--ner-label {it}: тип має бути одним із {', '.join(N.TYPES)}")
        m[k.strip().upper()] = v or None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--mode", choices=["model", "regex", "model_regex", "full"], default="full")
    ap.add_argument("--ner", default=DEFAULT_NER, help="NER-модель: spacy:<шлях> | hf:<repo>[?опції] | gliner:<repo>[?опції]")
    ap.add_argument("--ner-label", action="append", default=[], metavar="МІТКА=ТИП",
                    help="відповідність мітки моделі типу челенджу, напр. LOC=АДРЕСА; МІТКА= вимикає мітку")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src, encoding="utf8")]
    from regex_persons import persons as rx_persons
    ner = ner_spans(rows, a.ner, parse_label_map(a.ner_label)) if a.mode != "regex" else {}
    tagger = None
    if a.mode == "full":
        import pycrfsuite, predict
        tagger = pycrfsuite.Tagger(); tagger.open(os.path.join(HERE, "crf2.model"))
    with open(a.dst, "w", encoding="utf8") as g:
        for r in rows:
            text = r["original"]
            spans = ner.get(str(r["id"]), [])
            pers = [[x.start, x.end, x.text] for x in spans if x.label == "ОСОБА"]          # ОСОБА від моделі (пріоритет)
            extra = [(x.start, x.end, x.label) for x in spans if x.label != "ОСОБА"]       # інші типи, якщо модель їх дає
            if a.mode == "model":                                        # лише модель: без regex-осіб і правил
                g.write(json.dumps({"id": r["id"], "annotations": P.pseudonymize(text, pers, extra=extra, rules=False)}, ensure_ascii=False) + "\n")
                continue
            for s, e, _ in rx_persons(text):                             # + regex-особи, що не перетинаються з моделлю
                if not any(s < p[1] and p[0] < e for p in pers):
                    pers.append([s, e, text[s:e]])
            if tagger:
                extra += [x for x in predict.predict_doc(tagger, text) if x[2] in ("НОМЕР", "ІНФОРМАЦІЯ", "АДРЕСА")]
            g.write(json.dumps({"id": r["id"], "annotations": P.pseudonymize(text, pers, extra=extra)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
