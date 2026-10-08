"""Єдина точка запуску чотирьох конфігурацій.

python run.py cases.jsonl out.jsonl --mode model        # лише NER-модель (ОСОБА), без правил і CRF
python run.py cases.jsonl out.jsonl --mode regex        # лише правила (без нейромережі і CRF)
python run.py cases.jsonl out.jsonl --mode model_regex  # NER-модель + правила
python run.py cases.jsonl out.jsonl --mode full         # NER-модель + правила + CRF (найкраще)

Формат вхідного рядка: {"id": ..., "original": "текст"}. Вихід: {"id", "annotations": [{start, end, pseudonym}]}.
"""
import argparse, json, os
import predict_trf as P

HERE = os.path.dirname(os.path.abspath(__file__))


def ner_spans(rows):
    """Фрагменти ОСОБА від spaCy-моделі; результат кешується в ner_cache.json."""
    cache = json.load(open(P.CACHE, encoding="utf8")) if os.path.exists(P.CACHE) else {}
    todo = [r for r in rows if str(r["id"]) not in cache]
    if todo:
        import spacy
        nlp = spacy.load(P.MODEL)
        for r in todo:
            cache[str(r["id"])] = P.trf_spans(nlp, [r["original"]])[0]
        json.dump(cache, open(P.CACHE, "w", encoding="utf8"), ensure_ascii=False)
    return {str(r["id"]): cache[str(r["id"])] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--mode", choices=["model", "regex", "model_regex", "full"], default="full")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src, encoding="utf8")]
    from regex_persons import persons as rx_persons
    ner = ner_spans(rows) if a.mode != "regex" else {}
    tagger = None
    if a.mode == "full":
        import pycrfsuite, predict
        tagger = pycrfsuite.Tagger(); tagger.open(os.path.join(HERE, "crf2.model"))
    with open(a.dst, "w", encoding="utf8") as g:
        for r in rows:
            text = r["original"]
            pers = [list(p) for p in ner.get(str(r["id"]), [])]          # ОСОБА від моделі (пріоритет)
            if a.mode == "model":                                        # лише модель: без regex-осіб і правил
                g.write(json.dumps({"id": r["id"], "annotations": P.pseudonymize(text, pers, rules=False)}, ensure_ascii=False) + "\n")
                continue
            for s, e, _ in rx_persons(text):                             # + regex-особи, що не перетинаються з моделлю
                if not any(s < p[1] and p[0] < e for p in pers):
                    pers.append([s, e, text[s:e]])
            extra = []
            if tagger:
                extra = [x for x in predict.predict_doc(tagger, text) if x[2] in ("НОМЕР", "ІНФОРМАЦІЯ", "АДРЕСА")]
            g.write(json.dumps({"id": r["id"], "annotations": P.pseudonymize(text, pers, extra=extra)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
