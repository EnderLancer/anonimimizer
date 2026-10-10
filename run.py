"""Єдина точка запуску чотирьох конфігурацій.

python run.py cases.jsonl out.jsonl --mode model        # лише NER-модель (ОСОБА), без правил
python run.py cases.jsonl out.jsonl --mode clean_model  # сирий вихід NER-моделі: без постобробки, фільтрів, правил і кластеризації
python run.py cases.jsonl out.jsonl --mode regex        # лише правила (без нейромережі)
python run.py cases.jsonl out.jsonl --mode model_regex  # NER-модель + правила (найкраще)

Інша NER-модель (див. ner_adapters.py), наприклад з Hugging Face:
python run.py cases.jsonl out.jsonl --ner "hf:dslim/bert-base-NER?aggregation=first" --ner-label LOC=АДРЕСА

Формат вхідного рядка: {"id": ..., "original": "текст"}. Вихід: {"id", "annotations": [{start, end, pseudonym}]}.
"""
import argparse, json, re
import predict_trf as P
import ner_adapters as N
import ner_postprocess as NP

DEFAULT_NER = "spacy:" + P.MODEL


def ner_spans(rows, spec=DEFAULT_NER, label_map=None, raw_spans=False):
    """Сутності моделі для кожного документа: інференс (кешується; для моделі за замовчуванням — у ner_cache.json)
    ──► мітки моделі -> типи челенджу (ner_adapters) ──► складання частин сутностей (ner_postprocess;
    з raw_spans — без нього, для режиму clean_model)."""
    raw = N.run_ner(rows, spec, N.cache_path(spec, DEFAULT_NER, P.CACHE))
    if raw_spans:
        return {str(r["id"]): N.map_labels(raw[str(r["id"])], label_map) for r in rows}
    return {str(r["id"]): NP.process(N.map_labels(raw[str(r["id"])], label_map), r["original"]) for r in rows}


def parse_label_map(items):
    """["@pii", "LOC=АДРЕСА", "ORG="] -> DEFAULT_LABEL_MAP з доповненнями; @назва — набір з N.LABEL_PRESETS,
    порожнє значення вимикає мітку."""
    m = dict(N.DEFAULT_LABEL_MAP)
    for it in items:
        if it.startswith("@"):
            if it[1:] not in N.LABEL_PRESETS:
                raise SystemExit(f"--ner-label {it}: невідомий набір; доступні: {', '.join('@' + k for k in N.LABEL_PRESETS)}")
            m.update(N.LABEL_PRESETS[it[1:]])
            continue
        k, _, v = it.partition("=")
        v = v.strip().upper().replace("I", "І")
        if v and v.rstrip("*") not in N.TYPES:
            raise SystemExit(f"--ner-label {it}: тип має бути одним із {', '.join(N.TYPES)} (з * — лише як частина сусіднього)")
        m[k.strip().upper()] = v or None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--mode", choices=["clean_model", "model", "regex", "model_regex"], default="model_regex")
    ap.add_argument("--ner", default=DEFAULT_NER, help="NER-модель: spacy:<шлях> | hf:<repo>[?опції] | gliner:<repo>[?опції]")
    ap.add_argument("--ner-label", action="append", default=[], metavar="МІТКА=ТИП",
                    help="відповідність мітки моделі типу челенджу, напр. LOC=АДРЕСА; МІТКА= вимикає мітку; "
                         "@pii — набір міток PII-моделей (адреси, номери, дата народження, email)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src, encoding="utf8")]
    ner = ner_spans(rows, a.ner, parse_label_map(a.ner_label), raw_spans=a.mode == "clean_model") if a.mode != "regex" else {}
    with open(a.dst, "w", encoding="utf8") as g:
        for r in rows:
            spans = ner.get(str(r["id"]), [])
            ann = annotate_clean(r["original"], spans) if a.mode == "clean_model" else annotate(r["original"], spans, a.mode)
            g.write(json.dumps({"id": r["id"], "annotations": ann}, ensure_ascii=False) + "\n")


def annotate(text, spans, mode="model_regex"):
    """Анотації одного документа [{start, end, pseudonym}]. spans — сутності моделі з типами челенджу (після ner_spans)."""
    from regex_persons import persons as rx_persons
    pers = [[x.start, x.end, x.text] for x in spans if x.label == "ОСОБА"]          # ОСОБА від моделі (пріоритет)
    extra = [(x.start, x.end, x.label) for x in spans if x.label != "ОСОБА"]       # інші типи, якщо модель їх дає
    if mode == "model":                                                  # лише модель: без regex-осіб і правил
        return P.pseudonymize(text, pers, extra=extra, rules=False)
    for s, e, _ in rx_persons(text):                                     # + regex-особи, що не перетинаються з моделлю
        if not any(s < p[1] and p[0] < e for p in pers):
            pers.append([s, e, text[s:e]])
    return P.pseudonymize(text, pers, extra=extra)


def annotate_clean(text, spans):
    """Режим clean_model: сутності моделі як є. Без постобробки (ner_postprocess), чистки меж, фільтрів держорганів і НОМЕР,
    правил, regex-осіб і кластеризації. Лише мітка -> тип челенджу, відкидання перетинів (лишається довший фрагмент) і
    псевдонім: однакові тексти (без урахування регістру й розділових знаків) одного типу діляться номером."""
    out, kept = [], []
    for x in sorted((x for x in spans if x.label in N.TYPES), key=lambda x: (x.start, -(x.end - x.start))):
        if not any(x.start < o.end and o.start < x.end for o in kept):
            kept.append(x)
    counters, seen = {}, {}
    for x in kept:
        key = (x.label, re.sub(r"\W", "", text[x.start:x.end].lower()))
        if key not in seen:
            counters[x.label] = counters.get(x.label, 0) + 1
            seen[key] = counters[x.label]
        out.append({"start": x.start, "end": x.end, "pseudonym": f"{x.label}_{seen[key]}"})
    return out


if __name__ == "__main__":
    main()
