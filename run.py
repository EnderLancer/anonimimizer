"""Єдина точка запуску чотирьох конфігурацій.

python run.py cases.jsonl out.jsonl --mode model        # лише NER-модель (ОСОБА), без правил і CRF
python run.py cases.jsonl out.jsonl --mode regex        # лише правила (без нейромережі і CRF)
python run.py cases.jsonl out.jsonl --mode model_regex  # NER-модель + правила
python run.py cases.jsonl out.jsonl --mode full         # NER-модель + правила + CRF (найкраще)

Інша NER-модель (див. ner_adapters.py), наприклад з Hugging Face:
python run.py cases.jsonl out.jsonl --ner "hf:dslim/bert-base-NER?aggregation=first" --ner-label LOC=АДРЕСА

Формат вхідного рядка: {"id": ..., "original": "текст"}. Вихід: {"id", "annotations": [{start, end, pseudonym}]}.
"""
import argparse, json, os, re
import predict_trf as P
import ner_adapters as N
import ner_postprocess as NP

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NER = "spacy:" + P.MODEL


def ner_spans(rows, spec=DEFAULT_NER, label_map=None):
    """Сутності моделі для кожного документа: інференс (кешується; для моделі за замовчуванням — у ner_cache.json)
    ──► мітки моделі -> типи челенджу (ner_adapters) ──► складання частин сутностей (ner_postprocess)."""
    raw = N.run_ner(rows, spec, N.cache_path(spec, DEFAULT_NER, P.CACHE))
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


INI_END = re.compile(r"(?<![\w'’ʼ-])[А-ЯІЇЄҐ]\.\s?(?:[А-ЯІЇЄҐ]\.)?$")


def merge_persons(text, model, regex):
    """Об'єднує фрагменти моделі з regex-ПІБ, що з ними перетинаються, і склеює сусідні фрагменти осіб.
    Модель часто губить прізвище перед «Ім'я По-батькові», по батькові чи ініціали; regex-спан їх повертає.
    Не розширює вправо після «І. П.» (там починається наступне речення)."""
    spans = [[s, e] for s, e, *_ in model]
    for s, e, _ in regex:
        hit = [p for p in spans if s < p[1] and p[0] < e]
        if not hit:
            spans.append([s, e]); continue
        lo, hi = min([s] + [p[0] for p in hit]), max([e] + [p[1] for p in hit])
        right = max(p[1] for p in hit)
        if hi > right and INI_END.search(text[lo:right]):
            hi = right                                               # «Іванов І. П.» + «І. П. Суд» -> без «Суд»
        spans = [p for p in spans if p not in hit] + [[lo, hi]]
    spans.sort()
    out = []
    for s, e in spans:
        if out and (s <= out[-1][1] or (re.fullmatch(r"[ \t]+", text[out[-1][1]:s])
                                        and not INI_END.search(text[out[-1][0]:out[-1][1]]))):
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [[s, e, text[s:e]] for s, e in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--mode", choices=["model", "regex", "model_regex", "full"], default="full")
    ap.add_argument("--ner", default=DEFAULT_NER, help="NER-модель: spacy:<шлях> | hf:<repo>[?опції] | gliner:<repo>[?опції]")
    ap.add_argument("--ner-label", action="append", default=[], metavar="МІТКА=ТИП",
                    help="відповідність мітки моделі типу челенджу, напр. LOC=АДРЕСА; МІТКА= вимикає мітку; "
                         "@pii — набір міток PII-моделей (адреси, номери, дата народження, email)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src, encoding="utf8")]
    ner = ner_spans(rows, a.ner, parse_label_map(a.ner_label)) if a.mode != "regex" else {}
    tagger = load_tagger() if a.mode == "full" else None
    with open(a.dst, "w", encoding="utf8") as g:
        for r in rows:
            ann = annotate(r["original"], ner.get(str(r["id"]), []), a.mode, tagger)
            g.write(json.dumps({"id": r["id"], "annotations": ann}, ensure_ascii=False) + "\n")


def load_tagger():
    import pycrfsuite
    tagger = pycrfsuite.Tagger(); tagger.open(os.path.join(HERE, "crf2.model"))
    return tagger


def annotate(text, spans, mode="full", tagger=None):
    """Анотації одного документа [{start, end, pseudonym}]. spans — сутності моделі з типами челенджу (після ner_spans),
    tagger — CRF (load_tagger) для режиму full."""
    from regex_persons import persons as rx_persons
    pers = [[x.start, x.end, x.text] for x in spans if x.label == "ОСОБА"]          # ОСОБА від моделі (пріоритет)
    extra = [(x.start, x.end, x.label) for x in spans if x.label != "ОСОБА"]       # інші типи, якщо модель їх дає
    if mode == "model":                                                  # лише модель: без regex-осіб і правил
        return P.pseudonymize(text, pers, extra=extra, rules=False)
    pers = merge_persons(text, pers, rx_persons(text))                  # модель ∪ regex-ПІБ
    if mode == "full" and tagger:
        import predict
        extra += [x for x in predict.predict_doc(tagger, text) if x[2] in ("НОМЕР", "ІНФОРМАЦІЯ", "АДРЕСА")]
    return P.pseudonymize(text, pers, extra=extra)


if __name__ == "__main__":
    main()
