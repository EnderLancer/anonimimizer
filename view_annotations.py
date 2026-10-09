"""Run the production pipeline on selected parquet decisions and write a local HTML viewer."""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path

import run as pipeline
import ner_adapters as N
import ner_postprocess as NP

ROOT = Path(__file__).resolve().parent
TYPES = {"ОСОБА": "person", "АДРЕСА": "address", "НОМЕР": "number", "ІНФОРМАЦІЯ": "information"}


def select_documents(source, ids, limit, text_column="text", id_column="id"):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(source)
    requested = set(ids or [])
    found, rows = set(), []
    for batch in parquet.iter_batches(batch_size=256, columns=[id_column, text_column]):
        for item in batch.to_pylist():
            doc_id, text = str(item[id_column]), item[text_column]
            if requested and doc_id not in requested:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            rows.append({"id": doc_id, "original": text})
            found.add(doc_id)
            if (requested and found == requested) or (not requested and len(rows) >= limit):
                return rows
    if requested - found:
        raise ValueError(f"Не знайдено непорожніх документів із ID: {', '.join(sorted(requested - found))}")
    if not rows:
        raise ValueError("Датасет не містить непорожніх текстів")
    return rows


def render_document(row, annotations, index):
    text, position, fragments, table = row["original"], 0, [], []
    for number, annotation in enumerate(sorted(annotations, key=lambda a: a["start"])):
        start, end = annotation["start"], annotation["end"]
        if not (position <= start < end <= len(text)):
            raise ValueError(f"Некоректні або перетинні межі анотації в документі {row['id']}: {annotation}")
        pseudonym = annotation["pseudonym"]
        kind = pseudonym.rsplit("_", 1)[0]
        css = TYPES[kind]
        target = f"entity-{index}-{number}"
        value = escape(text[start:end])
        fragments.extend([escape(text[position:start]),
                          f'<mark id="{target}" class="{css}" title="{escape(pseudonym)} · {start}–{end}">{value}<small>{escape(pseudonym)}</small></mark>'])
        table.append(f'<tr><td><a href="#{target}">{value}</a></td><td>{escape(pseudonym)}</td><td>{start}–{end}</td></tr>')
        position = end
    fragments.append(escape(text[position:]))
    counts = Counter(a["pseudonym"].rsplit("_", 1)[0] for a in annotations)
    summary = " · ".join(f"{kind}: {counts[kind]}" for kind in TYPES)
    hidden = " hidden" if index else ""
    return f'''<article data-document="{index}"{hidden}>
<h2>Рішення {escape(row['id'])}</h2><p>{summary} · Усього: {len(annotations)}</p>
<div class="text">{''.join(fragments)}</div>
<details><summary>Таблиця анотацій ({len(annotations)})</summary>
<table><thead><tr><th>Знайдений фрагмент</th><th>Псевдонім / категорія</th><th>Позиції</th></tr></thead><tbody>{''.join(table)}</tbody></table></details>
</article>'''


def write_html(path, rows, predictions, metadata):
    options = ''.join(f'<option value="{i}">Рішення {escape(r["id"])} · {len(predictions[r["id"]])} анотацій</option>' for i, r in enumerate(rows))
    articles = ''.join(render_document(r, predictions[r["id"]], i) for i, r in enumerate(rows))
    legend = ''.join(f'<span class="{css}">{kind}</span>' for kind, css in TYPES.items())
    path.write_text(f'''<!doctype html><html lang="uk"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Анотації основного пайплайну</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;color:#213047;background:#f5f7fb}}header,main{{max-width:1100px;margin:auto;padding:24px}}header{{background:white}}h1{{font-size:26px}}h2{{font-size:21px}}p{{color:#526079}}select,button{{font:inherit;padding:9px;border:1px solid #bcc9dc;border-radius:6px;background:white}}.legend{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}.legend span{{padding:5px 9px;border-radius:5px}}.person{{background:#ffd5dc}}.address{{background:#ccefdc}}.number{{background:#d5e5ff}}.information{{background:#ffe4b5}}.text{{white-space:pre-wrap;overflow-wrap:anywhere;line-height:2.1;background:white;border:1px solid #d9e0eb;border-radius:9px;padding:24px}}mark{{padding:2px;border-radius:3px;color:inherit;scroll-margin-top:20px}}mark small{{font-size:10px;margin-left:5px;font-weight:650;white-space:nowrap}}.hide-labels mark small{{display:none}}mark:target{{outline:3px solid #204ba8}}details{{margin-top:20px}}table{{width:100%;border-collapse:collapse;background:white;margin-top:12px}}td,th{{text-align:left;border-bottom:1px solid #d9e0eb;padding:10px;overflow-wrap:anywhere}}a{{color:#204ba8}}[hidden]{{display:none}}
</style><header><h1>Анотації основного пайплайну</h1>
<p>Режим: {escape(metadata['mode'])} · Модель: {escape(metadata['ner'] or 'не використовується')}</p>
<label>Обрати документ <select id="document">{options}</select></label>
<label><input id="labels" type="checkbox" checked> Показати псевдоніми</label>
<div class="legend">{legend}</div><p>Фінальні анотації run.annotate. Наведіть на фрагмент, щоб побачити псевдонім і позиції. У таблиці можна натиснути фрагмент і перейти до нього в тексті.</p>
</header><main>{articles}</main><script>
document.getElementById('document').addEventListener('change',event=>{{
    for(const article of document.querySelectorAll('article'))article.hidden=article.dataset.document!==event.target.value;
    history.replaceState(null,'',location.pathname+location.search);
}});
document.getElementById('labels').addEventListener('change',event=>document.body.classList.toggle('hide-labels',!event.target.checked));
</script></html>''', encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", type=Path, default=ROOT / "data/train/2024-court-decisions.parquet")
    parser.add_argument("--id", action="append", help="Точний ID документа; можна повторювати")
    parser.add_argument("--limit", type=int, default=5, help="Перші N непорожніх документів, якщо --id не задано")
    parser.add_argument("--mode", choices=["model", "regex", "model_regex", "full"], default="full")
    parser.add_argument("--ner", default=pipeline.DEFAULT_NER)
    parser.add_argument("--ner-label", action="append", default=[])
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--out", type=Path, default=ROOT / "out/annotations")
    parser.add_argument("--annotations", type=Path, help="Готовий JSONL із run.py: рендер без повторного інференсу")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit має бути додатним")
    saved = None
    if args.annotations:
        with args.annotations.open(encoding="utf8") as source:
            saved = {str(r["id"]): r["annotations"] for r in (json.loads(line) for line in source if line.strip())}
        if not saved:
            parser.error("Файл анотацій порожній")
    try:
        rows = select_documents(args.src, args.id or (list(saved) if saved is not None else None), args.limit, args.text_column, args.id_column)
    except ValueError as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"mode": args.mode if saved is None else "готові анотації", "ner": args.ner if saved is None and args.mode != "regex" else None,
                "source": str(args.src.resolve()), "ner_labels": args.ner_label}
    if saved is not None:
        missing = {r["id"] for r in rows} - saved.keys()
        if missing:
            parser.error(f"У файлі анотацій немає ID: {', '.join(sorted(missing))}")
        predictions = saved
    else:
        print(f"Документів: {len(rows)}; режим: {args.mode}", flush=True)
        ner = {}
        if args.mode != "regex":
            # Text-aware keys keep parquet predictions separate from the test-set cache.
            keyed = [{"id": hashlib.sha256((args.ner + '\0' + r['id'] + '\0' + r['original']).encode()).hexdigest(), "original": r["original"]} for r in rows]
            raw = N.run_ner(keyed, args.ner, str(args.out / "ner_cache.json"))
            labels = pipeline.parse_label_map(args.ner_label)
            ner = {r["id"]: NP.process(N.map_labels(raw[k["id"]], labels), r["original"]) for r, k in zip(rows, keyed)}
        tagger = pipeline.load_tagger() if args.mode == "full" else None
        predictions = {r["id"]: pipeline.annotate(r["original"], ner.get(r["id"], []), args.mode, tagger) for r in rows}
    for name, records in [("input.jsonl", rows), ("annotations.jsonl", [{"id": r["id"], "annotations": predictions[r["id"]]} for r in rows])]:
        with (args.out / name).open("w", encoding="utf8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf8")
    write_html(args.out / "review.html", rows, predictions, metadata)
    print(f"HTML: {(args.out / 'review.html').resolve()}")


if __name__ == "__main__":
    main()
