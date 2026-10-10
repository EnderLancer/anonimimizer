"""Calculate presentation metrics from saved pipeline audit results (stdlib only)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex


CATEGORIES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
HERE = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"(?<!\w)(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_\d+(?!\w)")


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def calculate_metrics(rows):
    """Sum document-local identities; original masks and new pseudonyms are separate.

    A leaked value cannot reliably be linked to an existing mask. The entity
    share is therefore an estimate, not a measured anonymization failure rate.
    """
    documents = documents_with_leaks = 0
    documents_by_category, leaked, anonymized = Counter(), Counter(), Counter()
    for row in rows:
        if not isinstance(row.get("text"), str):
            raise ValueError("Each document must contain its published text")
        if not isinstance(row.get("candidates"), list):
            raise ValueError("Each document must contain a candidates list")
        masks = {match.group() for match in PLACEHOLDER.finditer(row["text"])}
        anonymized.update(mask.rsplit("_", 1)[0] for mask in masks)
        identities = set()
        for candidate in row["candidates"]:
            kind = candidate.get("type")
            pseudonym = candidate.get("pseudonym")
            if kind not in CATEGORIES:
                raise ValueError(f"Unsupported candidate type: {kind!r}")
            if (not isinstance(pseudonym, str)
                    or PLACEHOLDER.fullmatch(pseudonym) is None
                    or pseudonym.rsplit("_", 1)[0] != kind):
                raise ValueError("Candidates need pipeline pseudonyms; raw results are unsupported")
            if not isinstance(candidate.get("excluded"), bool):
                raise ValueError("Candidates must have a boolean service exclusion status")
            if not candidate["excluded"]:
                identities.add((kind, pseudonym))
        kinds = {kind for kind, _ in identities}
        documents += 1
        documents_with_leaks += bool(kinds)
        documents_by_category.update(kinds)
        leaked.update(kind for kind, _ in identities)

    if not documents:
        raise ValueError("No documents found in candidates.jsonl")
    categories = {}
    for kind in CATEGORIES:
        total = leaked[kind] + anonymized[kind]
        categories[kind] = {
            "documents_with_leaks": documents_by_category[kind],
            "documents_with_leaks_pct": percentage(documents_by_category[kind], documents),
            "leaked_unique": leaked[kind],
            "anonymized_unique": anonymized[kind],
            "total_unique": total,
            "leaked_unique_share_pct": percentage(leaked[kind], total),
        }
    return {
        "status": "potential_leaks_after_service_filter",
        "unique_count_scope": "document",
        "documents": documents,
        "documents_with_leaks": documents_with_leaks,
        "documents_with_leaks_pct": percentage(documents_with_leaks, documents),
        "categories": categories,
        "definitions": {
            "documents_with_leaks_pct": "100 * documents_with_leaks / documents",
            "leaked_unique_share_pct": "100 * leaked_unique / (leaked_unique + anonymized_unique)",
            "leaked_unique": "Distinct non-excluded pipeline pseudonyms within each document, summed across documents",
            "anonymized_unique": "Distinct existing placeholders in published text within each document, summed across documents",
            "zero_denominator": "null (not applicable)",
            "limitation": "Open values and existing masks may refer to the same entity; they are counted separately. Only detected open values are counted. Manual reviews are not applied.",
        },
    }


def read_rows(path):
    with path.open(encoding="utf8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a document object on line {line_number}")
            yield row


def format_pct(value):
    return "—" if value is None else f"{value:.2f}%"


def markdown_report(metrics):
    source = Path(metrics["source"])
    try:
        source_label = source.relative_to(HERE.parents[1]).as_posix()
    except ValueError:
        source_label = source.name
    command = ("python3 additional_tasks/audit/calculate_metrics.py "
               + shlex.quote(str(Path(source_label).parent)) + " --md")
    lines = [
        "# Метрики аудиту судових рішень ЄДРСР", "",
        "Результати автоматичного пошуку **потенційних витоків** після службового фільтра. "
        "Ручні оцінки не враховані.", "",
        f"Джерело результатів: `{source_label}`.", "",
        "## Рішення з потенційними витоками", "",
        f"Перевірено рішень: **{metrics['documents']}**. "
        f"Хоча б одну знахідку мають **{metrics['documents_with_leaks']}** "
        f"(**{format_pct(metrics['documents_with_leaks_pct'])}**).", "",
        "| Категорія | Рішень зі знахідками | Частка всіх перевірених рішень |",
        "|---|---:|---:|",
    ]
    for kind, values in metrics["categories"].items():
        lines.append(f"| {kind} | {values['documents_with_leaks']} | "
                     f"{format_pct(values['documents_with_leaks_pct'])} |")
    lines.extend([
        "", "Одне рішення може містити кілька категорій, тому відсотки за категоріями не сумуються до 100%.",
        "", "## Частка відкритих унікальних даних", "",
        "| Категорія | Відкритих унікальних | Замаскованих унікальних | Разом | Частка відкритих |",
        "|---|---:|---:|---:|---:|",
    ])
    for kind, values in metrics["categories"].items():
        lines.append(f"| {kind} | {values['leaked_unique']} | {values['anonymized_unique']} | "
                     f"{values['total_unique']} | {format_pct(values['leaked_unique_share_pct'])} |")
    lines.extend([
        "", "## Методика та обмеження", "",
        "- Частка рішень: `100 × рішення зі знахідками / усі перевірені рішення`. "
        "Знаменник включає рішення без знахідок.",
        "- Частка відкритих даних: `100 × відкриті / (відкриті + замасковані)`, окремо за категоріями.",
        "- Відкриті сутності — унікальні псевдоніми пайплайну серед невиключених кандидатів.",
        "- Замасковані сутності — унікальні наявні маски `ОСОБА_n`, `АДРЕСА_n`, "
        "`НОМЕР_n`, `ІНФОРМАЦІЯ_n` у вихідному тексті.",
        "- Унікальність визначається в межах рішення, після чого кількості сумуються. "
        "Та сама сутність у різних рішеннях може рахуватися повторно.",
        "- Відкриті значення й маски можуть стосуватися тієї самої сутності, але їх зіставлення "
        "не проводиться. Частка є наближеною оцінкою серед виявлених даних; детектор може пропускати витоки.",
        "- `—` означає, що категорія не має ані відкритих даних, ані масок (нульовий знаменник).",
        "", "## Параметри аудиту", "", "```json",
        json.dumps(metrics.get("audit_metadata", {}), ensure_ascii=False, indent=2), "```",
        "", "## Відтворення звіту", "",
        "Запустіть із кореня репозиторію за наявності вихідного `candidates.jsonl`:",
        "", "```bash", command, "```", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Audit output directory containing candidates.jsonl")
    parser.add_argument("--out", type=Path, help="Output JSON path (default: DIRECTORY/metrics.json)")
    parser.add_argument("--md", nargs="?", type=Path, const=HERE / "METRICS.md",
                        help="Also write Markdown (default path when flag is used: additional_tasks/audit/METRICS.md)")
    args = parser.parse_args()
    source = args.directory / "candidates.jsonl"
    output = args.out or args.directory / "metrics.json"
    if output.resolve() in {source.resolve(), (args.directory / "summary.json").resolve()}:
        parser.error("Output must not overwrite audit inputs")
    if args.md and args.md.resolve() in {
            source.resolve(), (args.directory / "summary.json").resolve(), output.resolve()}:
        parser.error("Markdown output must not overwrite audit inputs or JSON output")
    try:
        summary_path = args.directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf8")) if summary_path.exists() else {}
        if summary.get("candidate_source", "pipeline") != "pipeline":
            raise ValueError("Only pipeline audit results support unique-entity metrics")
        metrics = calculate_metrics(read_rows(source))
        if "selected" in summary and summary["selected"] != metrics["documents"]:
            raise ValueError("Document count differs from summary.json; check audit completeness")
        metrics["source"] = str(source.resolve())
        metrics["audit_metadata"] = {key: summary[key] for key in
                                     ("mode", "ner", "seed", "total", "candidate_source", "service_filter_version")
                                     if key in summary}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        if args.md:
            args.md.parent.mkdir(parents=True, exist_ok=True)
            args.md.write_text(markdown_report(metrics), encoding="utf8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print("Потенційні витоки після службового фільтра; ручні оцінки не враховано.")
    print(f"Рішень із ≥1 знахідкою: {metrics['documents_with_leaks']} / {metrics['documents']} "
          f"({format_pct(metrics['documents_with_leaks_pct'])})")
    print("Категорія | Рішень зі знахідками | % рішень | Відкритих унікальних | Масок унікальних | Частка відкритих")
    for kind, values in metrics["categories"].items():
        print(f"{kind} | {values['documents_with_leaks']} | {format_pct(values['documents_with_leaks_pct'])} | "
              f"{values['leaked_unique']} | {values['anonymized_unique']} | "
              f"{format_pct(values['leaked_unique_share_pct'])}")
    print("Частка відкритих = відкриті / (відкриті + маски); унікальність у межах рішення.")
    print(f"JSON: {output.resolve()}")
    if args.md:
        print(f"Markdown: {args.md.resolve()}")


if __name__ == "__main__":
    main()
