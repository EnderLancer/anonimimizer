"""Regenerate review.html from saved audit results without loading any models."""
import argparse
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
EMBED_LIMIT = 20 * 1024 * 1024


def report_html(payload):
    template = (HERE / 'audit_review.html').read_text(encoding='utf8')
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return template.replace('__AUDIT_DATA__', data)


def write_report(path, payload):
    path.write_text(report_html(payload), encoding='utf8')


def generate_report(directory, report_format='local'):
    source = directory / 'candidates.jsonl'
    metadata = json.loads((directory / 'summary.json').read_text(encoding='utf8'))
    if report_format == 'auto':
        report_format = 'local' if source.stat().st_size > EMBED_LIMIT else 'embedded'
    payload = {'metadata': metadata, 'documents': []}
    if report_format == 'embedded':
        with source.open(encoding='utf8') as stream:
            payload['documents'] = [json.loads(line) for line in stream if line.strip()]
    else:
        payload.update(server=True, serve_command='python3 additional_tasks/audit/serve_review.py ' + shlex.quote(str(directory)))
    path = directory / 'review.html'
    write_report(path, payload)
    print(f'Review: {path.resolve()} ({report_format}, {path.stat().st_size:,} bytes)', flush=True)
    if report_format == 'local':
        print('Start local viewer: ' + payload['serve_command'], flush=True)
    return report_format


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report-format', choices=('auto', 'embedded', 'local'), default='local',
                        help='local (default) loads pages via serve_review.py; embedded/auto are explicit legacy options')
    args = parser.parse_args()
    generate_report(args.directory, args.report_format)


if __name__ == '__main__':
    main()
