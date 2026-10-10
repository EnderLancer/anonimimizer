"""Serve existing audit JSONL locally, loading only one page at a time (stdlib only)."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from additional_tasks.audit.generate_review import report_html

HERE = Path(__file__).resolve().parent
INDEX_VERSION = 1
STATUSES = {'pending', 'personal', 'service', 'false', 'uncertain'}


def build_index(directory):
    source = directory / 'candidates.jsonl'
    stat = source.stat()
    signature = json.dumps([INDEX_VERSION, stat.st_size, stat.st_mtime_ns])
    target = directory / 'review-index.sqlite3'
    if target.exists():
        with closing(sqlite3.connect(target)) as db:
            if db.execute('SELECT signature FROM metadata').fetchone()[0] == signature:
                return target
    temporary = target.with_suffix('.building')
    temporary.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(temporary)) as db, db, source.open('rb') as stream:
            db.executescript('''
                CREATE TABLE metadata(signature TEXT);
                CREATE TABLE documents(n INTEGER PRIMARY KEY, id TEXT, offset INTEGER,
                    length INTEGER, unique_counts TEXT, personal INTEGER, service INTEGER);
                CREATE TABLE candidates(n INTEGER PRIMARY KEY, doc INTEGER, ordinal INTEGER,
                    key TEXT, type TEXT, excluded INTEGER, search TEXT);
            ''')
            db.execute('INSERT INTO metadata VALUES (?)', (signature,))
            n = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                candidates = row['candidates']
                service = sum(bool(c.get('excluded')) for c in candidates)
                db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?)',
                           (n, str(row['id']), offset, len(line), json.dumps(row.get('unique_counts')),
                            len(candidates) - service, service))
                db.executemany('INSERT INTO candidates(doc,ordinal,key,type,excluded,search) VALUES (?,?,?,?,?,?)',
                               [(n, i, c['key'], c['type'], bool(c.get('excluded')),
                                 ' '.join([str(row['id']), c['value'], c.get('before', ''), c.get('after', '')]).casefold())
                                for i, c in enumerate(candidates)])
                n += 1
                if n % 1000 == 0:
                    print(f'Indexed {n} documents ({stream.tell() / stat.st_size:.0%})', flush=True)
            db.executescript('''
                CREATE INDEX candidate_filters ON candidates(excluded,type,doc);
                CREATE INDEX candidate_documents ON candidates(doc);
                CREATE INDEX candidate_keys ON candidates(key);
            ''')
        if source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
            raise RuntimeError('candidates.jsonl changed during indexing; retry after the audit finishes')
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Index ready: {n} documents', flush=True)
    return target


class ReviewStore:
    def __init__(self, directory, index):
        self.source = directory / 'candidates.jsonl'
        self.index = index
        self.metadata = json.loads((directory / 'summary.json').read_text(encoding='utf8'))

    def document(self, n, db):
        row = db.execute('SELECT offset,length FROM documents WHERE n=?', (n,)).fetchone()
        if row is None:
            raise ValueError('Unknown document')
        with self.source.open('rb') as stream:
            stream.seek(row[0])
            return json.loads(stream.read(row[1]))

    def query(self, params):
        page = max(0, int(params.get('page', 0)))
        kind, typ, status = params.get('kind', 'personal'), params.get('type', ''), params.get('status', '')
        q = str(params.get('q', '')).strip().casefold()
        reviews = params.get('reviews', {})
        with closing(sqlite3.connect(self.index)) as db:
            db.row_factory = sqlite3.Row
            db.execute('CREATE TEMP TABLE reviews(key TEXT PRIMARY KEY,status TEXT)')
            db.executemany('INSERT INTO reviews VALUES (?,?)',
                           [(key, r.get('status')) for key, r in reviews.items()
                            if isinstance(r, dict) and r.get('status') in STATUSES])
            clauses, args = [], []
            if kind != 'all':
                clauses.append('c.excluded=?')
                args.append(int(kind == 'service'))
            if typ:
                clauses.append('c.type=?')
                args.append(typ)
            if status:
                clauses.append("COALESCE(r.status,'pending')=?")
                args.append(status)
            if q:
                clauses.append('instr(c.search,?)>0')
                args.append(q)
            where = ' AND '.join(clauses) or '1'
            matched = f"SELECT c.* FROM candidates c LEFT JOIN reviews r ON r.key=c.key WHERE {where}"
            documents = params.get('view') == 'documents'
            if documents:
                sql = f'SELECT d.* FROM documents d WHERE EXISTS (SELECT 1 FROM ({matched}) m WHERE m.doc=d.n)'
                # Retain decisions with no findings in the personal/all document view.
                if kind != 'service' and not typ and not status:
                    sql += ' OR (d.personal=0 AND d.service=0 AND instr(lower(d.id),?)>0)'
                    args.append(q)
            else:
                sql = matched
            total = db.execute(f'SELECT count(*) FROM ({sql})', args).fetchone()[0]
            page = min(page, max(0, (total - 1) // 40))
            rows = db.execute(sql + ' ORDER BY n LIMIT 40 OFFSET ?', [*args, page * 40]).fetchall()
            items, cache = [], {}
            for row in rows:
                if documents:
                    items.append({'id': row['id'], '_n': row['n'], 'unique_counts': json.loads(row['unique_counts']),
                                  'personal_count': row['personal'], 'service_count': row['service']})
                else:
                    n = row['doc']
                    if n not in cache:
                        cache[n] = self.document(n, db)
                    doc = cache[n]
                    items.append({**doc['candidates'][row['ordinal']], 'doc': {'id': doc['id'], '_n': n}})
            done = db.execute("SELECT count(*) FROM candidates c JOIN reviews r ON r.key=c.key WHERE c.excluded=0 AND r.status!='pending'").fetchone()[0]
            return {'items': items, 'total': total, 'page': page, 'done': done}


def handler_for(store, port):
    html = report_html({'metadata': store.metadata, 'documents': [], 'server': True}).encode('utf8')

    class Handler(BaseHTTPRequestHandler):
        def respond(self, content, kind='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def allowed(self):
            # Loopback binding plus Host/Origin checks prevent other sites from reading the corpus.
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            origin = self.headers.get('Origin')
            return self.headers.get('Host') in hosts and (origin is None or origin in {'http://' + h for h in hosts})

        def do_GET(self):
            if not self.allowed():
                return self.respond(b'Forbidden', status=403)
            url = urlparse(self.path)
            if url.path in ('/', '/review.html'):
                return self.respond(html, 'text/html')
            if url.path == '/api/document':
                try:
                    n = int(parse_qs(url.query)['n'][0])
                    with closing(sqlite3.connect(store.index)) as db:
                        result = store.document(n, db)
                    return self.respond(json.dumps(result, ensure_ascii=False).encode('utf8'))
                except (KeyError, ValueError):
                    return self.respond(b'Unknown document', status=404)
            self.respond(b'Not found', status=404)

        def do_POST(self):
            if not self.allowed():
                return self.respond(b'Forbidden', status=403)
            if self.path != '/api/query':
                return self.respond(b'Not found', status=404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                result = store.query(json.loads(self.rfile.read(length)))
                self.respond(json.dumps(result, ensure_ascii=False).encode('utf8'))
            except (ValueError, TypeError, AttributeError):
                self.respond(b'Invalid request', status=400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Existing audit output directory')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    index = build_index(args.directory)
    store = ReviewStore(args.directory, index)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(store, args.port))
    print(f'Open http://127.0.0.1:{args.port}/ (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
