"""Regression checks for pagination over saved results, without model dependencies."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from additional_tasks.audit.generate_review import generate_report
from additional_tasks.audit.serve_review import ReviewStore, build_index


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.rows = []
        for n in range(43):
            text = '😀 Суддя Петренко та учасник Іваненко </script>'
            cs = [] if n == 42 else [
                {'key': f'{n}-s', 'type': 'ОСОБА', 'start': 8, 'end': 16, 'value': 'Петренко', 'before': 'Суддя ', 'after': '', 'excluded': True, 'exclusion_reason': 'Роль судді', 'role': 'суддя', 'sources': ['pipeline']},
                {'key': f'{n}-p', 'type': 'ОСОБА', 'start': 28, 'end': 36, 'value': 'Іваненко', 'before': 'учасник ', 'after': '', 'excluded': False, 'sources': ['pipeline']},
            ]
            self.rows.append({'id': str(n), 'text': text, 'candidates': cs, 'unique_counts': {'ОСОБА': 1} if cs else {}})
        self.source = self.directory / 'candidates.jsonl'
        self.source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in self.rows), encoding='utf8')
        (self.directory / 'summary.json').write_text(json.dumps({'selected': 43}), encoding='utf8')
        self.index = build_index(self.directory)
        self.store = ReviewStore(self.directory, self.index)

    def test_pages_and_original_service_metadata(self):
        result = self.store.query({})
        self.assertEqual((result['total'], len(result['items'])), (42, 40))
        self.assertFalse(result['items'][0]['excluded'])
        self.assertNotIn('text', result['items'][0]['doc'])
        second = self.store.query({'page': 1})
        self.assertEqual(len(second['items']), 2)
        service = self.store.query({'kind': 'service'})['items'][0]
        self.assertEqual(service['role'], 'суддя')
        self.assertEqual(service['exclusion_reason'], 'Роль судді')
        with closing(sqlite3.connect(self.index)) as db:
            self.assertEqual(self.store.document(service['doc']['_n'], db), self.rows[0])

    def test_search_reviews_and_documents_without_findings(self):
        self.assertEqual(self.store.query({'q': 'ІВАНЕНКО'})['total'], 42)
        self.assertEqual(self.store.query({'q': 'ПЕТРЕНКО', 'kind': 'service'})['total'], 42)
        reviews = {'0-p': {'status': 'service', 'note': 'manual'}}
        self.assertEqual(self.store.query({'status': 'service', 'reviews': reviews})['total'], 1)
        pending = self.store.query({'status': 'pending', 'reviews': reviews})
        self.assertEqual((pending['total'], pending['done']), (41, 1))
        docs = self.store.query({'view': 'documents', 'page': 1})
        self.assertEqual(docs['total'], 43)
        self.assertEqual(docs['items'][-1]['id'], '42')
        self.assertEqual(self.store.query({'view': 'documents', 'kind': 'service'})['total'], 42)
        self.assertEqual(self.store.query({'view': 'documents', 'q': '42'})['total'], 1)

    def test_index_reused_and_rebuilt_after_source_changes(self):
        before = self.index.stat().st_mtime_ns
        self.assertEqual(build_index(self.directory).stat().st_mtime_ns, before)
        with self.source.open('a', encoding='utf8') as stream:
            stream.write(json.dumps({'id': 'new', 'text': '', 'candidates': []}) + '\n')
        build_index(self.directory)
        self.assertEqual(self.store.query({'view': 'documents'})['total'], 44)

    def test_generators_keep_saved_data_and_create_small_local_html(self):
        original = self.source.read_bytes()
        self.assertEqual(generate_report(self.directory), 'local')
        html = (self.directory / 'review.html').read_text(encoding='utf8')
        self.assertIn('"server": true', html)
        self.assertNotIn('😀 Суддя Петренко', html)
        with patch('additional_tasks.audit.generate_review.EMBED_LIMIT', 1):
            self.assertEqual(generate_report(self.directory, 'auto'), 'local')
        html = (self.directory / 'review.html').read_text(encoding='utf8')
        self.assertIn('"server": true', html)
        self.assertNotIn('😀 Суддя Петренко', html)
        self.assertEqual(self.source.read_bytes(), original)
        generate_report(self.directory, 'auto')
        html = (self.directory / 'review.html').read_text(encoding='utf8')
        self.assertIn('Роль судді', html)
        self.assertIn('\\u003c/script>', html)


if __name__ == '__main__':
    unittest.main()
