import ast
import errno
import importlib.util
import os
import sys
import threading
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fetch_edrsr", Path(__file__).resolve().parents[1] / "scripts" / "fetch_edrsr.py")
fetch_edrsr = importlib.util.module_from_spec(spec)
sys.modules["fetch_edrsr"] = fetch_edrsr
spec.loader.exec_module(fetch_edrsr)

RTF = (r"{\rtf1\ansi\ansicpg1251\deff0{\fonttbl{\f0 Times New Roman;}}\f0\fs24 "
       r"\'d3\'d5\'c2\'c0\'cb\'c0\par 01 \'f1\'b3\'f7\'ed\'ff 2024 \'f0\'ee\'ea\'f3 \'ec. \'ca\'e8\'bf\'e2\par "
       r"\'ef\'ee\'e7\'e8\'e2\'e0\'f7 \'ce\'d1\'ce\'c1\'c0_1 , \'b2\'cd\'d4\'ce\'d0\'cc\'c0\'d6\'b2\'df_1\par}")


@pytest.fixture(autouse=True)
def _no_error_reported_yet(monkeypatch):
    """Each test starts as a new run: the first unexpected error of a class is printed again."""
    monkeypatch.setattr(fetch_edrsr, "_reported", set())


def test_rtf_to_text_decodes_cp1251_and_keeps_placeholders(tmp_path):
    p = tmp_path / "x.rtf"
    p.write_bytes(RTF.encode("latin-1"))
    text = fetch_edrsr.rtf_text(p)
    assert text.startswith("УХВАЛА")
    assert "01 січня 2024 року м. Київ" in text
    assert "ОСОБА_1" in text and "ІНФОРМАЦІЯ_1" in text


class _Response:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code, self.content = status_code, content


class _Session:
    """Answers `get` with the queued responses; an exception in the queue is raised."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def get(self, url, timeout):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


OK = _Response(200, RTF.encode("latin-1"))
ROW = {"id": "7", "url": "http://x/7.rtf"}


def _fetch(monkeypatch, tmp_path, *answers):
    """Status, number of requests, the path of the decision and the waits (seconds given to time.sleep, in order)."""
    waits: list[float] = []
    monkeypatch.setattr(fetch_edrsr, "RTF_DIR", tmp_path)
    monkeypatch.setattr(fetch_edrsr.time, "sleep", waits.append)
    session = _Session(*answers)
    row, status = fetch_edrsr.fetch(session, ROW)
    assert row is ROW
    return status, session.calls, (tmp_path / "7.rtf"), waits


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_fetch_reports_throttling_when_the_attempts_run_out(monkeypatch, tmp_path):
    status, calls, path, waits = _fetch(monkeypatch, tmp_path, _Response(429), _Response(503), _Response(429))
    assert status == "http 429, gave up after 3 attempts" and calls == 3 and not path.exists()
    assert waits == [5, 10]  # nothing after the last attempt


def test_fetch_reports_the_network_error_when_the_attempts_run_out(monkeypatch, tmp_path):
    requests = fetch_edrsr.requests
    status, calls, path, waits = _fetch(monkeypatch, tmp_path, *[requests.Timeout()] * 3)
    assert status == "Timeout, gave up after 3 attempts" and calls == 3 and not path.exists()
    assert waits == [2, 4]
    broken = [requests.ConnectionError(), requests.exceptions.ChunkedEncodingError(), requests.exceptions.ContentDecodingError()]
    assert _fetch(monkeypatch, tmp_path, *broken)[:2] == ("ContentDecodingError, gave up after 3 attempts", 3)


def test_fetch_status_of_a_mixed_failure_names_the_last_error(monkeypatch, tmp_path):
    requests = fetch_edrsr.requests
    status, calls, _, waits = _fetch(monkeypatch, tmp_path, requests.ConnectionError(), requests.ConnectionError(), _Response(429))
    assert (status, calls, waits) == ("http 429, gave up after 3 attempts", 3, [2, 4])
    status, calls, _, waits = _fetch(monkeypatch, tmp_path, _Response(429), _Response(429), requests.Timeout())
    assert (status, calls, waits) == ("Timeout, gave up after 3 attempts", 3, [5, 10])


def test_fetch_retries_after_throttling_and_network_errors(monkeypatch, tmp_path):
    status, calls, path, waits = _fetch(monkeypatch, tmp_path, fetch_edrsr.requests.ConnectionError(), _Response(503), OK)
    assert status == "ok" and calls == 3 and path.read_bytes() == OK.content
    assert waits == [2, 10, fetch_edrsr.PAUSE]  # 2 s x attempt 1, 5 s x attempt 2, the polite pause after the download
    assert _files(tmp_path) == ["7.rtf"]  # no temporary file left
    assert _fetch(monkeypatch, tmp_path, _Response(404))[:2] == ("cached", 0)  # the file is there now
    assert _fetch(monkeypatch, tmp_path, _Response(404))[3] == []


def test_fetch_returns_other_answers_at_once(monkeypatch, tmp_path):
    for answer, status in [(_Response(404), "http 404"), (_Response(500), "http 500"), (_Response(502), "http 502"),
                           (_Response(200, b"<html>"), "not rtf"), (_Response(200), "not rtf")]:
        assert _fetch(monkeypatch, tmp_path, answer) == (status, 1, tmp_path / "7.rtf", [])
    assert _files(tmp_path) == []


def test_fetch_does_not_retry_an_invalid_url(monkeypatch, tmp_path):
    exceptions = fetch_edrsr.requests.exceptions
    for error in (exceptions.MissingSchema(), exceptions.InvalidSchema(), exceptions.InvalidURL()):
        assert _fetch(monkeypatch, tmp_path, error) == (type(error).__name__, 1, tmp_path / "7.rtf", [])
    # the real session refuses these before it opens any connection
    waits: list[float] = []
    monkeypatch.setattr(fetch_edrsr.time, "sleep", waits.append)
    with fetch_edrsr.requests.Session() as session:
        assert fetch_edrsr.fetch(session, {"id": "8", "url": ""})[1] == "MissingSchema"
        assert fetch_edrsr.fetch(session, {"id": "8", "url": "ftp://x/8.rtf"})[1] == "InvalidSchema"
    assert waits == [] and _files(tmp_path) == []


def test_fetch_turns_an_unexpected_error_into_a_status(monkeypatch, tmp_path, capsys):
    assert _fetch(monkeypatch, tmp_path, ValueError("boom")) == ("error: ValueError", 1, tmp_path / "7.rtf", [])
    monkeypatch.setattr(fetch_edrsr, "RTF_DIR", tmp_path)
    assert fetch_edrsr.fetch(_Session(), {"url": "http://x/7.rtf"})[1] == "error: KeyError"  # a row without an id
    printed = capsys.readouterr()
    assert printed.out == ""
    # the counters keep only the class: the message and the place are on stderr
    assert "unexpected ValueError at decision 7" in printed.err and "ValueError: boom" in printed.err
    assert "Traceback (most recent call last)" in printed.err and "in download" in printed.err
    assert "unexpected KeyError at decision None" in printed.err


def test_an_unexpected_error_is_printed_once_per_class(monkeypatch, tmp_path, capsys):
    for _ in range(3):
        assert _fetch(monkeypatch, tmp_path, ValueError("boom"))[0] == "error: ValueError"
    assert _fetch(monkeypatch, tmp_path, TypeError("other"))[0] == "error: TypeError"
    err = capsys.readouterr().err
    assert err.count("ValueError: boom") == 1 and err.count("TypeError: other") == 1
    assert err.count("Traceback (most recent call last)") == 2
    # the expected failures print nothing
    _fetch(monkeypatch, tmp_path, _Response(404))
    _fetch(monkeypatch, tmp_path, fetch_edrsr.requests.exceptions.InvalidURL())
    assert capsys.readouterr().err == ""


def test_fetch_reports_a_failed_write_and_downloads_again(monkeypatch, tmp_path):
    def denied(self, data):
        raise PermissionError(13, "denied")

    with monkeypatch.context() as m:
        m.setattr(Path, "write_bytes", denied)
        assert _fetch(m, tmp_path, OK) == ("write error: EACCES", 1, tmp_path / "7.rtf", [])
    assert _files(tmp_path) == []
    status, calls, path, _ = _fetch(monkeypatch, tmp_path, OK)
    assert (status, calls) == ("ok", 1) and path.read_bytes() == OK.content


def test_fetch_leaves_no_file_when_the_write_stops_half_way(monkeypatch, tmp_path):
    def disk_full(self, data):
        with open(self, "wb") as f:
            f.write(data[:100])
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m:
        m.setattr(Path, "write_bytes", disk_full)
        assert _fetch(m, tmp_path, OK)[:2] == ("write error: ENOSPC", 1)
    assert _files(tmp_path) == []  # neither a cut-off 7.rtf nor the temporary file
    status, calls, path, _ = _fetch(monkeypatch, tmp_path, OK)  # not «cached»: it is downloaded again
    assert (status, calls) == ("ok", 1) and path.read_bytes() == OK.content


def test_fetch_keeps_the_old_state_when_the_rename_fails(monkeypatch, tmp_path):
    def refuse(src, dst):
        raise OSError(5, "Input/output error")

    with monkeypatch.context() as m:
        m.setattr(fetch_edrsr.os, "replace", refuse)
        assert _fetch(m, tmp_path, OK)[:2] == ("write error: EIO", 1)
    assert _files(tmp_path) == []


def test_the_status_of_a_failed_write_names_the_cause(monkeypatch, tmp_path, capsys):
    for error, name in [(OSError(errno.ENOSPC, "full"), "ENOSPC"), (OSError(errno.EROFS, "read-only"), "EROFS"),
                        (OSError(errno.EIO, "i/o"), "EIO"), (OSError(errno.EMFILE, "files"), "EMFILE"),
                        (OSError(errno.EDQUOT, "quota"), "EDQUOT"), (FileNotFoundError(errno.ENOENT, "gone"), "ENOENT"),
                        (OSError("no errno"), "OSError"), (OSError(10**6, "unknown errno"), "OSError")]:
        def fail(self, data, error=error):
            raise error

        with monkeypatch.context() as m:
            m.setattr(Path, "write_bytes", fail)
            assert _fetch(m, tmp_path, OK)[:2] == (f"write error: {name}", 1)
    assert _files(tmp_path) == [] and capsys.readouterr().err == ""  # an expected failure: no traceback


def test_an_interrupted_write_leaves_no_file(monkeypatch, tmp_path):
    def interrupted(self, data):
        with open(self, "wb") as f:
            f.write(data[:100])
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(Path, "write_bytes", interrupted)
        with pytest.raises(KeyboardInterrupt):  # not an Exception: it is not turned into a status
            _fetch(m, tmp_path, OK)
    assert _files(tmp_path) == []


def test_each_writer_has_its_own_temporary_file(monkeypatch, tmp_path):
    """Two writers of one decision: B has written its temporary file, A opens its own, writes a part of it and
    fails only after B has renamed. With one shared name A would truncate the file that B then makes final."""
    real_write, real_replace = Path.write_bytes, os.replace
    b_wrote, a_open, b_done = threading.Event(), threading.Event(), threading.Event()
    names: dict[str, str] = {}

    def write_bytes(self, data):
        who = threading.current_thread().name
        names[who] = self.name
        if who == "A":
            assert b_wrote.wait(5)
            with open(self, "wb") as f:
                f.write(data[:100])
                f.flush()
                a_open.set()
                assert b_done.wait(5)
                raise OSError(errno.ENOSPC, "No space left on device")
        real_write(self, data)
        b_wrote.set()
        assert a_open.wait(5)

    def replace(src, dst):
        real_replace(src, dst)
        b_done.set()

    monkeypatch.setattr(fetch_edrsr, "RTF_DIR", tmp_path)
    monkeypatch.setattr(fetch_edrsr.time, "sleep", lambda s: None)
    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    monkeypatch.setattr(fetch_edrsr.os, "replace", replace)
    statuses: dict[str, str] = {}

    def run():
        statuses[threading.current_thread().name] = fetch_edrsr.fetch(_Session(OK), ROW)[1]

    threads = [threading.Thread(target=run, name=name) for name in "AB"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    monkeypatch.undo()
    assert statuses == {"A": "write error: ENOSPC", "B": "ok"}
    assert names["A"] != names["B"] and all(n.startswith("7.rtf.") and n.endswith(".part") for n in names.values())
    assert _files(tmp_path) == ["7.rtf"] and (tmp_path / "7.rtf").read_bytes() == OK.content  # whole, not A's 100 bytes


def test_two_writers_of_one_decision_both_succeed(monkeypatch, tmp_path):
    real_write = Path.write_bytes
    both_wrote = threading.Barrier(2)

    def write_bytes(self, data):  # both temporary files are written before either is renamed
        real_write(self, data)
        both_wrote.wait(5)

    monkeypatch.setattr(fetch_edrsr, "RTF_DIR", tmp_path)
    monkeypatch.setattr(fetch_edrsr.time, "sleep", lambda s: None)
    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    statuses: list[str] = []
    threads = [threading.Thread(target=lambda: statuses.append(fetch_edrsr.fetch(_Session(OK), ROW)[1])) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    monkeypatch.undo()
    assert statuses == ["ok", "ok"]  # not a write error for a file that is complete
    assert _files(tmp_path) == ["7.rtf"] and (tmp_path / "7.rtf").read_bytes() == OK.content


def test_select_takes_a_decision_listed_twice_once(tmp_path):
    kas, kgs = fetch_edrsr.KAS, "9911"
    (tmp_path / "cause_categories.csv").write_text("category_code\tname\n1\tСправи щодо проходження служби\n2\tІнше\n", encoding="utf-8")
    (tmp_path / "judgment_forms.csv").write_text("judgment_code\tname\n5\tПостанова\n", encoding="utf-8")
    lines = ["doc_id\tcourt_code\tjudgment_code\tcategory_code\tstatus\tdoc_url"]
    lines += [f"{doc_id}\t{court}\t5\t{cat}\t{status}\thttp://x/{doc_id}" for doc_id, court, cat, status in [
        ("10", kas, "1", "1"), ("10", kas, "1", "1"),  # the same row twice
        ("11", kas, "1", "1"), ("12", kas, "2", "1"),  # 12: a category that is not taken
        ("20", kgs, "2", "1"), ("20", kgs, "2", "1"), ("21", kgs, "2", "1"),
        ("10", kgs, "2", "1"),                         # an id already taken for another chamber
        ("30", kas, "1", "0"), ("30", kas, "1", "1"),  # the withdrawn copy does not hide the valid one
        ("31", kas, "1", "1"), ("40", "0101", "1", "1"), ("41", kas, "1", "1")]]
    (tmp_path / "documents.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = fetch_edrsr.select(tmp_path, sample=10, seed=0, known={"31"})
    assert sorted(r["id"] for r in rows) == ["10", "11", "20", "21", "30", "41"]
    assert rows[0] == {"id": "10", "court_code": kas, "url": "http://x/10", "category": "Справи щодо проходження служби",
                       "judgment": "Постанова"}


def test_an_empty_file_is_not_cached(monkeypatch, tmp_path):
    path = tmp_path / "7.rtf"
    assert not fetch_edrsr.is_cached(path)
    path.touch()
    assert not fetch_edrsr.is_cached(path)
    status, calls, _, _ = _fetch(monkeypatch, tmp_path, OK)
    assert (status, calls) == ("ok", 1) and path.read_bytes() == OK.content and fetch_edrsr.is_cached(path)


def _decision(doc_id: str) -> bytes:
    """An RTF decision long enough to be kept, with its own words (so that it is no near-copy of another one)."""
    words = " ".join(f"w{doc_id}x{i}" for i in range(300))
    placeholder = "".join(f"\\'{b:02x}" for b in "ІНФОРМАЦІЯ".encode("cp1251"))
    return (r"{\rtf1\ansi\ansicpg1251\deff0 " + words + " " + placeholder + r"_1\par}").encode("latin-1")


class _UrlSession:
    """Stands for requests.Session in main(): answers by URL and counts the requests."""

    def __init__(self, answers: dict[str, _Response]):
        self.answers, self.headers, self.urls = answers, {}, []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.answers[url]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _counts(printed: str) -> dict[str, int]:
    """The status counters of the line «download done in ...» that main() prints."""
    return ast.literal_eval(printed.split("download done in ")[1].split(": ", 1)[1].splitlines()[0])


def test_main_finishes_and_writes_the_parquet_when_a_worker_raises(monkeypatch, tmp_path, capsys):
    rtf_dir, out = tmp_path / "rtf", tmp_path / "out" / "vs.parquet"
    rows = [{"id": str(i), "court_code": fetch_edrsr.KAS, "url": f"http://x/{i}", "category": "c", "judgment": "j"} for i in range(6)]
    rtf_dir.mkdir()
    (rtf_dir / "0.rtf").write_bytes(_decision("0"))  # cached
    (rtf_dir / "1.rtf").touch()                      # an empty file: not cached, downloaded again
    (rtf_dir / "4.rtf").touch()                      # an empty file whose download fails: not a decision to build from
    answers = {r["url"]: _Response(200, _decision(r["id"])) for r in rows}
    answers["http://x/4"] = _Response(404)
    session = _UrlSession(answers)
    real_fetch, real_write = fetch_edrsr.fetch, Path.write_bytes

    def fetch(session, row):
        if row["id"] == "3":
            raise RuntimeError("worker died")
        return real_fetch(session, row)

    def write_bytes(self, data):
        if self.name.startswith("5.rtf"):
            raise PermissionError(13, "denied")
        return real_write(self, data)

    monkeypatch.setattr(fetch_edrsr, "RTF_DIR", rtf_dir)
    monkeypatch.setattr(fetch_edrsr, "VS_PARQUET", tmp_path / "no-5k-file.parquet")  # nothing is read from data/
    monkeypatch.setattr(fetch_edrsr, "select", lambda dump, sample, seed, known: list(rows))
    monkeypatch.setattr(fetch_edrsr, "fetch", fetch)
    monkeypatch.setattr(fetch_edrsr.requests, "Session", lambda: session)
    monkeypatch.setattr(fetch_edrsr.time, "sleep", lambda s: None)
    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    monkeypatch.setattr(sys, "argv", ["fetch_edrsr.py", "--dump", str(tmp_path), "--out", str(out), "--workers", "2"])
    code = fetch_edrsr.main()
    monkeypatch.undo()

    printed, err = capsys.readouterr()
    assert code == 1  # the parquet is built, but an unexpected error does not pass for a clean run
    assert "unexpected RuntimeError at decision 3" in err and "RuntimeError: worker died" in err and "Traceback" in err
    assert "1 of 6 downloads failed with an unexpected error" in err
    assert "selected 6 decisions" in printed and "; 1 already cached;" in printed
    assert _counts(printed) == {"cached": 1, "ok": 2, "error: RuntimeError": 1, "http 404": 1, "write error: EACCES": 1}
    assert sorted(session.urls) == ["http://x/1", "http://x/2", "http://x/4", "http://x/5"]
    assert _files(rtf_dir) == ["0.rtf", "1.rtf", "2.rtf", "4.rtf"] and (rtf_dir / "4.rtf").stat().st_size == 0
    table = fetch_edrsr.pq.read_table(out)
    assert table.column("id").to_pylist() == ["0", "1", "2"]
    assert all("ІНФОРМАЦІЯ_1" in t for t in table.column("text").to_pylist())
    assert "kept 3 decisions" in printed and "'missing': 3" in printed


def _run_main(monkeypatch, tmp_path, rows, session) -> tuple[int, Path]:
    out = tmp_path / "out" / "vs.parquet"
    monkeypatch.setattr(fetch_edrsr, "RTF_DIR", tmp_path / "rtf")
    monkeypatch.setattr(fetch_edrsr, "VS_PARQUET", tmp_path / "no-5k-file.parquet")
    monkeypatch.setattr(fetch_edrsr, "select", lambda dump, sample, seed, known: list(rows))
    monkeypatch.setattr(fetch_edrsr.requests, "Session", lambda: session)
    monkeypatch.setattr(fetch_edrsr.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["fetch_edrsr.py", "--dump", str(tmp_path), "--out", str(out), "--workers", "2"])
    return fetch_edrsr.main(), out


def _rows(n: int) -> list[dict]:
    return [{"id": str(i), "court_code": fetch_edrsr.KAS, "url": f"http://x/{i}", "category": "c", "judgment": "j"} for i in range(n)]


def test_main_shows_a_bug_that_hits_every_worker(monkeypatch, tmp_path, capsys):
    rows = _rows(20)
    session = _UrlSession({r["url"]: _Response(200, _decision(r["id"])) for r in rows})
    real_download = fetch_edrsr.download
    monkeypatch.setattr(fetch_edrsr, "download", lambda session, row: real_download(session, row) + None)
    code, out = _run_main(monkeypatch, tmp_path, rows, session)
    printed, err = capsys.readouterr()
    assert code == 1 and "{'error: TypeError': 20}" in printed
    assert err.count("Traceback (most recent call last)") == 1 and "TypeError: can only concatenate str" in err
    assert "20 of 20 downloads failed with an unexpected error" in err
    assert fetch_edrsr.pq.read_table(out).num_rows == 20  # the files were written before the bug: the build goes on


def test_main_returns_zero_when_the_only_failures_are_expected_ones(monkeypatch, tmp_path, capsys):
    rows = _rows(3)
    answers = {r["url"]: _Response(200, _decision(r["id"])) for r in rows}
    answers["http://x/1"] = _Response(404)
    code, out = _run_main(monkeypatch, tmp_path, rows, _UrlSession(answers))
    printed, err = capsys.readouterr()
    assert code == 0 and err == ""
    assert _counts(printed) == {"ok": 2, "http 404": 1}
    assert fetch_edrsr.pq.read_table(out).column("id").to_pylist() == ["0", "2"]


def test_keep_rule():
    long = "x" * fetch_edrsr.MIN_CHARS
    assert fetch_edrsr.keep_rule("1", "short ОСОБА_1") == "short"
    assert fetch_edrsr.keep_rule("1", long) == "no_placeholders"
    assert fetch_edrsr.keep_rule("1", long + " ІНФОРМАЦІЯ_1") is None
    persons_only = [fetch_edrsr.keep_rule(str(i), long + " ОСОБА_1") for i in range(400)]
    kept = persons_only.count(None)
    assert 0.15 * 400 < kept < 0.35 * 400 and set(persons_only) == {None, "persons_only"}


def test_near_copies_drop_a_templated_ruling_but_keep_a_different_one():
    base = " ".join(f"слово{i}" for i in range(400))
    copies = fetch_edrsr.NearCopies()
    assert not copies.is_copy(base)
    assert copies.is_copy(base.replace("слово10", "інше") + " ОСОБА_2")
    assert not copies.is_copy(" ".join(f"текст{i}" for i in range(400)))
