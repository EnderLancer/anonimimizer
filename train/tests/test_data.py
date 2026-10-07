from transformers import AutoTokenizer

from pseudo import LABELS
from pseudo.data import decode, encode

TOK = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")


def test_encode_decode_roundtrip():
    text = "Позов Коваля Івана Петровича до ТОВ «Альфа», РНОКПП 3127985494, задоволено. Суддя Іщук Т. П."
    anns = []
    for ent, t in (("Коваля Івана Петровича", "ОСОБА"), ("3127985494", "НОМЕР"), ("Іщук Т. П.", "ОСОБА")):
        s = text.index(ent)
        anns.append({"start": s, "end": s + len(ent), "type": t})
    windows = encode(TOK, text, anns, [], pad=False)
    assert len(windows) == 1
    labels = windows[0]["labels"]
    assert sum(1 for l in labels if l >= 0 and LABELS[l].startswith("B-")) == 3
    spans = decode(text, windows, [[max(l, 0) for l in labels]])
    assert [(s["start"], s["end"], s["type"]) for s in spans] == [(a["start"], a["end"], a["type"]) for a in anns]


def test_masks_become_ignore_index():
    text = "телефон 0501234567 і РНОКПП 3127985494"
    ps, ns = text.index("0501234567"), text.index("3127985494")
    windows = encode(TOK, text, [{"start": ns, "end": ns + 10, "type": "НОМЕР"}], [{"start": ps, "end": ps + 10, "reason": "phone"}], pad=False)
    labels = windows[0]["labels"]
    offs = windows[0]["offsets"]
    masked = [l for (s, e), l in zip(offs, labels) if s >= ps and e <= ps + 10]
    assert masked and all(l == -100 for l in masked)


def test_long_text_windows_overlap():
    text = ("Коваль Іван Петрович подав позов. " * 200).strip()
    windows = encode(TOK, text, [], [], pad=True)
    assert len(windows) > 1
    assert all(len(w["input_ids"]) == 512 for w in windows)


def test_windows_cover_whole_text():
    text = ("Позов Коваля Івана Петровича до Шевченко Ольги Іванівни задоволено. " * 120).strip()
    windows = encode(TOK, text, [], [], pad=True)
    last = max(e for w in windows for (s, e) in w["offsets"])
    assert last == len(text)
    assert len(windows) >= 3
    # consecutive windows overlap by the stride
    starts = [min(s for (s, e) in w["offsets"] if (s, e) != (0, 0)) for w in windows]
    ends = [max(e for (s, e) in w["offsets"]) for w in windows]
    assert all(starts[i + 1] < ends[i] for i in range(len(windows) - 1))


def test_long_window_encoding():
    text = ("Позов Коваля Івана Петровича до Шевченко Ольги Іванівни задоволено. " * 120).strip()
    windows = encode(TOK, text, [], [], window=4096, stride=512, pad=False)
    assert len(windows) == 1 and len(windows[0]["input_ids"]) <= 4096
    assert max(e for (s, e) in windows[0]["offsets"]) == len(text)
