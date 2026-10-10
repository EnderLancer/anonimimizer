import pytest
from transformers import AutoTokenizer

from pseudo import LABELS
from pseudo.data import check_window, decode, encode

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


@pytest.mark.parametrize("window, stride", [(512, 510), (512, 511), (512, 512), (512, -1), (2, 0), (1, 0)])
def test_window_and_stride_that_cannot_advance_are_rejected(window, stride):
    # longer than one window, so that without the check the loop would not end
    text = ("Позов Коваля Івана Петровича до Шевченко Ольги Іванівни задоволено. " * 120).strip()
    with pytest.raises(ValueError, match=f"window {window} / stride {stride} "):
        encode(TOK, text, [], [], window=window, stride=stride)
    with pytest.raises(ValueError):
        check_window(window, stride)


def test_smallest_step_still_ends_and_covers_the_text():
    text = ("Позов Коваля Івана Петровича до Шевченко Ольги Іванівни задоволено. " * 4).strip()
    n_tokens = len(TOK(text, add_special_tokens=False)["input_ids"])
    assert 40 <= n_tokens <= 80
    windows = encode(TOK, text, [], [], window=16, stride=13, pad=False)  # 14 text tokens per window, a step of one
    assert len(windows) == n_tokens - 14 + 1
    assert all(len(w["input_ids"]) <= 16 for w in windows)
    assert min(s for w in windows for (s, e) in w["offsets"] if (s, e) != (0, 0)) == 0
    assert max(e for w in windows for (s, e) in w["offsets"]) == len(text)


def test_usual_window_settings_pass_the_check():
    check_window(512, 128)
    check_window(4096, 512)


def _labels_per_char(text: str, spans: list[tuple[str, str]]) -> tuple[list[dict], list[list[int]]]:
    """Windows of `text` and the token labels a perfect model would give for (entity, type) spans, each starting with B-."""
    anns = []
    pos = 0
    for ent, t in spans:
        s = text.index(ent, pos)
        anns.append({"start": s, "end": s + len(ent), "type": t})
        pos = s + len(ent)
    windows = encode(TOK, text, anns, [], pad=False)
    return windows, [[max(l, 0) for l in w["labels"]] for w in windows]


def test_decode_keeps_adjacent_entities_apart():
    text = "Судді:\nКоваль І. П.\nШевченко О. В.\nпозов Мельника Петра Івановича задоволено"
    windows, preds = _labels_per_char(text, [("Коваль І. П.", "ОСОБА"), ("Шевченко О. В.", "ОСОБА"), ("Мельника Петра Івановича", "ОСОБА")])
    assert [text[s["start"]:s["end"]] for s in decode(text, windows, preds)] == ["Коваль І. П.", "Шевченко О. В.", "Мельника Петра Івановича"]


def test_training_settings_reach_the_encoder():
    from pseudo.train import build_features

    text = ("Позивач Коваль Іван Петрович, телефон 0501234567, просить стягнути кошти. " * 40).strip()
    ps = text.index("0501234567")
    rows = [{"text": text, "annotations": [], "masks": [{"start": ps, "end": ps + 10, "reason": "phone"}]}]
    masked = build_features(TOK, rows)
    plain = build_features(TOK, rows, use_masks=False)
    assert len(masked[0]["input_ids"]) == 512 and -100 in masked[0]["labels"][1:20]
    assert -100 not in plain[0]["labels"][1:20]
    short = build_features(TOK, rows, window=128, stride=32)
    assert all(len(w["input_ids"]) == 128 for w in short) and len(short) > len(masked)


def test_piece_offsets_from_bytes_and_from_characters():
    from pseudo.tokenization import piece_offsets

    text = "Суд  ОСОБА_1 ½ м."
    # (begin, end, surface) in characters: «▁Суд», «▁ОСОБА» with two spaces, «_1», a lone «▁», a byte piece, «2» for ½, «▁м.»
    chars = [(0, 3, "Суд"), (3, 10, "  ОСОБА"), (10, 12, "_1"), (12, 13, " "), (13, 13, ""), (13, 14, "½"), (14, 17, " м.")]
    want = [(0, 3), (5, 10), (10, 12), (12, 13), (13, 14), (13, 14), (15, 17)]
    assert piece_offsets(text, chars) == want
    at = lambda i: len(text[:i].encode("utf-8"))
    assert piece_offsets(text, [(at(b), at(e), s) for b, e, s in chars]) == want
