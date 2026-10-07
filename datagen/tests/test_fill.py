import json
import random
import re

from silver.fill.run import Filler
from silver.normalize import PLACEHOLDER, find_placeholders, normalize


def test_fill_roundtrip_offsets():
    raw = ("Позивач ОСОБА_1 , ІНФОРМАЦІЯ_1 , РНОКПП НОМЕР_1 , мешкає за адресою: АДРЕСА_1 , звернувся до суду з позовом "
           "до ОСОБА_2 . Представник ОСОБА_1 адвокат Іщук Т. П. просив стягнути з ОСОБА_2 на користь ОСОБА_1 кошти. "
           "Електронна адреса ОСОБА_2 : ІНФОРМАЦІЯ_2 . Народилась ІНФОРМАЦІЯ_3 .")
    text = normalize(raw)
    doc = {"id": "t", "text": text, "placeholders": find_placeholders(text)}
    out = Filler(seed=1).fill(doc)
    assert not PLACEHOLDER.search(out["text"])
    assert not re.search(r"\S ,", out["text"])
    for l in out["labels"]:
        frag = out["text"][l["start"]:l["end"]]
        assert frag == frag.strip() and frag
    persons = {l["pseudonym"] for l in out["labels"] if l["type"] == "ОСОБА"}
    assert persons == {"ОСОБА_1", "ОСОБА_2"}
    assert out["n_persons"] == 2
    types = {l["type"] for l in out["labels"]}
    assert types == {"ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ"}
    email = [out["text"][l["start"]:l["end"]] for l in out["labels"] if l["pseudonym"] == "ІНФОРМАЦІЯ_2"][0]
    assert "@" in email
    rnokpp = [out["text"][l["start"]:l["end"]] for l in out["labels"] if l["pseudonym"] == "НОМЕР_1"][0]
    assert re.fullmatch(r"\d{10}", rnokpp)


def test_fill_is_deterministic():
    text = normalize("Позов ОСОБА_1 до ОСОБА_2 , ІНФОРМАЦІЯ_1 .")
    doc = {"id": "t", "text": text, "placeholders": find_placeholders(text)}
    a = Filler(seed=7).fill(doc)
    b = Filler(seed=7).fill(doc)
    assert a == b
