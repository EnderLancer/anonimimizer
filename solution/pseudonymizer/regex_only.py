"""Режим без нейромережі та без CRF: правила (regex) + словник імен.

ОСОБА знаходиться за структурою запису (ПІБ із по батькові, «Прізвище І.Б.», «І.Б. Прізвище», пара ім'я+прізвище)
та за словником імен, побудованим з відкритих корпусів NER-UK/Ukr-Synth (resources/names_vocab.txt) і списком
«частих слів, що не є іменами» (resources/common_caps.txt). НОМЕР / ІНФОРМАЦІЯ / АДРЕСА — rules.py.
"""
import re
from pathlib import Path

from .persons import stem
from .pipeline import build_annotations

RES = Path(__file__).resolve().parent / "resources"
_read = lambda n: set((RES / n).read_text(encoding="utf8").split())
NAMES = _read("names_vocab.txt")
NAME_STEMS = {stem(w) for w in NAMES}
COMMON = _read("common_caps.txt")

CAP = r"[А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+"
WORD = re.compile(rf"{CAP}|[А-ЯІЇЄҐ]\.(?:\s?[А-ЯІЇЄҐ]\.)?")
PATR = re.compile(r"^[А-ЯІЇЄҐ][а-яіїєґ'’ʼ]+(?:вич|вна)(?:а|у|ем|і|ів|ею|ою|чу|чем|вні|вну|вни|вною)?$")
PATR2 = re.compile(r"^[А-ЯІЇЄҐ][а-яіїєґ'’ʼ]+(?:ович|евич|йович|івна|ївна|овна|евна|йовна)(?:а|у|ем|і|ів|ею|ою|чу|чем|вні|вну|вни|вною)?$")
SURNAME_SUF = re.compile(
    r"^[А-ЯІЇЄҐ][а-яіїєґ'’ʼ]{2,}(?:енко|ченко|ук|юк|чук|ський|цький|зький|ська|цька|зька|ського|цького|ської|цької|ському|цькому|ським|цьким|"
    r"ов|ова|ову|ового|овим|ев|єв|єва|ева|ин|іна|іну|іним|ін|ко|ка|ець|ця|цю|ич|иця|як|ак|ик|ій|ай|ець|юра|ура)$")
INITIALS = re.compile(r"^[А-ЯІЇЄҐ]\.(?:\s?[А-ЯІЇЄҐ]\.)?$")
ROLE = re.compile(
    r"(?i)(?:позивач\w*|відповідач\w*|обвинувачен\w*|підозрюван\w*|потерпіл\w*|свідок|свідк\w*|заявник\w*|заявниц\w*|"
    r"суддя|судді|адвокат\w*|прокурор\w*|секретар\w*|представник\w*|гр\.|громадян\w+|боржник\w*|стягувач\w*|"
    r"нотаріус\w*|засуджен\w*|спадкодав\w*|спадкоємц\w*|захисник\w*|перекладач\w*|експерт\w*|особ\w+|"
    r"чоловік\w*|дружин\w*|син\w*|дочк\w*|матір\w*|мати|батьк\w*|онук\w*|племінник\w*)\s*[:,—-]?\s*$")


def _tag(tok, prev_ctx):
    """Тег слова: I — ініціали, P — по батькові, N — слово зі словника імен, S — суфікс прізвища, C — інше, X — часте не-ім'я."""
    if INITIALS.match(tok):
        return "I"
    lw = tok.lower()
    if PATR.match(tok) or PATR2.match(tok):
        return "P"
    if lw in COMMON:
        return "X"
    if lw in NAMES or (len(lw) >= 4 and stem(tok) in NAME_STEMS):
        return "N"
    if SURNAME_SUF.match(tok):
        return "S"
    return "C"


def _runs(text):
    """Послідовності слів з великої літери / ініціалів, розділених лише пробілами."""
    items = [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(text)]
    runs, cur = [], []
    for it in items:
        if cur and re.fullmatch(r"[ \t\xa0]+", text[cur[-1][1]:it[0]] or "x"):
            cur.append(it)
        else:
            if cur:
                runs.append(cur)
            cur = [it]
    if cur:
        runs.append(cur)
    return runs


def _sentence_start(text, pos):
    pre = text[max(0, pos - 3):pos].rstrip(" \t\xa0")
    return pre == "" or pre[-1:] in ".!?\n:;»\"" and not re.search(r"\b[А-ЯІЇЄҐ]$", pre)


def person_spans(text):
    out = []
    for run in _runs(text):
        tags = [_tag(t, None) for _, _, t in run]
        n = len(run)
        used = [False] * n

        def take(i, j):
            for k in range(i, j + 1):
                used[k] = True
            out.append((run[i][0], run[j][1]))

        # 1) по батькові — сильний сигнал: беремо до 2 слів перед і 1 після
        for i, tg in enumerate(tags):
            if tg == "P" and not used[i]:
                a = i
                while a > 0 and i - a < 2 and tags[a - 1] in "NSCI" and not used[a - 1]:
                    a -= 1
                b = i
                if b + 1 < n and tags[b + 1] in "NSI" and not used[b + 1]:
                    b += 1
                take(a, b)
        # 2) ініціали + слово (прізвище): «Іваненко П.П.», «П.П. Іваненко»
        for i, tg in enumerate(tags):
            if used[i] or tg != "I":
                continue
            if i > 0 and tags[i - 1] in "NSC" and not used[i - 1] and tags[i - 1] != "X":
                a = i - 1
                b = i
                take(a, b)
            elif i + 1 < n and tags[i + 1] in "NSC" and not used[i + 1]:
                take(i, i + 1)
            elif i + 1 < n and tags[i + 1] == "I":
                continue
        # 3) пара слів зі словника / суфіксів: «Дарина Соя», «Петренко Іван»
        i = 0
        while i < n - 1:
            if not used[i] and not used[i + 1] and tags[i] in "NS" and tags[i + 1] in "NS" and "X" not in (tags[i], tags[i + 1]):
                j = i + 1
                if j + 1 < n and tags[j + 1] in "NS" and not used[j + 1]:
                    j += 1
                take(i, j)
                i = j + 1
            else:
                i += 1
        # 4) одиночне слово: за роллю перед ним, або слово зі словника не на початку речення
        for i, tg in enumerate(tags):
            if used[i] or tg in "XIP":
                continue
            s, e, tok = run[i]
            role = ROLE.search(text[max(0, s - 28):s]) is not None
            if role and tg in "NSC" and len(tok) >= 3:
                take(i, i)
            elif tg == "N" and len(tok) >= 4 and not _sentence_start(text, s):
                take(i, i)
            elif tg == "S" and not _sentence_start(text, s) and len(tok) >= 5 and role:
                take(i, i)
    out.sort()
    return out


def annotate(text):
    pers = person_spans(text)
    return build_annotations(text, pers)
