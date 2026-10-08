"""Обробка сутностей NER-моделі після адаптера: окремий крок між ner_adapters (формат) і pseudonymize (злиття з правилами).

    run_ner ──► map_labels (ner_adapters: сирі мітки -> типи) ──► process (тут) ──► pseudonymize

PII-моделі ділять сутність на частини: GIVEN_NAME «Олена» + SURNAME «Соловей»; STREET_NAME + BUILDING_NUMBER + CITY.
Для челенджу це одна сутність, а окремі частини не дотягують до IoU 0,5 з розміткою, тож частини одного типу
складаються в одну сутність. Тип із зірочкою (CITY -> «АДРЕСА*») — слабка частина: лишається, лише якщо приєдналася
до звичайної (місто в адресі з вулицею — так, «суду міста Харкова» — ні).
"""
import re

from ner_adapters import EntitySpan

SPACES = re.compile(r"[ \t\xa0]+")
# що може стояти між частинами однієї адреси: «вул. Шевченка[, буд. ]5[, кв. ]14[, м. ]Київ[, ]Київська область»
ADDRESS_GAP = re.compile(r"[ \t\xa0,]*(?:(?:буд|б|кв|корп|оф|м|с|смт|обл|р-н|вул|просп|пров|пл)\.|будин\w*|квартир\w*|"
                         r"корпус\w*|офіс\w*|місто|міста|місті|село|села|селі|селище\w*)?[ \t\xa0,.№]*")
JOIN_GAP = {"АДРЕСА": ADDRESS_GAP}  # для інших типів — лише пробіли
MAX_GAP = 20


def assemble(spans: list[EntitySpan], text: str) -> list[EntitySpan]:
    """Об'єднує частини одного типу, що перетинаються або розділені допустимим проміжком (JOIN_GAP),
    і не перескакує через сутність іншого типу. Слабкі частини (тип із *) без звичайної відкидаються."""
    items = sorted(([sp.start, sp.end, sp.label.rstrip("*"), sp.score, not sp.label.endswith("*")] for sp in spans),
                   key=lambda x: (x[0], x[1]))
    merged: list[list] = []
    for it in items:
        last = next((m for m in reversed(merged) if m[2] == it[2]), None)
        joined = False
        if last and not any(m[2] != it[2] and m[0] >= last[1] and m[1] <= it[0] for m in merged):
            gap = text[last[1]:it[0]]
            joined = it[0] <= last[1] or len(gap) <= MAX_GAP and bool(JOIN_GAP.get(it[2], SPACES).fullmatch(gap))
        if joined:
            last[1], last[3], last[4] = max(last[1], it[1]), max(last[3], it[3]), last[4] or it[4]
        else:
            merged.append(it)
    return [EntitySpan(s, e, text[s:e], t, sc) for s, e, t, sc, strong in sorted(merged, key=lambda x: (x[0], x[1])) if strong]


def process(spans: list[EntitySpan], text: str) -> list[EntitySpan]:
    return assemble(spans, text)
