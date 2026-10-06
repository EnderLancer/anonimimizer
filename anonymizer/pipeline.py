"""Hidden values at positions -> Mapping values -> Link results -> Cleaned document.

RegEx і Model незалежно знаходять приховані значення в оригіналі. Тут їх об'єднуємо,
даємо псевдоніми і підставляємо в оригінал.
"""
from __future__ import annotations

from .annotations import Hidden

# за перетину фрагментів лишаємо той, чиє джерело стоїть раніше
DEFAULT_PRIORITY = ("regex", "model")


def merge(*sources: list[Hidden], priority: tuple[str, ...] = DEFAULT_PRIORITY) -> list[Hidden]:
    """Об'єднує анотації етапів. Якщо фрагменти перетинаються, лишаємо фрагмент пріоритетнішого джерела,
    а в його metadata.agreed_with дописуємо джерела, які знайшли те саме місце того самого типу."""
    rank = {s: i for i, s in enumerate(priority)}
    candidates = sorted((h for src in sources for h in src),
                        key=lambda h: (rank.get(h.source, len(rank)), h.start, -(h.end - h.start)))
    kept: list[Hidden] = []
    for h in candidates:
        clash = [k for k in kept if k.overlaps(h)]
        if not clash:
            kept.append(Hidden(h.start, h.end, h.type, h.text, dict(h.metadata)))
            continue
        for k in clash:
            if k.type == h.type and h.source != k.source:
                agreed = k.metadata.setdefault("agreed_with", [])
                if h.source not in agreed:
                    agreed.append(h.source)
    return sorted(kept, key=lambda h: h.start)


def link(original: str, labels: dict[Hidden, str]) -> str:
    """Cleaned document: оригінал з псевдонімами замість прихованих значень."""
    parts, pos = [], 0
    for h in sorted(labels, key=lambda h: h.start):
        parts += [original[pos:h.start], labels[h]]
        pos = h.end
    parts.append(original[pos:])
    return "".join(parts)
