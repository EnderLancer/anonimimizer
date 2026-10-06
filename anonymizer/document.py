"""Документ із замаскованими фрагментами, який пам'ятає позиції в оригіналі."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")


@dataclass(frozen=True)
class Hidden:
    """Прихований фрагмент original[start:end] певного типу."""
    start: int
    end: int
    type: str
    value: str
    source: str  # "regex" або "model"


class MaskedDocument:
    """original, у якому фрагменти hidden замінено на мітки типу «[ОСОБА]».

    Для кожного шматка тексту пам'ятаємо його позицію в оригіналі, тож фрагмент, знайдений
    у замаскованому тексті, можна перенести назад в оригінал.
    """

    def __init__(self, original: str, hidden: list[Hidden]):
        self.original = original
        self.hidden = sorted(hidden, key=lambda h: h.start)
        parts: list[str] = []
        # (позиція в text, позиція в original, чи це мітка)
        self._segments: list[tuple[int, int, bool]] = []
        pos = mpos = 0
        for h in self.hidden:
            if h.start < pos:
                continue
            if h.start > pos:
                self._segments.append((mpos, pos, False))
                parts.append(original[pos:h.start])
                mpos += h.start - pos
            mark = f"[{h.type}]"
            self._segments.append((mpos, h.start, True))
            parts.append(mark)
            mpos += len(mark)
            pos = h.end
        self._segments.append((mpos, pos, False))
        parts.append(original[pos:])
        self.text = "".join(parts)
        self._starts = [s[0] for s in self._segments]

    def to_original(self, start: int, end: int) -> tuple[int, int] | None:
        """Переносить text[start:end] в оригінал. None, якщо фрагмент зачіпає мітку."""
        i = bisect_right(self._starts, start) - 1
        j = bisect_right(self._starts, max(start, end - 1)) - 1
        if i != j or self._segments[i][2]:
            return None
        mstart, ostart, _ = self._segments[i]
        return ostart + start - mstart, ostart + end - mstart
