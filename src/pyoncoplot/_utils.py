"""Small shared utility functions."""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def prettify(value: str, space_after_apostrophe: bool = True, autodetect_units: bool = True) -> str:
    """Make snake/camel case labels friendlier for legends and axes."""

    text = str(value).replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\b([a-z])", lambda match: match.group(1).upper(), text)

    if space_after_apostrophe:
        text = re.sub(r"([A-Za-z0-9]')([^ \t\n\r\f\v])", r"\1 \2", text)

    if autodetect_units:
        replacements = {
            "mm": "(mm)",
            "cm": "(cm)",
            "km": "(km)",
            "kg": "(kg)",
            "mg": "(mg)",
            "oz": "(oz)",
            "lb": "(lb)",
            "in": "(in)",
            "ft": "(ft)",
            "yd": "(yd)",
            "mi": "(mi)",
            "m": "(m)",
            "g": "(g)",
        }
        for unit, replacement in replacements.items():
            text = re.sub(rf"\s{unit}(\s|$)", f" {replacement} ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(values: Iterable[T]) -> List[T]:
    seen = set()
    out: List[T] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def reorder_by_priority(original: Sequence[T], priority_values: Sequence[T]) -> List[T]:
    priority_set = set(priority_values)
    front = [value for value in priority_values if value in original]
    tail = [value for value in original if value not in priority_set]
    return front + tail


def as_percent(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA%"
    rounded = round(value * 100, digits)
    if digits == 0:
        rounded = int(rounded)
    return f"{rounded}%"
