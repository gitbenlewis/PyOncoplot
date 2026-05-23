"""Helpers for config-driven parameter dictionaries."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Optional


def merge_params(
    params: Optional[Mapping[str, Any]] = None,
    *,
    allowed_keys: Optional[Collection[str]] = None,
    context: str = "parameters",
    **kwargs: Any,
) -> dict[str, Any]:
    """Merge a params mapping with explicit keyword arguments.

    Values in ``kwargs`` intentionally override values from ``params``. When
    ``allowed_keys`` is supplied, unknown keys are rejected with a clear error
    before downstream code starts doing partial work.
    """

    if params is None:
        merged: dict[str, Any] = {}
    elif isinstance(params, Mapping):
        merged = dict(params)
    else:
        raise TypeError(f"{context} params must be a mapping or None.")

    merged.update(kwargs)

    if allowed_keys is not None:
        unknown = sorted(set(merged) - set(allowed_keys))
        if unknown:
            allowed = ", ".join(sorted(allowed_keys))
            unknown_display = ", ".join(unknown)
            raise ValueError(f"Unknown {context} parameter(s): {unknown_display}. Allowed keys: {allowed}.")

    return merged
