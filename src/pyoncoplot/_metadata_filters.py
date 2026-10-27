"""Explicit analysis-cohort predicates; never alter reference denominators."""
from collections.abc import Mapping, Sequence
from numbers import Real
import math

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def metadata_filter_samples(metadata, sample_col, samples, filters):
    """Return eligible identifiers without adding samples to any plotting population.

    Missing includes absent metadata rows. include bypasses only this source's
    predicates. Duplicate records (including missing versus observed) are rejected
    by the caller's metadata validator before evaluation.
    """
    if not filters:
        return None
    if metadata is None:
        raise ValueError("metadata_filters require metadata or source metadata columns in mutation data.")
    table = metadata.set_index(sample_col)
    index = pd.Index(list(dict.fromkeys(str(s) for s in samples)))
    keep = pd.Series(True, index=index)
    for source, rules in filters.items():
        if not isinstance(source, str) or source not in table.columns:
            raise ValueError(f"metadata_filters references unknown source {source!r}.")
        if not isinstance(rules, Mapping) or set(rules) - {"include", "exclude", "min", "max", "missing"}:
            raise ValueError(f"Invalid metadata_filters operators for {source!r}.")
        missing = rules.get("missing", "exclude")
        if missing not in ("exclude", "include", "error"):
            raise ValueError("metadata_filters missing must be exclude, include, or error.")
        values = table[source].reindex(index)
        mask = pd.Series(True, index=index)
        for op in ("include", "exclude"):
            if op in rules:
                choices = rules[op]
                if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                    raise ValueError(f"metadata_filters {op} must be a sequence of scalar values.")
                if any(not pd.api.types.is_scalar(v) or pd.isna(v) for v in choices):
                    raise ValueError("Use missing policy rather than null or nested include/exclude values.")
                selected = values.isin(choices)
                mask &= selected if op == "include" else ~selected
        for op in ("min", "max"):
            if op in rules:
                bound = rules[op]
                if isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(bound):
                    raise ValueError("metadata_filters bounds must be finite numbers.")
                if not is_numeric_dtype(table[source]) or is_bool_dtype(table[source]):
                    raise ValueError("metadata_filters ranges require numeric metadata, not categorical coercion.")
                mask &= values.ge(bound) if op == "min" else values.le(bound)
        if "min" in rules and "max" in rules and rules["min"] > rules["max"]:
            raise ValueError("metadata_filters min must not exceed max.")
        absent = values.isna()
        if missing == "error" and absent.any():
            raise ValueError(f"Missing metadata for filter source {source!r}.")
        mask.loc[absent] = missing == "include"
        keep &= mask
    eligible = set(index[keep])
    if not eligible:
        raise ValueError("Metadata filters matched zero samples.")
    return eligible
