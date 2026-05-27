"""Config and table-loading helpers for public oncoplot params."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import yaml


TABLE_PARAM_KEYS = {"data", "metadata", "tmb_data", "pathway"}


def _is_path_like(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


PathInput = Union[str, os.PathLike]


def _read_table(path: PathInput, *, base_dir: Optional[Path] = None, **read_csv_kwargs: Any) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.is_absolute() and base_dir is not None:
        table_path = base_dir / table_path
    if not table_path.exists():
        raise FileNotFoundError(f"Could not find table file: {table_path}")

    kwargs = dict(read_csv_kwargs)
    if "sep" not in kwargs and "delimiter" not in kwargs:
        suffix = table_path.suffix.lower()
        kwargs["sep"] = "\t" if suffix in {".tsv", ".tab"} else ","
    return pd.read_csv(table_path, **kwargs)


def _coerce_table_param(value: Any, *, key: str, base_dir: Optional[Path]) -> Any:
    if value is None or isinstance(value, pd.DataFrame):
        return value
    if _is_path_like(value):
        return _read_table(value, base_dir=base_dir)
    if isinstance(value, Mapping):
        if "path" not in value:
            raise ValueError(f"Table parameter {key!r} mapping must include a 'path' entry.")
        read_spec = dict(value)
        path = read_spec.pop("path")
        return _read_table(path, base_dir=base_dir, **read_spec)
    return value


def materialize_table_params(params: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> dict[str, Any]:
    """Return params with table-valued entries loaded as DataFrames."""

    out = dict(params)
    for key in TABLE_PARAM_KEYS:
        if key in out:
            out[key] = _coerce_table_param(out[key], key=key, base_dir=base_dir)
    return out


def _extract_key(config: Mapping[str, Any], key: Optional[str]) -> Mapping[str, Any]:
    if not key:
        return config
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Could not find config key path: {key}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise TypeError(f"Config key path {key!r} must resolve to a mapping of oncoplot params.")
    return current


def load_oncoplot_params(path: PathInput, *, key: Optional[str] = None) -> dict[str, Any]:
    """Load oncoplot params from YAML and materialize any table paths."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise TypeError("YAML config root must be a mapping.")
    selected = _extract_key(config, key)
    return materialize_table_params(selected, base_dir=config_path.parent)
