"""Data validation and transformation for oncoplots."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._utils import reorder_by_priority, unique_preserve_order


@dataclass
class PathwayGroup:
    """Contiguous pathway block after gene ordering."""

    name: str
    genes: List[str]
    start: int
    end: int


@dataclass
class MetadataTrackInfo:
    """Renderer-neutral metadata summary for one annotation track."""

    column: str
    kind: str
    levels: List[str]
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    missing_samples: Optional[List[str]] = None
    level_order_source: Optional[str] = None


@dataclass
class PreparedOncoplotData:
    """Transformed data shared by renderers."""

    tiles: pd.DataFrame
    samples: List[str]
    genes: List[str]
    total_samples: int
    mutation_type_col: Optional[str]
    metadata: Optional[pd.DataFrame] = None
    metadata_cols: Optional[List[str]] = None
    pathway: Optional[pd.DataFrame] = None
    tmb: Optional[pd.DataFrame] = None
    tmb_sample_col: Optional[str] = None
    tmb_value_col: Optional[str] = None
    tmb_type_col: Optional[str] = None
    tmb_render_stacked: bool = False
    tmb_is_custom: bool = False
    pathway_by_gene: Optional[Mapping[str, str]] = None
    pathway_groups: Optional[List[PathwayGroup]] = None
    mutation_counts: Optional[pd.DataFrame] = None
    tmb_totals: Optional[pd.Series] = None
    tmb_type_counts: Optional[pd.DataFrame] = None
    metadata_tracks: Optional[List[MetadataTrackInfo]] = None
    mutation_type_levels: Optional[List[str]] = None
    mutation_type_order_source: Optional[str] = None
    tmb_type_levels: Optional[List[str]] = None
    tmb_type_order_source: Optional[str] = None
    variant_value_col: Optional[str] = None
    variant_value_agg: Optional[str] = None
    variant_value_min: Optional[float] = None
    variant_value_max: Optional[float] = None
    variant_value_cols: Optional[List[str]] = None
    variant_value_missing: str = "blank"
    variant_value_scale: str = "per_column"
    main_grid_rows: Optional[pd.DataFrame] = None
    main_grid_tiles: Optional[pd.DataFrame] = None
    main_grid_mode: str = "legacy"


def _is_dataframe(value: Any) -> bool:
    return isinstance(value, pd.DataFrame)


def _empty_or_missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: isinstance(value, str) and value == "")


def _as_str_series(series: pd.Series) -> pd.Series:
    return series.astype("string").astype(object)


def _observed_category_levels(values: Iterable[object]) -> List[str]:
    series = pd.Series(list(values), dtype="object")
    series = series[~series.isna()]
    return [str(value) for value in pd.unique(series)]


def _categorical_dtype_levels(series: Optional[pd.Series]) -> Optional[List[str]]:
    if series is None or not isinstance(series.dtype, pd.CategoricalDtype):
        return None
    return [str(value) for value in series.cat.categories]


def _coerce_category_order(order: Optional[Sequence[object]], label: str) -> Optional[List[str]]:
    if order is None:
        return None
    if isinstance(order, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of category values, not a string.")
    levels = [str(value) for value in order]
    duplicates = [level for level in unique_preserve_order(levels) if levels.count(level) > 1]
    if duplicates:
        raise ValueError(f"{label} cannot contain duplicate values: {', '.join(duplicates)}")
    return levels


def _coerce_metadata_category_orders(
    orders: Optional[Mapping[str, Sequence[object]]],
) -> dict[str, List[str]]:
    if orders is None:
        return {}
    if not isinstance(orders, Mapping):
        raise TypeError("metadata_category_orders must be a mapping of column name to category order.")
    return {
        str(column): _coerce_category_order(order, f"metadata_category_orders[{str(column)!r}]") or []
        for column, order in orders.items()
    }


def _resolve_category_levels(
    values: Iterable[object],
    *,
    explicit_order: Optional[Sequence[object]] = None,
    categorical_order: Optional[Sequence[object]] = None,
) -> tuple[List[str], str]:
    observed = _observed_category_levels(values)
    if not observed:
        return [], "observed"

    if explicit_order is not None:
        priority = [str(value) for value in explicit_order]
        return reorder_by_priority(observed, priority), "explicit"

    if categorical_order is not None:
        priority = [str(value) for value in categorical_order]
        return reorder_by_priority(observed, priority), "categorical"

    return observed, "observed"


def check_valid_dataframe_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a useful error if a DataFrame is missing required columns."""

    if not _is_dataframe(data):
        raise TypeError("data must be a pandas DataFrame.")
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        if column not in data.columns:
            available = ", ".join(map(str, data.columns))
            raise ValueError(f"Could not find column: {column}. Available columns: {available}")


def _validate_id_column(data: pd.DataFrame, column: str, label: str) -> None:
    check_valid_dataframe_columns(data, [column])
    if _empty_or_missing_mask(data[column]).any():
        raise ValueError(f"{label} column cannot contain missing values or empty strings: {column}")


def _validate_mutation_inputs(
    data: pd.DataFrame,
    gene_col: str,
    sample_col: str,
    mutation_type_col: Optional[str],
    tooltip_col: str,
) -> pd.DataFrame:
    if not _is_dataframe(data):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must contain at least one mutation row.")
    check_valid_dataframe_columns(data, [gene_col, sample_col, tooltip_col])
    _validate_id_column(data, sample_col, "Sample")
    _validate_id_column(data, gene_col, "Gene")
    out = data.copy()
    out[gene_col] = _as_str_series(out[gene_col])
    out[sample_col] = _as_str_series(out[sample_col])
    if mutation_type_col is not None:
        check_valid_dataframe_columns(out, [mutation_type_col])
        if _empty_or_missing_mask(out[mutation_type_col]).any():
            raise ValueError(
                f"Mutation type column cannot contain missing values or empty strings: {mutation_type_col}"
            )
        out[mutation_type_col] = _as_str_series(out[mutation_type_col])
        compound_terms = out[mutation_type_col].astype(str).str.contains("&", regex=False)
        if compound_terms.any():
            examples = ", ".join(pd.unique(out.loc[compound_terms, mutation_type_col].astype(str))[:3])
            raise ValueError(
                "Mutation type values cannot contain ampersand-delimited Sequence Ontology terms. "
                "Please preselect the most severe consequence before plotting. "
                f"Examples: {examples}"
            )
    return out


def _coerce_variant_value_agg(variant_value_agg: str) -> str:
    allowed = {"max", "mean", "median", "min"}
    agg = str(variant_value_agg).lower()
    if agg not in allowed:
        raise ValueError("variant_value_agg must be one of: max, mean, median, min.")
    return agg


def _coerce_variant_value_scale(variant_value_scale: str) -> str:
    scale = str(variant_value_scale).lower()
    if scale not in {"per_column", "shared"}:
        raise ValueError("variant_value_scale must be either 'per_column' or 'shared'.")
    return scale


def _coerce_variant_value_missing(variant_value_missing: object) -> str:
    missing = str(variant_value_missing).lower()
    if missing not in {"blank", "zero"}:
        raise ValueError("variant_value_missing must be either 'blank' or 'zero'.")
    return missing


def _coerce_variant_value_cols(variant_value_cols: Optional[Sequence[str]]) -> Optional[List[str]]:
    if variant_value_cols is None:
        return None
    if isinstance(variant_value_cols, (str, bytes, bytearray)):
        raise TypeError("variant_value_cols must be a sequence of column names, not a string.")
    cols = [str(column) for column in variant_value_cols]
    if not cols:
        raise ValueError("variant_value_cols must contain at least one column name when supplied.")
    duplicates = [column for column in unique_preserve_order(cols) if cols.count(column) > 1]
    if duplicates:
        raise ValueError(f"variant_value_cols cannot contain duplicate columns: {', '.join(duplicates)}")
    return cols


def _validate_variant_value_columns(
    data: pd.DataFrame,
    variant_value_cols: Optional[Sequence[str]],
    *,
    label: str = "variant value column",
) -> None:
    if not variant_value_cols:
        return
    check_valid_dataframe_columns(data, list(variant_value_cols))
    for column in variant_value_cols:
        values = data[column]
        if pd.api.types.is_numeric_dtype(values):
            continue
        non_missing = values.dropna()
        inferred = pd.api.types.infer_dtype(non_missing, skipna=True)
        if non_missing.empty or inferred in {"integer", "floating", "mixed-integer-float"}:
            continue
        raise ValueError(f"{label} must reference numeric columns: {column}")


def _coerce_isin_filters(
    filters: Optional[Mapping[str, Sequence[object]]],
    label: str,
) -> dict[str, List[object]]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise TypeError(f"{label} must be a mapping of column name to allowed values.")
    out: dict[str, List[object]] = {}
    for column, values in filters.items():
        column_name = str(column)
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise TypeError(f"{label}[{column_name!r}] must be a sequence of allowed values, not a string.")
        out[column_name] = list(values)
    return out


def _coerce_numeric_filters(
    filters: Optional[Mapping[str, object]],
    label: str,
) -> dict[str, float]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise TypeError(f"{label} must be a mapping of column name to numeric cutoff.")
    out: dict[str, float] = {}
    for column, cutoff in filters.items():
        column_name = str(column)
        try:
            numeric_cutoff = pd.to_numeric(pd.Series([cutoff]), errors="coerce").iloc[0]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{column_name!r}] cutoff must be numeric and not NaN.") from exc
        if pd.isna(numeric_cutoff):
            raise ValueError(f"{label}[{column_name!r}] cutoff must be numeric and not NaN.")
        out[column_name] = float(numeric_cutoff)
    return out


def _filter_mask(
    data: pd.DataFrame,
    *,
    isin_filters: Mapping[str, Sequence[object]],
    greater_than_filters: Mapping[str, float],
    less_than_filters: Mapping[str, float],
    label: str,
) -> pd.Series:
    columns = unique_preserve_order(
        list(isin_filters)
        + list(greater_than_filters)
        + list(less_than_filters)
    )
    check_valid_dataframe_columns(data, columns)
    mask = pd.Series(True, index=data.index)

    for column, values in isin_filters.items():
        mask &= (data[column].notna() & data[column].isin(values)).fillna(False).astype(bool)

    numeric_columns = unique_preserve_order(list(greater_than_filters) + list(less_than_filters))
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{label} numeric filters must reference numeric columns: {column}")

    for column, cutoff in greater_than_filters.items():
        mask &= (data[column] > cutoff).fillna(False).astype(bool)
    for column, cutoff in less_than_filters.items():
        mask &= (data[column] < cutoff).fillna(False).astype(bool)
    return mask


def _split_sample_filters(
    data: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    *,
    isin_filters: Mapping[str, Sequence[object]],
    greater_than_filters: Mapping[str, float],
    less_than_filters: Mapping[str, float],
) -> tuple[
    dict[str, List[object]],
    dict[str, float],
    dict[str, float],
    dict[str, List[object]],
    dict[str, float],
    dict[str, float],
]:
    metadata_isin: dict[str, List[object]] = {}
    metadata_greater_than: dict[str, float] = {}
    metadata_less_than: dict[str, float] = {}
    data_isin: dict[str, List[object]] = {}
    data_greater_than: dict[str, float] = {}
    data_less_than: dict[str, float] = {}

    for column, values in isin_filters.items():
        if metadata is not None and column in metadata.columns:
            metadata_isin[column] = list(values)
        elif column in data.columns:
            data_isin[column] = list(values)
        else:
            available = ", ".join(map(str, data.columns))
            metadata_available = ", ".join(map(str, metadata.columns)) if metadata is not None else "none"
            raise ValueError(
                f"Could not find sample filter column: {column}. "
                f"Available metadata columns: {metadata_available}. Available mutation columns: {available}"
            )
    for column, cutoff in greater_than_filters.items():
        if metadata is not None and column in metadata.columns:
            metadata_greater_than[column] = cutoff
        elif column in data.columns:
            data_greater_than[column] = cutoff
        else:
            available = ", ".join(map(str, data.columns))
            metadata_available = ", ".join(map(str, metadata.columns)) if metadata is not None else "none"
            raise ValueError(
                f"Could not find sample filter column: {column}. "
                f"Available metadata columns: {metadata_available}. Available mutation columns: {available}"
            )
    for column, cutoff in less_than_filters.items():
        if metadata is not None and column in metadata.columns:
            metadata_less_than[column] = cutoff
        elif column in data.columns:
            data_less_than[column] = cutoff
        else:
            available = ", ".join(map(str, data.columns))
            metadata_available = ", ".join(map(str, metadata.columns)) if metadata is not None else "none"
            raise ValueError(
                f"Could not find sample filter column: {column}. "
                f"Available metadata columns: {metadata_available}. Available mutation columns: {available}"
            )
    return (
        metadata_isin,
        metadata_greater_than,
        metadata_less_than,
        data_isin,
        data_greater_than,
        data_less_than,
    )


def _default_track_label(column: object) -> str:
    return str(column)


def _normalise_main_grid_tracks(
    main_grid_rows: Optional[Sequence[Mapping[str, object]]],
    variant_value_cols: Optional[Sequence[str]],
    *,
    mutation_type_col: Optional[str],
    variant_value_col: Optional[str],
    variant_value_agg: str,
    variant_value_missing: str,
    variant_value_scale: str,
) -> tuple[List[dict[str, object]], str]:
    if main_grid_rows is not None and variant_value_cols is not None:
        raise ValueError("Please specify either main_grid_rows or variant_value_cols, not both.")
    if variant_value_col is not None and (main_grid_rows is not None or variant_value_cols is not None):
        raise ValueError("variant_value_col cannot be combined with main_grid_rows or variant_value_cols.")

    tracks: List[dict[str, object]] = []
    mode = "legacy"

    if main_grid_rows is not None:
        if isinstance(main_grid_rows, Mapping) or isinstance(main_grid_rows, (str, bytes, bytearray)):
            raise TypeError("main_grid_rows must be a sequence of row-spec mappings.")
        allowed_keys = {"kind", "label", "column", "agg", "palette", "missing"}
        for index, raw_spec in enumerate(main_grid_rows):
            if not isinstance(raw_spec, Mapping):
                raise TypeError("Each main_grid_rows entry must be a mapping.")
            unknown = set(raw_spec) - allowed_keys
            if unknown:
                raise ValueError(f"Unknown main_grid_rows keys: {', '.join(sorted(unknown))}")
            kind = str(raw_spec.get("kind", "")).lower()
            if kind not in {"mutation_type", "variant_value"}:
                raise ValueError("main_grid_rows entries must have kind 'mutation_type' or 'variant_value'.")
            if kind == "mutation_type":
                if "column" in raw_spec:
                    raise ValueError("main_grid_rows mutation_type entries cannot define a column.")
                label = str(raw_spec.get("label") or "Variant type")
                tracks.append(
                    {
                        "TrackId": f"row{index}",
                        "TrackIndex": index,
                        "Kind": "mutation_type",
                        "Label": label,
                        "VariantValueColumn": None,
                        "VariantValueAgg": None,
                        "VariantValuePalette": None,
                        "VariantValueKey": None,
                        "VariantValueMissing": None,
                    }
                )
                continue
            if "column" not in raw_spec or raw_spec.get("column") in (None, ""):
                raise ValueError("main_grid_rows variant_value entries must define a column.")
            agg = _coerce_variant_value_agg(str(raw_spec.get("agg", variant_value_agg)))
            palette = raw_spec.get("palette")
            if variant_value_scale == "shared" and palette is not None:
                raise ValueError("Row-level palettes are not supported when variant_value_scale='shared'.")
            column = str(raw_spec["column"])
            missing = _coerce_variant_value_missing(raw_spec.get("missing", variant_value_missing))
            tracks.append(
                {
                    "TrackId": f"row{index}",
                    "TrackIndex": index,
                    "Kind": "variant_value",
                    "Label": str(raw_spec.get("label") or _default_track_label(column)),
                    "VariantValueColumn": column,
                    "VariantValueAgg": agg,
                    "VariantValuePalette": palette,
                    "VariantValueKey": None,
                    "VariantValueMissing": missing,
                }
            )
        if not tracks:
            raise ValueError("main_grid_rows must contain at least one row specification.")
        return tracks, "expanded"

    if variant_value_cols is not None:
        mode = "expanded"
        track_index = 0
        if mutation_type_col is not None:
            tracks.append(
                {
                    "TrackId": "row0",
                    "TrackIndex": track_index,
                    "Kind": "mutation_type",
                    "Label": "Variant type",
                    "VariantValueColumn": None,
                    "VariantValueAgg": None,
                    "VariantValuePalette": None,
                    "VariantValueKey": None,
                    "VariantValueMissing": None,
                }
            )
            track_index += 1
        for column in variant_value_cols:
            tracks.append(
                {
                    "TrackId": f"row{track_index}",
                    "TrackIndex": track_index,
                    "Kind": "variant_value",
                    "Label": _default_track_label(column),
                    "VariantValueColumn": str(column),
                    "VariantValueAgg": variant_value_agg,
                    "VariantValuePalette": None,
                    "VariantValueKey": None,
                    "VariantValueMissing": variant_value_missing,
                }
            )
            track_index += 1
        return tracks, mode

    if variant_value_col is not None:
        return [
            {
                "TrackId": "row0",
                "TrackIndex": 0,
                "Kind": "variant_value",
                "Label": _default_track_label(variant_value_col),
                "VariantValueColumn": str(variant_value_col),
                "VariantValueAgg": variant_value_agg,
                "VariantValuePalette": None,
                "VariantValueKey": None,
                "VariantValueMissing": variant_value_missing,
            }
        ], "legacy"

    return [
        {
            "TrackId": "row0",
            "TrackIndex": 0,
            "Kind": "mutation_type",
            "Label": "Variant type",
            "VariantValueColumn": None,
            "VariantValueAgg": None,
            "VariantValuePalette": None,
            "VariantValueKey": None,
            "VariantValueMissing": None,
        }
    ], "legacy"


def _assign_variant_value_keys(
    tracks: Sequence[Mapping[str, object]],
) -> tuple[List[dict[str, object]], List[dict[str, object]]]:
    keyed_tracks = [dict(track) for track in tracks]
    value_specs: List[dict[str, object]] = []
    key_by_column_agg: dict[tuple[str, str, str], str] = {}
    for track in keyed_tracks:
        if track["Kind"] != "variant_value":
            continue
        column = str(track["VariantValueColumn"])
        agg = str(track["VariantValueAgg"])
        missing = str(track["VariantValueMissing"])
        key = key_by_column_agg.get((column, agg, missing))
        if key is None:
            key = f"VariantValue__{len(value_specs)}"
            key_by_column_agg[(column, agg, missing)] = key
            value_specs.append({"column": column, "agg": agg, "missing": missing, "key": key})
        track["VariantValueKey"] = key
    return keyed_tracks, value_specs


def _validate_metadata(
    metadata: Optional[pd.DataFrame],
    metadata_sample_col: str,
) -> Optional[pd.DataFrame]:
    if metadata is None:
        return None
    if not _is_dataframe(metadata):
        raise TypeError("metadata must be a pandas DataFrame.")
    check_valid_dataframe_columns(metadata, [metadata_sample_col])
    if metadata[metadata_sample_col].duplicated().any():
        duplicates = metadata.loc[metadata[metadata_sample_col].duplicated(), metadata_sample_col]
        raise ValueError(
            "metadata sample column must contain unique identifiers. "
            f"Duplicated values: {', '.join(map(str, pd.unique(duplicates)))}"
        )
    out = metadata.copy()
    out[metadata_sample_col] = _as_str_series(out[metadata_sample_col])
    return out


def _metadata_columns_list(metadata_cols: Sequence[str]) -> List[str]:
    if isinstance(metadata_cols, str):
        return [metadata_cols]
    return list(metadata_cols)


def _derive_metadata_from_data(
    data: pd.DataFrame,
    metadata_sample_col: str,
    metadata_cols: Sequence[str],
) -> pd.DataFrame:
    columns = unique_preserve_order([metadata_sample_col] + _metadata_columns_list(metadata_cols))
    check_valid_dataframe_columns(data, columns)
    return data.copy().loc[:, columns].drop_duplicates()


def _validate_pathway(
    pathway: Optional[pd.DataFrame],
    pathway_gene_col: Optional[str],
    gene_col: str,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    if pathway is None:
        return None, None
    if not _is_dataframe(pathway):
        raise TypeError("pathway must be a pandas DataFrame.")
    if pathway.shape[1] != 2:
        raise ValueError("pathway must have exactly two columns: gene and pathway.")
    pathway_gene_col = pathway_gene_col or gene_col
    check_valid_dataframe_columns(pathway, [pathway_gene_col])
    pathway_col = [column for column in pathway.columns if column != pathway_gene_col][0]
    if _empty_or_missing_mask(pathway[pathway_gene_col]).any():
        raise ValueError("pathway gene column cannot contain missing values or empty strings.")
    if _empty_or_missing_mask(pathway[pathway_col]).any():
        raise ValueError("pathway name column cannot contain missing values or empty strings.")
    if pathway[pathway_gene_col].duplicated().any():
        raise ValueError("pathway gene column cannot contain duplicates.")
    if (pathway[pathway_col].astype(str) == "Other").any():
        raise ValueError("The pathway name 'Other' is reserved for unmapped genes.")
    out = pathway[[pathway_gene_col, pathway_col]].copy()
    out[pathway_gene_col] = _as_str_series(out[pathway_gene_col])
    out[pathway_col] = _as_str_series(out[pathway_col])
    out.columns = ["Gene", "Pathway"]
    return out, pathway_col


def identify_top_genes(
    data: pd.DataFrame,
    gene_col: str,
    sample_col: str,
    top_n: Optional[float] = 10,
    ignore_genes: Optional[Sequence[str]] = None,
    return_extra_genes_if_tied: bool = False,
    verbose: bool = False,
) -> List[str]:
    """Identify genes with the most distinct mutated samples."""

    check_valid_dataframe_columns(data, [gene_col, sample_col])
    if top_n is None:
        top_n = math.inf
    if not (top_n == math.inf or top_n > 0):
        raise ValueError("top_n must be positive, None, or math.inf.")

    distinct = data[[sample_col, gene_col]].drop_duplicates()
    counts = (
        distinct.groupby(gene_col, dropna=False)[sample_col]
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", gene_col], ascending=[False, True], kind="mergesort")
    )

    if ignore_genes:
        counts = counts[~counts[gene_col].isin(ignore_genes)]

    if top_n == math.inf:
        selected = counts
    elif return_extra_genes_if_tied:
        n = int(top_n)
        if len(counts) <= n:
            selected = counts
        else:
            cutoff = counts.iloc[n - 1]["count"]
            selected = counts[counts["count"] >= cutoff]
    else:
        selected = counts.head(int(top_n))

    genes = selected[gene_col].astype(str).tolist()
    if verbose and top_n != math.inf:
        if len(genes) > int(top_n):
            warnings.warn(
                f"The top {int(top_n)} genes were requested, but {len(genes)} were returned due to ties.",
                stacklevel=2,
            )
        elif len(genes) < int(top_n):
            warnings.warn(
                f"The top {int(top_n)} genes were requested, but only {len(genes)} genes were available.",
                stacklevel=2,
            )
    return genes


def _get_genes_for_oncoplot(
    data: pd.DataFrame,
    gene_col: str,
    sample_col: str,
    top_n: Optional[float],
    include_genes: Optional[Sequence[str]],
    ignore_genes: Optional[Sequence[str]],
    return_extra_genes_if_tied: bool,
    verbose: bool,
) -> List[str]:
    if include_genes is not None:
        available = set(data[gene_col].astype(str))
        include = [str(gene) for gene in include_genes]
        missing = [gene for gene in include if gene not in available]
        if len(missing) == len(include):
            raise ValueError(
                "Couldn't find any of the genes supplied in the dataset. "
                "Either no samples have mutations in these genes, or the gene names differ."
            )
        if missing and verbose:
            warnings.warn(
                "Failed to find the following genes in the dataset: " + ", ".join(missing),
                stacklevel=2,
            )
        ignored = set(ignore_genes or [])
        return [gene for gene in include if gene not in missing and gene not in ignored]

    return identify_top_genes(
        data=data,
        gene_col=gene_col,
        sample_col=sample_col,
        top_n=top_n,
        ignore_genes=ignore_genes,
        return_extra_genes_if_tied=return_extra_genes_if_tied,
        verbose=verbose,
    )


def score_sample_by_gene_rank(
    mutated_genes: Sequence[str],
    genes_informing_score: Sequence[str],
    gene_rank: Sequence[float],
    debug_mode: bool = False,
) -> int:
    """Score one sample so high-rank mutated genes sort first."""

    if len(genes_informing_score) != len(gene_rank):
        raise ValueError("genes_informing_score and gene_rank must have the same length.")

    indexed = list(enumerate(gene_rank))
    indexed.sort(key=lambda pair: (pair[1], pair[0]))
    stable_ranks = [0] * len(indexed)
    for rank_index, (original_index, _value) in enumerate(indexed, start=1):
        stable_ranks[original_index] = rank_index

    mutated = set(map(str, mutated_genes))
    total = 0
    for gene, rank in zip(genes_informing_score, stable_ranks):
        if str(gene) in mutated:
            total += 2 ** (rank - 1)
    return total


def rank_genes_by_pathway(
    gene_pathway_map: pd.DataFrame,
    gene_ranks: Optional[Sequence[str]] = None,
    pathway_ranks: Optional[Sequence[str]] = None,
) -> List[str]:
    """Sort genes first by pathway order, then by gene order."""

    if gene_pathway_map.shape[1] != 2:
        raise ValueError("gene_pathway_map must have exactly two columns.")
    mapping = gene_pathway_map.copy()
    mapping.columns = ["Gene", "Pathway"]
    if gene_ranks is None:
        gene_ranks = unique_preserve_order(mapping["Gene"].astype(str))
    if pathway_ranks is None:
        pathway_ranks = unique_preserve_order(mapping["Pathway"].astype(str))

    df = pd.DataFrame({"Gene": list(gene_ranks), "GeneRank": range(len(gene_ranks))})
    path_by_gene = dict(zip(mapping["Gene"].astype(str), mapping["Pathway"].astype(str)))
    path_rank = {pathway: index for index, pathway in enumerate(pathway_ranks)}
    df["Pathway"] = df["Gene"].map(path_by_gene)
    df["PathwayRank"] = df["Pathway"].map(path_rank)
    df["PathwayRank"] = df["PathwayRank"].fillna(len(path_rank) + 1)
    df = df.sort_values(["PathwayRank", "GeneRank"], kind="mergesort")
    return df["Gene"].astype(str).tolist()


def _pathway_summary(
    genes: Sequence[str],
    pathway: Optional[pd.DataFrame],
) -> tuple[Optional[Mapping[str, str]], Optional[List[PathwayGroup]]]:
    if pathway is None:
        return None, None
    path_by_gene = dict(zip(pathway["Gene"].astype(str), pathway["Pathway"].astype(str)))
    pathway_by_gene = {str(gene): path_by_gene.get(str(gene), "Other") for gene in genes}
    groups: List[PathwayGroup] = []
    for pathway_name in unique_preserve_order(pathway_by_gene[gene] for gene in genes):
        indexes = [index for index, gene in enumerate(genes) if pathway_by_gene[gene] == pathway_name]
        groups.append(
            PathwayGroup(
                name=str(pathway_name),
                genes=[str(genes[index]) for index in indexes],
                start=min(indexes),
                end=max(indexes),
            )
        )
    return pathway_by_gene, groups


def _aggregate_variant_value(series: pd.Series, agg: str, missing: str) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if missing == "zero":
        values = values.fillna(0)
    else:
        values = values.dropna()
    if values.empty:
        return np.nan
    return float(getattr(values, agg)())


def _collapse_mutations(
    data: pd.DataFrame,
    gene_col: str,
    sample_col: str,
    mutation_type_col: Optional[str],
    tooltip_col: str,
    variant_value_specs: Optional[Sequence[Mapping[str, object]]] = None,
    legacy_variant_value_key: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    variant_value_specs = list(variant_value_specs or [])
    value_columns = [str(spec["key"]) for spec in variant_value_specs]
    for (sample, gene), group in data.groupby([sample_col, gene_col], sort=False, dropna=False):
        tooltip_values = unique_preserve_order(group[tooltip_col].astype(str).tolist())
        mutation_count = len(group)
        if mutation_type_col is None:
            mutation_type = np.nan
        else:
            mutation_values = unique_preserve_order(group[mutation_type_col].astype(str).tolist())
            mutation_type = "; ".join(mutation_values)
            if mutation_count > 1:
                mutation_type = "Multi_Hit"
        tooltip = "<br>".join(tooltip_values)
        if tooltip_col != sample_col:
            tooltip = f"Sample: {sample}<br>{tooltip}"
        aggregated_values = {
            str(spec["key"]): _aggregate_variant_value(
                group[str(spec["column"])],
                str(spec["agg"]),
                str(spec["missing"]),
            )
            for spec in variant_value_specs
        }
        variant_value = (
            aggregated_values.get(legacy_variant_value_key, np.nan)
            if legacy_variant_value_key is not None
            else np.nan
        )
        rows.append(
            {
                "Sample": str(sample),
                "Gene": str(gene),
                "MutationType": mutation_type,
                "MutationCount": mutation_count,
                "VariantValue": variant_value,
                "Tooltip": tooltip,
                **aggregated_values,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Sample", "Gene", "MutationType", "MutationCount", "VariantValue", "Tooltip"] + value_columns,
    )


def _empty_main_grid_rows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "RowId",
            "RowIndex",
            "Gene",
            "GeneIndex",
            "TrackId",
            "TrackIndex",
            "Kind",
            "Label",
            "VariantValueColumn",
            "VariantValueAgg",
            "VariantValuePalette",
            "VariantValueKey",
            "VariantValueMissing",
            "VariantValueMin",
            "VariantValueMax",
            "ScaleGroup",
        ]
    )


def _empty_main_grid_tiles() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Sample",
            "Gene",
            "MutationType",
            "MutationCount",
            "Tooltip",
            "RowId",
            "RowIndex",
            "GeneIndex",
            "TrackId",
            "TrackIndex",
            "Kind",
            "Label",
            "VariantValueColumn",
            "VariantValueAgg",
            "VariantValuePalette",
            "VariantValueMissing",
            "VariantValue",
            "VariantValueMin",
            "VariantValueMax",
            "ScaleGroup",
        ]
    )


def _variant_track_ranges(
    tiles: pd.DataFrame,
    tracks: Sequence[Mapping[str, object]],
    variant_value_scale: str,
) -> dict[str, tuple[Optional[float], Optional[float], str]]:
    ranges: dict[str, tuple[Optional[float], Optional[float], str]] = {}
    variant_tracks = [track for track in tracks if track["Kind"] == "variant_value"]
    if not variant_tracks:
        return ranges

    shared_values: List[float] = []
    if variant_value_scale == "shared" and not tiles.empty:
        for track in variant_tracks:
            key = str(track["VariantValueKey"])
            if key in tiles:
                shared_values.extend(tiles[key].astype(float).dropna().tolist())
    shared_min = float(np.min(shared_values)) if shared_values else None
    shared_max = float(np.max(shared_values)) if shared_values else None

    for track in variant_tracks:
        track_id = str(track["TrackId"])
        if variant_value_scale == "shared":
            ranges[track_id] = (shared_min, shared_max, "variant_value_shared")
            continue
        key = str(track["VariantValueKey"])
        if tiles.empty or key not in tiles:
            ranges[track_id] = (None, None, track_id)
            continue
        values = tiles[key].astype(float).dropna()
        if values.empty:
            ranges[track_id] = (None, None, track_id)
            continue
        ranges[track_id] = (float(values.min()), float(values.max()), track_id)
    return ranges


def _build_main_grid_rows(
    genes: Sequence[str],
    tracks: Sequence[Mapping[str, object]],
    tiles: pd.DataFrame,
    variant_value_scale: str,
) -> pd.DataFrame:
    if not genes or not tracks:
        return _empty_main_grid_rows()
    ranges = _variant_track_ranges(tiles, tracks, variant_value_scale)
    rows = []
    row_index = 0
    for gene_index, gene in enumerate(genes):
        for track in tracks:
            track_id = str(track["TrackId"])
            value_min, value_max, scale_group = ranges.get(track_id, (None, None, ""))
            rows.append(
                {
                    "RowId": f"{gene}::{track_id}",
                    "RowIndex": row_index,
                    "Gene": str(gene),
                    "GeneIndex": gene_index,
                    "TrackId": track_id,
                    "TrackIndex": int(track["TrackIndex"]),
                    "Kind": str(track["Kind"]),
                    "Label": str(track["Label"]),
                    "VariantValueColumn": track["VariantValueColumn"],
                    "VariantValueAgg": track["VariantValueAgg"],
                    "VariantValuePalette": track["VariantValuePalette"],
                    "VariantValueKey": track["VariantValueKey"],
                    "VariantValueMissing": track["VariantValueMissing"],
                    "VariantValueMin": value_min,
                    "VariantValueMax": value_max,
                    "ScaleGroup": scale_group,
                }
            )
            row_index += 1
    return pd.DataFrame(rows, columns=_empty_main_grid_rows().columns)


def _build_main_grid_tiles(
    tiles: pd.DataFrame,
    tracks: Sequence[Mapping[str, object]],
    main_grid_rows: pd.DataFrame,
    include_variant_summaries: bool = False,
) -> pd.DataFrame:
    if tiles.empty or main_grid_rows.empty or not tracks:
        return _empty_main_grid_tiles()
    rows_by_gene = {
        str(gene): group.sort_values("TrackIndex", kind="mergesort")
        for gene, group in main_grid_rows.groupby("Gene", sort=False, observed=False)
    }
    rows = []
    for _tile_index, tile in tiles.iterrows():
        gene = str(tile["Gene"])
        gene_rows = rows_by_gene.get(gene, pd.DataFrame())
        variant_summaries: dict[str, str] = {}
        if include_variant_summaries:
            for _variant_index, variant_spec in gene_rows.iterrows():
                if variant_spec["Kind"] != "variant_value":
                    continue
                key = variant_spec["VariantValueKey"]
                value = tile[str(key)]
                if pd.isna(value):
                    continue
                variant_summaries[str(variant_spec["TrackId"])] = (
                    f"{variant_spec['Label']}: {float(value):g}"
                )
        for _row_index, row_spec in gene_rows.iterrows():
            variant_value = np.nan
            if row_spec["Kind"] == "variant_value":
                key = row_spec["VariantValueKey"]
                value = tile[str(key)]
                variant_value = np.nan if pd.isna(value) else float(value)
            tooltip = str(tile["Tooltip"])
            if include_variant_summaries:
                if row_spec["Kind"] == "variant_value":
                    summaries = [variant_summaries[str(row_spec["TrackId"])]] if str(row_spec["TrackId"]) in variant_summaries else []
                else:
                    summaries = list(variant_summaries.values())
                if summaries:
                    tooltip = f"{tooltip}<br>{'<br>'.join(summaries)}"
            rows.append(
                {
                    "Sample": str(tile["Sample"]),
                    "Gene": gene,
                    "MutationType": tile["MutationType"],
                    "MutationCount": int(tile["MutationCount"]),
                    "Tooltip": tooltip,
                    "RowId": str(row_spec["RowId"]),
                    "RowIndex": int(row_spec["RowIndex"]),
                    "GeneIndex": int(row_spec["GeneIndex"]),
                    "TrackId": str(row_spec["TrackId"]),
                    "TrackIndex": int(row_spec["TrackIndex"]),
                    "Kind": str(row_spec["Kind"]),
                    "Label": str(row_spec["Label"]),
                    "VariantValueColumn": row_spec["VariantValueColumn"],
                    "VariantValueAgg": row_spec["VariantValueAgg"],
                    "VariantValuePalette": row_spec["VariantValuePalette"],
                    "VariantValueMissing": row_spec["VariantValueMissing"],
                    "VariantValue": variant_value,
                    "VariantValueMin": row_spec["VariantValueMin"],
                    "VariantValueMax": row_spec["VariantValueMax"],
                    "ScaleGroup": row_spec["ScaleGroup"],
                }
            )
    return pd.DataFrame(rows, columns=_empty_main_grid_tiles().columns)


def _metadata_sort_order(
    metadata: pd.DataFrame,
    metadata_sample_col: str,
    samples_to_show: Sequence[str],
    metadata_sort_cols: Sequence[str],
    metadata_sort_desc: Sequence[bool],
    metadata_sort_by: Sequence[str],
) -> List[str]:
    check_valid_dataframe_columns(metadata, list(metadata_sort_cols))
    sort_df = metadata[metadata[metadata_sample_col].isin(samples_to_show)].copy()
    sort_df["_gene_rank"] = sort_df[metadata_sample_col].map(
        {sample: index for index, sample in enumerate(samples_to_show)}
    )

    ascending = []
    sort_columns = []
    for index, column in enumerate(metadata_sort_cols):
        sort_column = f"_sort_{column}"
        series = sort_df[column]
        if pd.api.types.is_numeric_dtype(series):
            sort_df[sort_column] = series
        elif metadata_sort_by[index] == "frequency":
            counts = series.value_counts(dropna=False)
            sort_df[sort_column] = series.map(counts)
        elif metadata_sort_by[index] == "alphabetical":
            sort_df[sort_column] = series.astype("string")
        else:
            raise ValueError("metadata_sort_by values must be 'frequency' or 'alphabetical'.")
        sort_columns.append(sort_column)
        ascending.append(not metadata_sort_desc[index])

    sort_columns.append("_gene_rank")
    ascending.append(True)
    sort_df = sort_df.sort_values(sort_columns, ascending=ascending, na_position="last", kind="mergesort")
    return sort_df[metadata_sample_col].astype(str).tolist()


def _prepare_tmb_data(
    data: pd.DataFrame,
    samples_to_show: Sequence[str],
    sample_col: str,
    mutation_type_col: Optional[str],
    tmb_data: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str], bool, Optional[List[str]]]:
    type_category_order = None
    if tmb_data is not None:
        if not _is_dataframe(tmb_data):
            raise TypeError("tmb_data must be a pandas DataFrame.")
        if tmb_data.shape[1] not in (2, 3):
            raise ValueError("tmb_data must have either 2 or 3 columns.")
        check_valid_dataframe_columns(tmb_data, [sample_col])
        if _empty_or_missing_mask(tmb_data[sample_col]).any():
            raise ValueError("tmb_data sample column cannot contain missing values or empty strings.")
        non_sample_cols = [column for column in tmb_data.columns if column != sample_col]
        numeric_cols = [column for column in non_sample_cols if pd.api.types.is_numeric_dtype(tmb_data[column])]
        if not numeric_cols:
            raise ValueError("Could not find a numeric column in tmb_data to visualise as TMB.")
        value_col = numeric_cols[0]
        if tmb_data[value_col].isna().any():
            raise ValueError("tmb_data numeric value column cannot contain missing values.")
        if tmb_data.shape[1] == 2 and tmb_data[sample_col].duplicated().any():
            raise ValueError("tmb_data sample column cannot contain duplicates when no type column is supplied.")
        type_candidates = [column for column in tmb_data.columns if column not in {sample_col, value_col}]
        type_col = type_candidates[0] if type_candidates else "__tmb_type"
        if type_col in tmb_data.columns:
            type_category_order = _categorical_dtype_levels(tmb_data[type_col])
        out = tmb_data.copy()
        out[sample_col] = _as_str_series(out[sample_col])
        if type_col not in out.columns:
            out[type_col] = np.nan
        else:
            if out[type_col].isna().any():
                raise ValueError("tmb_data type column cannot contain missing values.")
            out[type_col] = _as_str_series(out[type_col])
    else:
        if mutation_type_col is None:
            out = data.groupby(sample_col, dropna=False).size().rename("Mutations").reset_index()
            out["__tmb_type"] = np.nan
            value_col = "Mutations"
            type_col = "__tmb_type"
        else:
            out = (
                data.groupby([sample_col, mutation_type_col], dropna=False)
                .size()
                .rename("Mutations")
                .reset_index()
            )
            value_col = "Mutations"
            type_col = mutation_type_col

    out = out[out[sample_col].isin(samples_to_show)].copy()
    out[sample_col] = pd.Categorical(out[sample_col].astype(str), categories=list(samples_to_show), ordered=True)
    out = out.sort_values(sample_col, kind="mergesort")
    render_stacked = not out[type_col].isna().all()
    return out, sample_col, value_col, type_col, render_stacked, type_category_order


def _custom_tmb_sample_ids(tmb_data: Optional[pd.DataFrame], sample_col: str) -> List[str]:
    if tmb_data is None:
        return []
    if not _is_dataframe(tmb_data):
        raise TypeError("tmb_data must be a pandas DataFrame.")
    check_valid_dataframe_columns(tmb_data, [sample_col])
    if _empty_or_missing_mask(tmb_data[sample_col]).any():
        raise ValueError("tmb_data sample column cannot contain missing values or empty strings.")
    return unique_preserve_order(_as_str_series(tmb_data[sample_col]))


def _summarise_mutation_counts(tiles: pd.DataFrame) -> pd.DataFrame:
    columns = ["Gene", "MutationType", "Count"]
    if tiles.empty:
        return pd.DataFrame(columns=columns)
    return (
        tiles.groupby(["Gene", "MutationType"], observed=False, dropna=False)
        .size()
        .rename("Count")
        .reset_index()
        .loc[:, columns]
    )


def _summarise_tmb(
    tmb: Optional[pd.DataFrame],
    sample_col: Optional[str],
    value_col: Optional[str],
    type_col: Optional[str],
    samples: Sequence[str],
    type_levels: Optional[Sequence[str]] = None,
) -> tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    if tmb is None or sample_col is None or value_col is None:
        return None, None
    totals = (
        tmb.groupby(sample_col, observed=False)[value_col]
        .sum()
        .reindex(samples, fill_value=0)
        .astype(float)
    )
    if type_col is None or tmb[type_col].isna().all():
        return totals, None
    type_counts = (
        tmb.pivot_table(
            index=sample_col,
            columns=type_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(samples, fill_value=0)
        .astype(float)
    )
    if type_levels is not None:
        type_counts = type_counts.reindex(columns=list(type_levels), fill_value=0)
    return totals, type_counts


def _summarise_metadata_tracks(
    metadata: Optional[pd.DataFrame],
    metadata_cols: Optional[Sequence[str]],
    samples: Sequence[str],
    metadata_category_orders: Optional[Mapping[str, Sequence[object]]] = None,
) -> Optional[List[MetadataTrackInfo]]:
    if metadata is None or not metadata_cols:
        return None
    metadata_by_sample = metadata.set_index("Sample")
    explicit_orders = _coerce_metadata_category_orders(metadata_category_orders)
    tracks: List[MetadataTrackInfo] = []
    for column in metadata_cols:
        values = metadata_by_sample[column].reindex(samples)
        missing_samples = [str(sample) for sample, value in values.items() if pd.isna(value)]
        if pd.api.types.is_numeric_dtype(values):
            tracks.append(
                MetadataTrackInfo(
                    column=str(column),
                    kind="numeric",
                    levels=[],
                    level_order_source=None,
                    value_min=float(values.min(skipna=True)) if values.notna().any() else None,
                    value_max=float(values.max(skipna=True)) if values.notna().any() else None,
                    missing_samples=missing_samples,
                )
            )
            continue
        levels, level_order_source = _resolve_category_levels(
            values,
            explicit_order=explicit_orders.get(str(column)),
            categorical_order=_categorical_dtype_levels(values),
        )
        tracks.append(
            MetadataTrackInfo(
                column=str(column),
                kind="categorical",
                levels=levels,
                level_order_source=level_order_source,
                missing_samples=missing_samples,
            )
        )
    return tracks


def prepare_oncoplot_data(
    data: pd.DataFrame,
    *,
    gene_col: str,
    sample_col: str,
    mutation_type_col: Optional[str] = None,
    tooltip_col: Optional[str] = None,
    include_genes: Optional[Sequence[str]] = None,
    ignore_genes: Optional[Sequence[str]] = None,
    top_n: Optional[float] = 10,
    return_extra_genes_if_tied: bool = False,
    metadata: Optional[pd.DataFrame] = None,
    metadata_sample_col: Optional[str] = None,
    metadata_cols: Optional[Sequence[str]] = None,
    metadata_require_mutations: bool = True,
    filter_samples_by_isin_lists: Optional[Mapping[str, Sequence[object]]] = None,
    filter_samples_by_greater_than: Optional[Mapping[str, object]] = None,
    filter_samples_by_less_than: Optional[Mapping[str, object]] = None,
    filter_mutations_by_isin_lists: Optional[Mapping[str, Sequence[object]]] = None,
    filter_mutations_by_greater_than: Optional[Mapping[str, object]] = None,
    filter_mutations_by_less_than: Optional[Mapping[str, object]] = None,
    pathway: Optional[pd.DataFrame] = None,
    pathway_gene_col: Optional[str] = None,
    show_all_samples: bool = False,
    total_samples: str = "any_mutations",
    sample_order: Optional[Sequence[str]] = None,
    metadata_sort_cols: Optional[Sequence[str]] = None,
    metadata_sort_desc: Any = True,
    metadata_sort_by: Any = "frequency",
    mutation_type_order: Optional[Sequence[object]] = None,
    metadata_category_orders: Optional[Mapping[str, Sequence[object]]] = None,
    tmb_type_order: Optional[Sequence[object]] = None,
    tmb_data: Optional[pd.DataFrame] = None,
    prepare_tmb: bool = False,
    variant_value_col: Optional[str] = None,
    variant_value_cols: Optional[Sequence[str]] = None,
    variant_value_agg: str = "max",
    variant_value_missing: str = "blank",
    variant_value_scale: str = "per_column",
    main_grid_rows: Optional[Sequence[Mapping[str, object]]] = None,
    verbose: bool = False,
) -> PreparedOncoplotData:
    """Validate and transform mutation-level data into oncoplot-ready tables."""

    tooltip_was_generated = tooltip_col is None
    validation_tooltip_col = sample_col if tooltip_was_generated else tooltip_col
    metadata_sample_col = metadata_sample_col or sample_col
    total_samples_options = {"any_mutations", "all", "oncoplot"}
    if total_samples not in total_samples_options:
        raise ValueError("total_samples must be one of: any_mutations, all, oncoplot.")
    if sample_order is not None and metadata_sort_cols is not None:
        raise ValueError("Please specify either sample_order or metadata_sort_cols, not both.")

    mutation_type_explicit_order = _coerce_category_order(mutation_type_order, "mutation_type_order")
    metadata_category_orders = _coerce_metadata_category_orders(metadata_category_orders)
    tmb_type_explicit_order = _coerce_category_order(tmb_type_order, "tmb_type_order")
    variant_value_agg = _coerce_variant_value_agg(variant_value_agg)
    variant_value_missing = _coerce_variant_value_missing(variant_value_missing)
    variant_value_scale = _coerce_variant_value_scale(variant_value_scale)
    sample_isin_filters = _coerce_isin_filters(filter_samples_by_isin_lists, "filter_samples_by_isin_lists")
    sample_greater_than_filters = _coerce_numeric_filters(
        filter_samples_by_greater_than,
        "filter_samples_by_greater_than",
    )
    sample_less_than_filters = _coerce_numeric_filters(
        filter_samples_by_less_than,
        "filter_samples_by_less_than",
    )
    mutation_isin_filters = _coerce_isin_filters(filter_mutations_by_isin_lists, "filter_mutations_by_isin_lists")
    mutation_greater_than_filters = _coerce_numeric_filters(
        filter_mutations_by_greater_than,
        "filter_mutations_by_greater_than",
    )
    mutation_less_than_filters = _coerce_numeric_filters(
        filter_mutations_by_less_than,
        "filter_mutations_by_less_than",
    )
    sample_filters_active = bool(sample_isin_filters or sample_greater_than_filters or sample_less_than_filters)
    mutation_filters_active = bool(mutation_isin_filters or mutation_greater_than_filters or mutation_less_than_filters)
    variant_value_cols = _coerce_variant_value_cols(variant_value_cols)
    main_grid_tracks, main_grid_mode = _normalise_main_grid_tracks(
        main_grid_rows,
        variant_value_cols,
        mutation_type_col=mutation_type_col,
        variant_value_col=variant_value_col,
        variant_value_agg=variant_value_agg,
        variant_value_missing=variant_value_missing,
        variant_value_scale=variant_value_scale,
    )
    main_grid_tracks, variant_value_specs = _assign_variant_value_keys(main_grid_tracks)
    variant_value_columns = unique_preserve_order(
        [str(spec["column"]) for spec in variant_value_specs]
    )
    legacy_variant_value_key = next(
        (
            str(track["VariantValueKey"])
            for track in main_grid_tracks
            if track["Kind"] == "variant_value"
            and variant_value_col is not None
            and str(track["VariantValueColumn"]) == str(variant_value_col)
        ),
        None,
    )
    mutation_type_categorical_order = (
        _categorical_dtype_levels(data[mutation_type_col])
        if _is_dataframe(data) and mutation_type_col is not None and mutation_type_col in data.columns
        else None
    )

    data = _validate_mutation_inputs(data, gene_col, sample_col, mutation_type_col, validation_tooltip_col)
    if tooltip_was_generated:
        tooltip_col = "__pyoncoplot_default_tooltip__"
        while tooltip_col in data.columns:
            tooltip_col = f"_{tooltip_col}"
        data[tooltip_col] = data[gene_col].astype(str)
        if mutation_type_col is not None:
            data[tooltip_col] = data[tooltip_col] + ": " + data[mutation_type_col].astype(str)
    else:
        tooltip_col = str(tooltip_col)
    _validate_variant_value_columns(data, variant_value_columns, label="variant value column")
    if mutation_filters_active:
        mutation_filter_mask = _filter_mask(
            data,
            isin_filters=mutation_isin_filters,
            greater_than_filters=mutation_greater_than_filters,
            less_than_filters=mutation_less_than_filters,
            label="filter_mutations",
        )
        if not bool(mutation_filter_mask.any()):
            raise ValueError("Mutation filters matched zero mutation rows.")
        data = data[mutation_filter_mask].copy()
    if metadata is None and metadata_cols is not None:
        metadata = _derive_metadata_from_data(data, metadata_sample_col, metadata_cols)
    metadata = _validate_metadata(metadata, metadata_sample_col)
    pathway_df, _pathway_col = _validate_pathway(pathway, pathway_gene_col, gene_col)

    if sample_filters_active:
        (
            metadata_isin_filters,
            metadata_greater_than_filters,
            metadata_less_than_filters,
            data_isin_filters,
            data_greater_than_filters,
            data_less_than_filters,
        ) = _split_sample_filters(
            data,
            metadata,
            isin_filters=sample_isin_filters,
            greater_than_filters=sample_greater_than_filters,
            less_than_filters=sample_less_than_filters,
        )
        passing_sample_ids: Optional[set[str]] = None
        metadata_filters_active = bool(
            metadata_isin_filters or metadata_greater_than_filters or metadata_less_than_filters
        )
        if metadata_filters_active:
            if metadata is None:
                raise ValueError("metadata must be supplied when metadata sample filters are selected.")
            metadata_filter_mask = _filter_mask(
                metadata,
                isin_filters=metadata_isin_filters,
                greater_than_filters=metadata_greater_than_filters,
                less_than_filters=metadata_less_than_filters,
                label="filter_samples",
            )
            passing_sample_ids = set(metadata.loc[metadata_filter_mask, metadata_sample_col].astype(str))
        data_filters_active = bool(data_isin_filters or data_greater_than_filters or data_less_than_filters)
        if data_filters_active:
            data_filter_mask = _filter_mask(
                data,
                isin_filters=data_isin_filters,
                greater_than_filters=data_greater_than_filters,
                less_than_filters=data_less_than_filters,
                label="filter_samples",
            )
            data_sample_ids = set(data.loc[data_filter_mask, sample_col].astype(str))
            passing_sample_ids = (
                data_sample_ids
                if passing_sample_ids is None
                else passing_sample_ids & data_sample_ids
            )
        if not passing_sample_ids:
            raise ValueError("Sample filters matched zero samples.")
        data = data[data[sample_col].isin(passing_sample_ids)].copy()
        if metadata is not None:
            metadata = metadata[metadata[metadata_sample_col].isin(passing_sample_ids)].copy()

    if metadata is not None and metadata_require_mutations:
        metadata = metadata[metadata[metadata_sample_col].isin(pd.unique(data[sample_col]))].copy()

    genes = _get_genes_for_oncoplot(
        data=data,
        gene_col=gene_col,
        sample_col=sample_col,
        top_n=top_n,
        include_genes=include_genes,
        ignore_genes=ignore_genes,
        return_extra_genes_if_tied=return_extra_genes_if_tied,
        verbose=verbose,
    )
    if pathway_df is not None:
        genes = rank_genes_by_pathway(
            pathway_df,
            gene_ranks=genes,
            pathway_ranks=unique_preserve_order(pathway_df["Pathway"].astype(str)),
        )

    selected_rows = data[data[gene_col].isin(genes)].copy()
    rank_values = list(range(len(genes), 0, -1))
    if selected_rows.empty:
        sample_scores = pd.Series(dtype="int64")
        samples_with_selected_mutations: List[str] = []
        tiles = pd.DataFrame(
            columns=["Sample", "Gene", "MutationType", "MutationCount", "VariantValue", "Tooltip"]
            + [str(spec["key"]) for spec in variant_value_specs]
        )
    else:
        sample_scores = selected_rows.groupby(sample_col)[gene_col].apply(
            lambda values: score_sample_by_gene_rank(values.astype(str), genes, rank_values)
        )
        samples_with_selected_mutations = (
            sample_scores.sort_values(ascending=False, kind="mergesort").index.astype(str).tolist()
        )
        tiles = _collapse_mutations(
            selected_rows,
            gene_col,
            sample_col,
            mutation_type_col,
            tooltip_col,
            variant_value_specs,
            legacy_variant_value_key,
        )

    samples_with_any_mutations = unique_preserve_order(data[sample_col].astype(str))
    samples_with_metadata = (
        unique_preserve_order(metadata[metadata_sample_col].astype(str)) if metadata is not None else []
    )
    samples_with_custom_tmb = (
        _custom_tmb_sample_ids(tmb_data, sample_col) if prepare_tmb and tmb_data is not None else []
    )
    if mutation_filters_active or sample_filters_active:
        eligible_custom_tmb_samples = set(samples_with_any_mutations) | set(samples_with_metadata)
        samples_with_custom_tmb = [
            sample for sample in samples_with_custom_tmb if sample in eligible_custom_tmb_samples
        ]
    all_sample_ids = unique_preserve_order(
        list(samples_with_selected_mutations)
        + list(samples_with_any_mutations)
        + list(samples_with_metadata)
        + list(samples_with_custom_tmb)
    )
    samples_to_show = list(all_sample_ids if show_all_samples else samples_with_selected_mutations)

    if metadata_sort_cols is not None:
        if metadata is None:
            raise ValueError("metadata must be supplied to sort by clinical annotations.")
        sort_cols = list(metadata_sort_cols)
        desc_values = metadata_sort_desc if isinstance(metadata_sort_desc, Sequence) and not isinstance(metadata_sort_desc, str) else [metadata_sort_desc]
        by_values = metadata_sort_by if isinstance(metadata_sort_by, Sequence) and not isinstance(metadata_sort_by, str) else [metadata_sort_by]
        if len(desc_values) == 1:
            desc_values = list(desc_values) * len(sort_cols)
        if len(by_values) == 1:
            by_values = list(by_values) * len(sort_cols)
        if len(desc_values) != len(sort_cols) or len(by_values) != len(sort_cols):
            raise ValueError("metadata_sort_desc and metadata_sort_by must have length 1 or match metadata_sort_cols.")
        sample_order = _metadata_sort_order(
            metadata,
            metadata_sample_col,
            samples_to_show,
            sort_cols,
            [bool(value) for value in desc_values],
            [str(value) for value in by_values],
        )

    if sample_order is not None:
        samples_to_show = reorder_by_priority(samples_to_show, [str(sample) for sample in sample_order])

    tiles = tiles[tiles["Sample"].isin(samples_to_show)].copy()
    tiles["Sample"] = pd.Categorical(tiles["Sample"], categories=samples_to_show, ordered=True)
    tiles["Gene"] = pd.Categorical(tiles["Gene"], categories=genes, ordered=True)
    tiles = tiles.sort_values(["Gene", "Sample"], kind="mergesort")
    if mutation_type_col is None:
        mutation_type_levels: List[str] = []
        mutation_type_order_source = None
    else:
        mutation_type_levels, mutation_type_order_source = _resolve_category_levels(
            tiles["MutationType"] if not tiles.empty else [],
            explicit_order=mutation_type_explicit_order,
            categorical_order=mutation_type_categorical_order,
        )

    pathway_by_gene, pathway_groups = _pathway_summary(genes, pathway_df)
    if pathway_by_gene is not None and not tiles.empty:
        tiles["Pathway"] = tiles["Gene"].astype(str).map(pathway_by_gene).fillna("Other")

    metadata_out = None
    metadata_cols_out = None
    if metadata is not None:
        metadata_out = metadata[metadata[metadata_sample_col].isin(samples_to_show)].copy()
        metadata_out = metadata_out.rename(columns={metadata_sample_col: "Sample"})
        metadata_out["Sample"] = pd.Categorical(
            metadata_out["Sample"].astype(str), categories=samples_to_show, ordered=True
        )
        metadata_out = metadata_out.sort_values("Sample", kind="mergesort")
        if metadata_cols is None:
            metadata_cols_out = [column for column in metadata_out.columns if column != "Sample"]
        else:
            metadata_cols_out = _metadata_columns_list(metadata_cols)
            check_valid_dataframe_columns(metadata_out, metadata_cols_out)

    if total_samples == "any_mutations":
        n_total_samples = len(samples_with_any_mutations)
    elif total_samples == "all":
        n_total_samples = len(all_sample_ids)
    else:
        n_total_samples = len(pd.unique(tiles["Sample"].astype(str))) if not tiles.empty else 0

    tmb, tmb_sample_col, tmb_value_col, tmb_type_col, tmb_render_stacked = (None, None, None, None, False)
    tmb_type_category_order = None
    tmb_type_levels: List[str] = []
    tmb_type_order_source = None
    if prepare_tmb:
        (
            tmb,
            tmb_sample_col,
            tmb_value_col,
            tmb_type_col,
            tmb_render_stacked,
            tmb_type_category_order,
        ) = _prepare_tmb_data(
            data=data,
            samples_to_show=samples_to_show,
            sample_col=sample_col,
            mutation_type_col=mutation_type_col,
            tmb_data=tmb_data,
        )
        if tmb_data is None:
            tmb_type_category_order = mutation_type_categorical_order
        if tmb is not None and tmb_type_col is not None and not tmb[tmb_type_col].isna().all():
            tmb_type_levels, tmb_type_order_source = _resolve_category_levels(
                tmb[tmb_type_col],
                explicit_order=tmb_type_explicit_order,
                categorical_order=tmb_type_category_order,
            )
    tmb_totals, tmb_type_counts = _summarise_tmb(
        tmb,
        tmb_sample_col,
        tmb_value_col,
        tmb_type_col,
        samples_to_show,
        tmb_type_levels,
    )
    variant_value_min = None
    variant_value_max = None
    if variant_value_col is not None and not tiles.empty:
        variant_values = tiles["VariantValue"].astype(float).dropna()
        if not variant_values.empty:
            variant_value_min = float(variant_values.min())
            variant_value_max = float(variant_values.max())
    main_grid_rows_out = _build_main_grid_rows(
        genes,
        main_grid_tracks,
        tiles,
        variant_value_scale,
    )
    main_grid_tiles_out = _build_main_grid_tiles(
        tiles,
        main_grid_tracks,
        main_grid_rows_out,
        include_variant_summaries=tooltip_was_generated,
    )

    return PreparedOncoplotData(
        tiles=tiles.reset_index(drop=True),
        samples=samples_to_show,
        genes=genes,
        total_samples=n_total_samples,
        mutation_type_col=mutation_type_col,
        metadata=metadata_out,
        metadata_cols=metadata_cols_out,
        pathway=pathway_df,
        tmb=tmb,
        tmb_sample_col=tmb_sample_col,
        tmb_value_col=tmb_value_col,
        tmb_type_col=tmb_type_col,
        tmb_render_stacked=tmb_render_stacked,
        tmb_is_custom=prepare_tmb and tmb_data is not None,
        pathway_by_gene=pathway_by_gene,
        pathway_groups=pathway_groups,
        mutation_counts=_summarise_mutation_counts(tiles),
        tmb_totals=tmb_totals,
        tmb_type_counts=tmb_type_counts,
        metadata_tracks=_summarise_metadata_tracks(
            metadata_out,
            metadata_cols_out,
            samples_to_show,
            metadata_category_orders,
        ),
        mutation_type_levels=mutation_type_levels,
        mutation_type_order_source=mutation_type_order_source,
        tmb_type_levels=tmb_type_levels,
        tmb_type_order_source=tmb_type_order_source,
        variant_value_col=variant_value_col,
        variant_value_agg=variant_value_agg if variant_value_col is not None else None,
        variant_value_min=variant_value_min,
        variant_value_max=variant_value_max,
        variant_value_cols=variant_value_cols,
        variant_value_missing=variant_value_missing,
        variant_value_scale=variant_value_scale,
        main_grid_rows=main_grid_rows_out,
        main_grid_tiles=main_grid_tiles_out,
        main_grid_mode=main_grid_mode,
    )
