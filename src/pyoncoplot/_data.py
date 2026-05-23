"""Data validation and transformation for oncoplots."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

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


def _is_dataframe(value: Any) -> bool:
    return isinstance(value, pd.DataFrame)


def _empty_or_missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: isinstance(value, str) and value == "")


def _as_str_series(series: pd.Series) -> pd.Series:
    return series.astype("string").astype(object)


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


def _collapse_mutations(
    data: pd.DataFrame,
    gene_col: str,
    sample_col: str,
    mutation_type_col: Optional[str],
    tooltip_col: str,
) -> pd.DataFrame:
    rows = []
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
            tooltip = f"<strong>{sample}</strong><br>{tooltip}"
        rows.append(
            {
                "Sample": str(sample),
                "Gene": str(gene),
                "MutationType": mutation_type,
                "MutationCount": mutation_count,
                "Tooltip": tooltip,
            }
        )
    return pd.DataFrame(rows, columns=["Sample", "Gene", "MutationType", "MutationCount", "Tooltip"])


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
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str], bool]:
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
    return out, sample_col, value_col, type_col, render_stacked


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
    return totals, type_counts


def _summarise_metadata_tracks(
    metadata: Optional[pd.DataFrame],
    metadata_cols: Optional[Sequence[str]],
    samples: Sequence[str],
) -> Optional[List[MetadataTrackInfo]]:
    if metadata is None or not metadata_cols:
        return None
    metadata_by_sample = metadata.set_index("Sample")
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
                    value_min=float(values.min(skipna=True)) if values.notna().any() else None,
                    value_max=float(values.max(skipna=True)) if values.notna().any() else None,
                    missing_samples=missing_samples,
                )
            )
            continue
        levels = [str(value) for value in pd.unique(values.dropna())]
        tracks.append(
            MetadataTrackInfo(
                column=str(column),
                kind="categorical",
                levels=levels,
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
    pathway: Optional[pd.DataFrame] = None,
    pathway_gene_col: Optional[str] = None,
    show_all_samples: bool = False,
    total_samples: str = "any_mutations",
    sample_order: Optional[Sequence[str]] = None,
    metadata_sort_cols: Optional[Sequence[str]] = None,
    metadata_sort_desc: Any = True,
    metadata_sort_by: Any = "frequency",
    tmb_data: Optional[pd.DataFrame] = None,
    prepare_tmb: bool = False,
    verbose: bool = False,
) -> PreparedOncoplotData:
    """Validate and transform mutation-level data into oncoplot-ready tables."""

    tooltip_col = tooltip_col or sample_col
    metadata_sample_col = metadata_sample_col or sample_col
    total_samples_options = {"any_mutations", "all", "oncoplot"}
    if total_samples not in total_samples_options:
        raise ValueError("total_samples must be one of: any_mutations, all, oncoplot.")
    if sample_order is not None and metadata_sort_cols is not None:
        raise ValueError("Please specify either sample_order or metadata_sort_cols, not both.")

    data = _validate_mutation_inputs(data, gene_col, sample_col, mutation_type_col, tooltip_col)
    metadata = _validate_metadata(metadata, metadata_sample_col)
    pathway_df, _pathway_col = _validate_pathway(pathway, pathway_gene_col, gene_col)

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
        tiles = pd.DataFrame(columns=["Sample", "Gene", "MutationType", "MutationCount", "Tooltip"])
    else:
        sample_scores = selected_rows.groupby(sample_col)[gene_col].apply(
            lambda values: score_sample_by_gene_rank(values.astype(str), genes, rank_values)
        )
        samples_with_selected_mutations = (
            sample_scores.sort_values(ascending=False, kind="mergesort").index.astype(str).tolist()
        )
        tiles = _collapse_mutations(selected_rows, gene_col, sample_col, mutation_type_col, tooltip_col)

    samples_with_any_mutations = unique_preserve_order(data[sample_col].astype(str))
    samples_with_metadata = (
        unique_preserve_order(metadata[metadata_sample_col].astype(str)) if metadata is not None else []
    )
    samples_with_custom_tmb = (
        _custom_tmb_sample_ids(tmb_data, sample_col) if prepare_tmb and tmb_data is not None else []
    )
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
            metadata_cols_out = list(metadata_cols)
            check_valid_dataframe_columns(metadata_out, metadata_cols_out)

    if total_samples == "any_mutations":
        n_total_samples = len(samples_with_any_mutations)
    elif total_samples == "all":
        n_total_samples = len(all_sample_ids)
    else:
        n_total_samples = len(pd.unique(tiles["Sample"].astype(str))) if not tiles.empty else 0

    tmb, tmb_sample_col, tmb_value_col, tmb_type_col, tmb_render_stacked = (None, None, None, None, False)
    if prepare_tmb:
        tmb, tmb_sample_col, tmb_value_col, tmb_type_col, tmb_render_stacked = _prepare_tmb_data(
            data=data,
            samples_to_show=samples_to_show,
            sample_col=sample_col,
            mutation_type_col=mutation_type_col,
            tmb_data=tmb_data,
        )
    tmb_totals, tmb_type_counts = _summarise_tmb(tmb, tmb_sample_col, tmb_value_col, tmb_type_col, samples_to_show)

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
        metadata_tracks=_summarise_metadata_tracks(metadata_out, metadata_cols_out, samples_to_show),
    )
