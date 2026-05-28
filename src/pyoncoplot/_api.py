"""Public plotting API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pandas as pd

from ._data import prepare_oncoplot_data
from ._io import load_oncoplot_params, materialize_table_params
from ._matplotlib import render_matplotlib_oncoplot
from ._options import OncoplotOptions, coerce_options
from ._palette import assert_palette_is_sensible, get_sensible_default_palette
from ._params import merge_params
from ._plotly import render_plotly_oncoplot
from ._result import OncoplotResult


ONCOPLOT_PARAM_KEYS = {
    "data",
    "gene_col",
    "sample_col",
    "mutation_type_col",
    "tooltip_col",
    "include_genes",
    "ignore_genes",
    "top_n",
    "return_extra_genes_if_tied",
    "draw_gene_bar",
    "draw_tmb_bar",
    "copy_on_click",
    "palette",
    "metadata",
    "metadata_palette",
    "metadata_sample_col",
    "metadata_cols",
    "metadata_require_mutations",
    "filter_samples_by_isin_lists",
    "filter_samples_by_greater_than",
    "filter_samples_by_less_than",
    "filter_mutations_by_isin_lists",
    "filter_mutations_by_greater_than",
    "filter_mutations_by_less_than",
    "pathway",
    "pathway_gene_col",
    "show_all_samples",
    "total_samples",
    "sample_order",
    "metadata_sort_cols",
    "metadata_sort_desc",
    "metadata_sort_by",
    "mutation_type_order",
    "metadata_category_orders",
    "tmb_type_order",
    "tmb_data",
    "tmb_palette",
    "variant_value_col",
    "variant_value_cols",
    "variant_value_agg",
    "variant_value_missing",
    "variant_value_palette",
    "variant_value_scale",
    "main_grid_rows",
    "gene_name_x_offset",
    "backend",
    "interactive",
    "options",
    "verbose",
}

ONCOPLOT_DEFAULTS: dict[str, Any] = {
    "mutation_type_col": None,
    "tooltip_col": None,
    "include_genes": None,
    "ignore_genes": None,
    "top_n": 10,
    "return_extra_genes_if_tied": False,
    "draw_gene_bar": False,
    "draw_tmb_bar": False,
    "copy_on_click": "sample",
    "palette": None,
    "metadata": None,
    "metadata_palette": None,
    "metadata_sample_col": None,
    "metadata_cols": None,
    "metadata_require_mutations": True,
    "filter_samples_by_isin_lists": None,
    "filter_samples_by_greater_than": None,
    "filter_samples_by_less_than": None,
    "filter_mutations_by_isin_lists": None,
    "filter_mutations_by_greater_than": None,
    "filter_mutations_by_less_than": None,
    "pathway": None,
    "pathway_gene_col": None,
    "show_all_samples": False,
    "total_samples": "any_mutations",
    "sample_order": None,
    "metadata_sort_cols": None,
    "metadata_sort_desc": True,
    "metadata_sort_by": "frequency",
    "mutation_type_order": None,
    "metadata_category_orders": None,
    "tmb_type_order": None,
    "tmb_data": None,
    "tmb_palette": None,
    "variant_value_col": None,
    "variant_value_cols": None,
    "variant_value_agg": "max",
    "variant_value_missing": "blank",
    "variant_value_palette": "viridis",
    "variant_value_scale": "per_column",
    "main_grid_rows": None,
    "gene_name_x_offset": None,
    "backend": "plotly",
    "interactive": None,
    "options": None,
    "verbose": False,
}


def _palette_for_data(
    prepared,
    palette: Optional[Mapping[str, str]],
    tmb_palette: Optional[Mapping[str, str]],
    options: OncoplotOptions,
) -> tuple[Mapping[str, str], Optional[Mapping[str, str]]]:
    mutation_types = prepared.mutation_type_levels or (
        prepared.tiles["MutationType"] if not prepared.tiles.empty else []
    )
    if prepared.mutation_type_col is None:
        mutation_palette = {}
    elif palette is None:
        mutation_palette = get_sensible_default_palette(mutation_types) or {}
        if "Multi_Hit" in mutation_palette:
            mutation_palette = dict(mutation_palette)
            mutation_palette["Multi_Hit"] = options.multi_hit_color
    else:
        mutation_palette = assert_palette_is_sensible(palette, mutation_types)

    render_stacked_tmb = (
        prepared.tmb is not None
        and prepared.tmb_type_col is not None
        and prepared.tmb_render_stacked
        and not options.log10_transform_tmb
    )
    if render_stacked_tmb:
        tmb_terms = prepared.tmb_type_levels or prepared.tmb[prepared.tmb_type_col]
        source_palette = mutation_palette if tmb_palette is None else tmb_palette
        try:
            return mutation_palette, assert_palette_is_sensible(source_palette, tmb_terms)
        except ValueError as exc:
            raise ValueError(
                "tmb_palette must define colors for every custom stacked TMB category, "
                "or those categories must be covered by the mutation palette fallback."
            ) from exc
    return mutation_palette, None


def _load_supplied_oncoplot_params(
    params: Optional[Union[Mapping[str, Any], str, os.PathLike]],
    *,
    params_key: Optional[str],
) -> Any:
    if params_key is not None and params is None:
        raise TypeError("params_key can only be used when params is a YAML config path.")
    if params_key is not None and isinstance(params, Mapping):
        raise TypeError("params_key can only be used when params is a YAML config path.")
    if isinstance(params, (str, os.PathLike)):
        return load_oncoplot_params(params, key=params_key)
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return materialize_table_params(params, base_dir=Path.cwd())
    return params


def merge_oncoplot_params(
    params: Optional[Union[Mapping[str, Any], str, os.PathLike]] = None,
    *,
    params_key: Optional[str] = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Merge reusable oncoplot params with explicit overrides.

    Use this helper when you need an unpacked keyword dictionary, such as
    ``oncoplot(**merge_oncoplot_params(params, options=options))``. Python
    raises before ``oncoplot`` runs for calls like ``oncoplot(options=...,
    **params)`` when ``params`` already contains ``"options"``.
    """

    supplied_params = _load_supplied_oncoplot_params(params, params_key=params_key)
    supplied = merge_params(supplied_params, allowed_keys=ONCOPLOT_PARAM_KEYS, context="oncoplot", **overrides)
    return materialize_table_params(supplied, base_dir=Path.cwd())


def oncoplot(
    data: Optional[Union[pd.DataFrame, str, os.PathLike]] = None,
    *,
    params: Optional[Union[Mapping[str, Any], str, os.PathLike]] = None,
    params_key: Optional[str] = None,
    **kwargs: Any,
) -> OncoplotResult:
    """Create a Pythonic oncoplot from mutation-level cohort data.

    Parameters are intentionally Pythonic rather than R-compatible. Use
    `gene_col`, `sample_col`, `mutation_type_col`, `top_n`, and friends.
    """

    supplied_params = _load_supplied_oncoplot_params(params, params_key=params_key)
    supplied = merge_params(supplied_params, allowed_keys=ONCOPLOT_PARAM_KEYS, context="oncoplot", **kwargs)
    supplied = materialize_table_params(supplied, base_dir=Path.cwd())
    merged = {**ONCOPLOT_DEFAULTS, **supplied}
    if data is None:
        data = merged.pop("data", None)
    else:
        merged.pop("data", None)
    if data is None:
        raise TypeError("oncoplot requires mutation data as the first argument or params['data'].")
    data = materialize_table_params({"data": data}, base_dir=Path.cwd())["data"]
    if "gene_col" not in merged or "sample_col" not in merged:
        raise TypeError("oncoplot requires 'gene_col' and 'sample_col'.")

    copy_on_click = merged["copy_on_click"]
    backend = merged["backend"]
    interactive = merged["interactive"]
    if copy_on_click not in {"sample", "gene", "tooltip", "mutation_type", "nothing"}:
        raise ValueError("copy_on_click must be one of: sample, gene, tooltip, mutation_type, nothing.")
    if interactive is not None:
        backend = "plotly" if interactive else "matplotlib"
    if backend not in {"plotly", "matplotlib"}:
        raise ValueError("backend must be either 'plotly' or 'matplotlib'.")
    options = coerce_options(merged["options"])
    if merged["gene_name_x_offset"] is not None:
        try:
            gene_name_x_offset = float(merged["gene_name_x_offset"])
        except (TypeError, ValueError) as exc:
            raise ValueError("gene_name_x_offset must be numeric when supplied.") from exc
        if gene_name_x_offset < 0:
            raise ValueError("gene_name_x_offset must be >= 0.")
        options.gene_name_x_offset = gene_name_x_offset

    prepared = prepare_oncoplot_data(
        data,
        gene_col=merged["gene_col"],
        sample_col=merged["sample_col"],
        mutation_type_col=merged["mutation_type_col"],
        tooltip_col=merged["tooltip_col"],
        include_genes=merged["include_genes"],
        ignore_genes=merged["ignore_genes"],
        top_n=merged["top_n"],
        return_extra_genes_if_tied=merged["return_extra_genes_if_tied"],
        metadata=merged["metadata"],
        metadata_sample_col=merged["metadata_sample_col"],
        metadata_cols=merged["metadata_cols"],
        metadata_require_mutations=merged["metadata_require_mutations"],
        filter_samples_by_isin_lists=merged["filter_samples_by_isin_lists"],
        filter_samples_by_greater_than=merged["filter_samples_by_greater_than"],
        filter_samples_by_less_than=merged["filter_samples_by_less_than"],
        filter_mutations_by_isin_lists=merged["filter_mutations_by_isin_lists"],
        filter_mutations_by_greater_than=merged["filter_mutations_by_greater_than"],
        filter_mutations_by_less_than=merged["filter_mutations_by_less_than"],
        pathway=merged["pathway"],
        pathway_gene_col=merged["pathway_gene_col"],
        show_all_samples=merged["show_all_samples"],
        total_samples=merged["total_samples"],
        sample_order=merged["sample_order"],
        metadata_sort_cols=merged["metadata_sort_cols"],
        metadata_sort_desc=merged["metadata_sort_desc"],
        metadata_sort_by=merged["metadata_sort_by"],
        mutation_type_order=merged["mutation_type_order"],
        metadata_category_orders=merged["metadata_category_orders"],
        tmb_type_order=merged["tmb_type_order"],
        tmb_data=merged["tmb_data"],
        prepare_tmb=merged["draw_tmb_bar"],
        variant_value_col=merged["variant_value_col"],
        variant_value_cols=merged["variant_value_cols"],
        variant_value_agg=merged["variant_value_agg"],
        variant_value_missing=merged["variant_value_missing"],
        variant_value_scale=merged["variant_value_scale"],
        main_grid_rows=merged["main_grid_rows"],
        verbose=merged["verbose"],
    )
    mutation_palette, validated_tmb_palette = _palette_for_data(
        prepared,
        merged["palette"],
        merged["tmb_palette"],
        options,
    )
    palette_for_render = dict(mutation_palette)
    tmb_palette_for_render = dict(validated_tmb_palette) if validated_tmb_palette is not None else None

    if backend == "plotly":
        figure = render_plotly_oncoplot(
            prepared,
            palette=palette_for_render,
            tmb_palette=tmb_palette_for_render,
            metadata_palette=merged["metadata_palette"],
            variant_value_palette=merged["variant_value_palette"],
            options=options,
            draw_gene_bar=merged["draw_gene_bar"],
            draw_tmb_bar=merged["draw_tmb_bar"],
            copy_on_click=copy_on_click,
        )
    else:
        figure = render_matplotlib_oncoplot(
            prepared,
            palette=palette_for_render,
            tmb_palette=tmb_palette_for_render,
            metadata_palette=merged["metadata_palette"],
            variant_value_palette=merged["variant_value_palette"],
            options=options,
            draw_gene_bar=merged["draw_gene_bar"],
            draw_tmb_bar=merged["draw_tmb_bar"],
        )

    return OncoplotResult(
        figure=figure,
        backend=backend,
        prepared_data=prepared,
        copy_on_click=copy_on_click,
    )
