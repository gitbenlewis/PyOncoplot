"""Public plotting API."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from ._data import prepare_oncoplot_data
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
    "pathway",
    "pathway_gene_col",
    "show_all_samples",
    "total_samples",
    "sample_order",
    "metadata_sort_cols",
    "metadata_sort_desc",
    "metadata_sort_by",
    "tmb_data",
    "tmb_palette",
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
    "pathway": None,
    "pathway_gene_col": None,
    "show_all_samples": False,
    "total_samples": "any_mutations",
    "sample_order": None,
    "metadata_sort_cols": None,
    "metadata_sort_desc": True,
    "metadata_sort_by": "frequency",
    "tmb_data": None,
    "tmb_palette": None,
    "backend": "plotly",
    "interactive": None,
    "options": None,
    "verbose": False,
}


def _palette_for_data(
    prepared,
    palette: Optional[Mapping[str, str]],
    tmb_palette: Optional[Mapping[str, str]],
) -> tuple[Mapping[str, str], Optional[Mapping[str, str]]]:
    mutation_types = prepared.tiles["MutationType"] if not prepared.tiles.empty else []
    if prepared.mutation_type_col is None:
        mutation_palette = {}
    elif palette is None:
        mutation_palette = get_sensible_default_palette(mutation_types) or {}
    else:
        mutation_palette = assert_palette_is_sensible(palette, mutation_types)

    if prepared.tmb is not None and prepared.tmb_type_col is not None and prepared.tmb_render_stacked:
        tmb_terms = prepared.tmb[prepared.tmb_type_col]
        source_palette = mutation_palette if tmb_palette is None else tmb_palette
        try:
            return mutation_palette, assert_palette_is_sensible(source_palette, tmb_terms)
        except ValueError as exc:
            raise ValueError(
                "tmb_palette must define colors for every custom stacked TMB category, "
                "or those categories must be covered by the mutation palette fallback."
            ) from exc
    if tmb_palette is None:
        return mutation_palette, None
    return mutation_palette, tmb_palette


def oncoplot(
    data: Optional[pd.DataFrame] = None,
    *,
    params: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> OncoplotResult:
    """Create a Pythonic oncoplot from mutation-level cohort data.

    Parameters are intentionally Pythonic rather than R-compatible. Use
    `gene_col`, `sample_col`, `mutation_type_col`, `top_n`, and friends.
    """

    supplied = merge_params(params, allowed_keys=ONCOPLOT_PARAM_KEYS, context="oncoplot", **kwargs)
    merged = {**ONCOPLOT_DEFAULTS, **supplied}
    if data is None:
        data = merged.pop("data", None)
    else:
        merged.pop("data", None)
    if data is None:
        raise TypeError("oncoplot requires mutation data as the first argument or params['data'].")
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
        pathway=merged["pathway"],
        pathway_gene_col=merged["pathway_gene_col"],
        show_all_samples=merged["show_all_samples"],
        total_samples=merged["total_samples"],
        sample_order=merged["sample_order"],
        metadata_sort_cols=merged["metadata_sort_cols"],
        metadata_sort_desc=merged["metadata_sort_desc"],
        metadata_sort_by=merged["metadata_sort_by"],
        tmb_data=merged["tmb_data"],
        prepare_tmb=merged["draw_tmb_bar"],
        verbose=merged["verbose"],
    )
    mutation_palette, validated_tmb_palette = _palette_for_data(prepared, merged["palette"], merged["tmb_palette"])
    palette_for_render = dict(mutation_palette)
    if validated_tmb_palette:
        palette_for_render.update(validated_tmb_palette)

    if backend == "plotly":
        figure = render_plotly_oncoplot(
            prepared,
            palette=palette_for_render,
            metadata_palette=merged["metadata_palette"],
            options=options,
            draw_gene_bar=merged["draw_gene_bar"],
            draw_tmb_bar=merged["draw_tmb_bar"],
            copy_on_click=copy_on_click,
        )
    else:
        figure = render_matplotlib_oncoplot(
            prepared,
            palette=palette_for_render,
            metadata_palette=merged["metadata_palette"],
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
