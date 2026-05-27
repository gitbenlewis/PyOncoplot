"""Pythonic oncoplots for mutation-level cohort data."""

from ._api import oncoplot
from ._data import (
    PreparedOncoplotData,
    check_valid_dataframe_columns,
    identify_top_genes,
    prepare_oncoplot_data,
    rank_genes_by_pathway,
    score_sample_by_gene_rank,
)
from ._io import load_oncoplot_params
from ._options import OncoplotOptions
from ._palette import assert_palette_is_sensible, get_sensible_default_palette
from ._result import OncoplotResult
from ._utils import prettify
from .palettes import (
    Iridescent,
    cividis_greyzero,
    default_20,
    default_28,
    default_102,
    godsnot_102,
    inferno_greyzero,
    magma_greyzero,
    make_greyzero_colormap,
    plasma_greyzero,
    tol_colors,
    turbo_greyzero,
    vega_10,
    vega_10_scanpy,
    vega_20,
    vega_20_scanpy,
    viridis_greyzero,
    zeileis_28,
)

__all__ = [
    "OncoplotOptions",
    "OncoplotResult",
    "PreparedOncoplotData",
    "Iridescent",
    "assert_palette_is_sensible",
    "check_valid_dataframe_columns",
    "cividis_greyzero",
    "default_20",
    "default_28",
    "default_102",
    "get_sensible_default_palette",
    "godsnot_102",
    "inferno_greyzero",
    "identify_top_genes",
    "load_oncoplot_params",
    "magma_greyzero",
    "make_greyzero_colormap",
    "oncoplot",
    "plasma_greyzero",
    "prepare_oncoplot_data",
    "prettify",
    "rank_genes_by_pathway",
    "score_sample_by_gene_rank",
    "tol_colors",
    "turbo_greyzero",
    "vega_10",
    "vega_10_scanpy",
    "vega_20",
    "vega_20_scanpy",
    "viridis_greyzero",
    "zeileis_28",
]
