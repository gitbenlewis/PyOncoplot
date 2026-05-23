import pandas as pd
import pytest

from pyoncoplot import OncoplotOptions, oncoplot, prepare_oncoplot_data
from pyoncoplot._matplotlib import render_matplotlib_oncoplot
from pyoncoplot._params import merge_params
from pyoncoplot._plotly import render_plotly_oncoplot


def small_df():
    return pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "gene": ["TP53", "EGFR", "PTEN"],
            "type": ["Missense_Mutation", "Frame_Shift_Del", "Nonsense_Mutation"],
        }
    )


def test_merge_params_accepts_params_and_kwargs_with_kwargs_precedence():
    assert merge_params({"a": 1, "b": 2}, allowed_keys={"a", "b"}, b=3) == {"a": 1, "b": 3}
    assert merge_params(None, allowed_keys={"a"}, a=4) == {"a": 4}


def test_merge_params_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown demo parameter"):
        merge_params({"known": 1, "extra": 2}, allowed_keys={"known"}, context="demo")


def test_oncoplot_params_matches_explicit_kwargs_and_allows_override():
    explicit = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        top_n=2,
        backend="matplotlib",
        options=OncoplotOptions(width=700, height=450),
    )
    from_params = oncoplot(
        params={
            "data": small_df(),
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "top_n": 1,
            "backend": "matplotlib",
            "options": {"width": 700, "height": 450},
        },
        top_n=2,
    )
    assert from_params.prepared_data.genes == explicit.prepared_data.genes
    assert from_params.prepared_data.samples == explicit.prepared_data.samples
    assert from_params.backend == "matplotlib"


def test_renderers_accept_params_dictionaries():
    prepared = prepare_oncoplot_data(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        prepare_tmb=True,
    )
    palette = {
        "Missense_Mutation": "#1f77b4",
        "Frame_Shift_Del": "#ff7f0e",
        "Nonsense_Mutation": "#2ca02c",
    }
    matplotlib_figure = render_matplotlib_oncoplot(
        params={
            "prepared": prepared,
            "palette": palette,
            "options": {"width": 700, "height": 450},
            "draw_gene_bar": True,
            "draw_tmb_bar": True,
        }
    )
    plotly_figure = render_plotly_oncoplot(
        params={
            "prepared": prepared,
            "palette": palette,
            "options": {"width": 700, "height": 450},
            "draw_gene_bar": True,
            "draw_tmb_bar": True,
        }
    )
    assert len(matplotlib_figure.axes) >= 2
    assert plotly_figure.layout.width == 700
