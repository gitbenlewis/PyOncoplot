from pathlib import Path

import pandas as pd
import pytest

from pyoncoplot import OncoplotOptions, oncoplot


def small_df():
    return pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S3"],
            "gene": ["TP53", "EGFR", "TP53", "PTEN"],
            "type": ["Missense_Mutation", "Frame_Shift_Del", "Nonsense_Mutation", "Splice_Site"],
        }
    )


def test_plotly_render_and_html_clipboard():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        draw_tmb_bar=True,
        backend="plotly",
        options=OncoplotOptions(width=700, height=450),
    )
    assert result.backend == "plotly"
    html = result.to_html(full_html=True)
    assert "navigator.clipboard.writeText" in html
    assert "applyLinkedSelection" in html
    assert "plotly" in html.lower()


def test_matplotlib_render_and_save(tmp_path):
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        draw_tmb_bar=True,
        backend="matplotlib",
        options=OncoplotOptions(width=700, height=450),
    )
    output = tmp_path / "oncoplot.png"
    result.save(str(output), dpi=90)
    assert output.exists()
    assert output.stat().st_size > 0


def test_matplotlib_gallery_features_smoke():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "group": ["A", "B", "A"],
            "score": [1, 5, 3],
        }
    )
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S2", "S3"],
            "type": ["Missense_Mutation", "Splice_Site", "Missense_Mutation", "Frame_Shift_Del", "Splice_Site"],
            "mutations": [6, 2, 3, 4, 7],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        draw_tmb_bar=True,
        tmb_data=tmb,
        metadata=metadata,
        metadata_cols=["group", "score"],
        metadata_palette={"group": {"A": "#00AA88", "B": "#CC5500"}},
        backend="matplotlib",
        options=OncoplotOptions(
            log10_transform_tmb=False,
            show_gene_bar_labels=True,
            metadata_numeric_plot_type="bar",
            mutation_legend_position="right",
            metadata_legend_position="right",
        ),
    )
    assert result.backend == "matplotlib"
    assert len(result.figure.axes) >= 4
    assert len(result.figure.legends) >= 2


def test_custom_stacked_tmb_requires_palette_coverage():
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": ["Subtype_A", "Subtype_B", "Subtype_A"],
            "mutations": [4, 2, 3],
        }
    )
    with pytest.raises(ValueError, match="tmb_palette"):
        oncoplot(
            small_df(),
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            draw_tmb_bar=True,
            tmb_data=tmb,
            backend="matplotlib",
        )


def test_plotly_tmb_hover_uses_real_totals_and_metadata_palette_legend():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": ["Subtype_A", "Subtype_B", "Subtype_A"],
            "mutations": [4, 2, 3],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette={"Subtype_A": "#111111", "Subtype_B": "#222222"},
        metadata=metadata,
        metadata_cols=["group"],
        metadata_palette={"group": {"A": "#00AA88", "B": "#CC5500"}},
        backend="plotly",
        options=OncoplotOptions(width=700, height=450),
    )
    tmb_traces = [trace for trace in result.figure.data if getattr(trace, "type", "") == "bar" and trace.orientation is None]
    assert tmb_traces
    assert tmb_traces[0].customdata[0]["tmb"] == 6.0
    assert any(getattr(trace, "name", "") == "group: A" for trace in result.figure.data)


def test_metadata_max_levels_validation_applies_to_renderers():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "C"]})
    with pytest.raises(ValueError, match="metadata_max_levels"):
        oncoplot(
            small_df(),
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            metadata=metadata,
            metadata_cols=["group"],
            backend="matplotlib",
            options=OncoplotOptions(metadata_max_levels=2),
        )


def test_matplotlib_pathway_strip_and_metadata_na_marker_render():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", None, "B"]})
    pathway = pd.DataFrame({"gene": ["TP53", "PTEN"], "pathway": ["Cell cycle", "PI3K"]})
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        pathway=pathway,
        backend="matplotlib",
        options=OncoplotOptions(pathway_text_angle=90),
    )
    texts = [text.get_text() for axis in result.figure.axes for text in axis.texts]
    assert "Cell cycle" in texts
    assert "PI3K" in texts
    assert "Other" in texts
    assert "!" in texts


def test_new_options_validate_boundaries():
    with pytest.raises(ValueError, match="metadata_max_levels"):
        OncoplotOptions(metadata_max_levels=0)
    with pytest.raises(ValueError, match="gene_bar_scale_n_breaks"):
        OncoplotOptions(gene_bar_scale_n_breaks=0)
