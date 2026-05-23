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


def test_custom_tmb_sample_column_order_renders_in_both_backends():
    tmb = pd.DataFrame(
        {
            "mutations": [5, 7, 11],
            "sample": ["S1", "S2", "S3"],
        }
    )
    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        backend="plotly",
        options=OncoplotOptions(log10_transform_tmb=False),
    )
    expected = {row["sample"]: float(row["mutations"]) for _index, row in tmb.iterrows()}
    tmb_traces = [
        trace for trace in plotly_result.figure.data if getattr(trace, "type", "") == "bar" and trace.orientation is None
    ]
    assert tmb_traces
    assert list(tmb_traces[0].x) == plotly_result.prepared_data.samples
    assert list(tmb_traces[0].y) == [expected[sample] for sample in plotly_result.prepared_data.samples]

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        backend="matplotlib",
        options=OncoplotOptions(log10_transform_tmb=False),
    )
    tmb_axis = matplotlib_result.figure.axes[0]
    heights = [patch.get_height() for patch in tmb_axis.patches]
    assert heights == [expected[sample] for sample in matplotlib_result.prepared_data.samples]


def test_sample_id_position_and_axis_label_fonts_are_applied():
    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="matplotlib",
        options=OncoplotOptions(
            show_sample_ids=True,
            sample_id_position="top",
            show_x_label=True,
            show_y_label=True,
            font_size_x_label=31,
            font_size_y_label=29,
        ),
    )
    main_axis = matplotlib_result.figure.axes[0]
    assert main_axis.xaxis.get_ticks_position() == "top"
    assert main_axis.xaxis.get_label_position() == "top"
    assert main_axis.xaxis.label.get_size() == 31
    assert main_axis.yaxis.label.get_size() == 29

    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="plotly",
        options=OncoplotOptions(show_sample_ids=True, sample_id_position="top", sample_id_angle=45),
    )
    assert plotly_result.figure.layout.xaxis.side == "top"
    assert plotly_result.figure.layout.xaxis.tickangle == 45


def test_plotly_gene_bar_hover_and_visible_labels_are_separate():
    result_without_labels = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        backend="plotly",
    )
    bar_traces = [
        trace
        for trace in result_without_labels.figure.data
        if getattr(trace, "type", "") == "bar" and trace.orientation == "h"
    ]
    assert bar_traces
    assert bar_traces[0].text is None
    assert "Total Samples Mutated" in bar_traces[0].hovertext[0]
    assert "of all mutations in this gene" in bar_traces[0].hovertext[0]

    result_with_labels = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        backend="plotly",
        options=OncoplotOptions(show_gene_bar_labels=True),
    )
    label_traces = [
        trace
        for trace in result_with_labels.figure.data
        if getattr(trace, "type", "") == "scatter" and getattr(trace, "mode", "") == "text"
    ]
    assert label_traces
    assert all("%" in text for text in label_traces[0].text)
    assert all("Total Samples Mutated" not in text for text in label_traces[0].text)


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
