from pathlib import Path

import pandas as pd
import pytest
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, to_hex

from pyoncoplot import Iridescent, OncoplotOptions, oncoplot, tol_colors


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
            options=OncoplotOptions(log10_transform_tmb=False),
        )


def test_custom_stacked_tmb_log_scale_does_not_require_palette_coverage():
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": ["Subtype_A", "Subtype_B", "Subtype_A"],
            "mutations": [4, 2, 3],
        }
    )
    with pytest.warns(UserWarning, match="log10_transform_tmb=True"):
        result = oncoplot(
            small_df(),
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            draw_tmb_bar=True,
            tmb_data=tmb,
            backend="plotly",
        )
    assert result.prepared_data.tmb_type_col == "tmb_type"
    assert result.prepared_data.tmb_totals is not None
    assert result.prepared_data.tmb_totals.loc["S1"] == 6.0


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
    assert any(getattr(trace, "name", "") == "Group: A" for trace in result.figure.data)


def test_custom_stacked_tmb_legends_render_in_both_backends():
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": ["Subtype_A", "Subtype_B", "Subtype_A"],
            "mutations": [4, 2, 3],
        }
    )
    tmb_palette = {"Subtype_A": "#111111", "Subtype_B": "#222222"}
    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette=tmb_palette,
        backend="plotly",
        options=OncoplotOptions(log10_transform_tmb=False, mutation_legend_position="right"),
    )
    tmb_legend_traces = [
        trace for trace in plotly_result.figure.data if getattr(trace, "legendgroup", "") == "tmb"
    ]
    assert [trace.name for trace in tmb_legend_traces] == ["TMB: Subtype A", "TMB: Subtype B"]
    assert all(trace.showlegend is True for trace in tmb_legend_traces)

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette=tmb_palette,
        backend="matplotlib",
        options=OncoplotOptions(log10_transform_tmb=False, mutation_legend_position="right"),
    )
    legend_titles = [legend.get_title().get_text() for legend in matplotlib_result.figure.legends]
    legend_labels = [
        text.get_text() for legend in matplotlib_result.figure.legends for text in legend.get_texts()
    ]
    assert "TMB Type" in legend_titles
    assert "Subtype A" in legend_labels
    assert "Subtype B" in legend_labels


def test_tmb_legend_none_hides_tmb_but_metadata_can_remain():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": ["Subtype_A", "Subtype_B", "Subtype_A"],
            "mutations": [4, 2, 3],
        }
    )
    tmb_palette = {"Subtype_A": "#111111", "Subtype_B": "#222222"}
    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette=tmb_palette,
        metadata=metadata,
        metadata_cols=["group"],
        backend="plotly",
        options=OncoplotOptions(
            log10_transform_tmb=False,
            mutation_legend_position="none",
            show_metadata_legends=True,
        ),
    )
    tmb_traces = [trace for trace in plotly_result.figure.data if getattr(trace, "legendgroup", "") == "tmb"]
    metadata_legend_traces = [
        trace for trace in plotly_result.figure.data if str(getattr(trace, "legendgroup", "")).startswith("metadata:")
    ]
    assert tmb_traces
    assert all(trace.showlegend is False for trace in tmb_traces)
    assert metadata_legend_traces
    assert all(trace.showlegend is True for trace in metadata_legend_traces)

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette=tmb_palette,
        metadata=metadata,
        metadata_cols=["group"],
        backend="matplotlib",
        options=OncoplotOptions(
            log10_transform_tmb=False,
            mutation_legend_position="none",
            show_metadata_legends=True,
        ),
    )
    legend_titles = [legend.get_title().get_text() for legend in matplotlib_result.figure.legends]
    assert "TMB Type" not in legend_titles
    assert "Group" in legend_titles


def test_tmb_palette_does_not_recolor_main_mutation_tiles_when_keys_overlap():
    mutation_palette = {
        "Missense_Mutation": "#1F78B4",
        "Frame_Shift_Del": "#33A02C",
        "Nonsense_Mutation": "#E31A1C",
        "Splice_Site": "#6A3D9A",
    }
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "type": ["Missense_Mutation", "Subtype_B"],
            "mutations": [4, 3],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        palette=mutation_palette,
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette={"Missense_Mutation": "#00FF00", "Subtype_B": "#222222"},
        backend="plotly",
        options=OncoplotOptions(log10_transform_tmb=False),
    )
    main_trace = next(trace for trace in result.figure.data if getattr(trace, "name", "") == "Missense Mutation")
    tmb_trace = next(trace for trace in result.figure.data if getattr(trace, "name", "") == "TMB: Missense Mutation")
    assert main_trace.marker.color == "#1F78B4"
    assert tmb_trace.marker.color == "#00FF00"


def test_duplicate_tmb_categories_do_not_add_redundant_tmb_legend():
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "type": ["Missense_Mutation", "Frame_Shift_Del"],
            "mutations": [4, 3],
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
    tmb_traces = [
        trace for trace in plotly_result.figure.data if getattr(trace, "legendgroup", "") == "tmb"
    ]
    assert tmb_traces
    assert all(trace.showlegend is False for trace in tmb_traces)

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        backend="matplotlib",
        options=OncoplotOptions(log10_transform_tmb=False, mutation_legend_position="right"),
    )
    legend_titles = [legend.get_title().get_text() for legend in matplotlib_result.figure.legends]
    assert "Mutation Type" in legend_titles
    assert "TMB Type" not in legend_titles


def test_typed_tmb_log_scale_warns_in_both_backends():
    for backend in ("plotly", "matplotlib"):
        with pytest.warns(UserWarning, match="log10_transform_tmb=True disables stacked TMB rendering"):
            oncoplot(
                small_df(),
                gene_col="gene",
                sample_col="sample",
                mutation_type_col="type",
                draw_tmb_bar=True,
                backend=backend,
            )


def test_matplotlib_tmb_axis_label_and_scientific_formatting():
    tmb = pd.DataFrame({"sample": ["S1", "S2", "S3"], "score": [10, 100, 1000]})
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        backend="matplotlib",
        options=OncoplotOptions(
            log10_transform_tmb=False,
            scientific_tmb=True,
            show_tmb_y_label=True,
        ),
    )
    tmb_axis = result.figure.axes[0]
    result.figure.canvas.draw()
    assert tmb_axis.get_ylabel() == "score"
    assert any("e" in label.get_text() for label in tmb_axis.get_yticklabels() if label.get_text())

    log_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        backend="matplotlib",
        options=OncoplotOptions(show_tmb_y_label=True),
    )
    assert log_result.figure.axes[0].get_ylabel() == "log10 score"


def test_metadata_prettification_applies_to_both_backends_without_mutating_customdata():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "clinical_group": ["high_risk", "low_risk", "high_risk"],
        }
    )
    metadata_palette = {"clinical_group": {"high_risk": "#00AA88", "low_risk": "#CC5500"}}
    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["clinical_group"],
        metadata_palette=metadata_palette,
        backend="plotly",
        options=OncoplotOptions(metadata_position="top", mutation_legend_position="none"),
    )
    metadata_heatmap = next(
        trace
        for trace in plotly_result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "metadata"
    )
    assert list(metadata_heatmap.y) == ["Clinical Group"]
    assert metadata_heatmap.text[0][0] == "Clinical Group: High Risk"
    assert metadata_heatmap.customdata[0][0]["column"] == "clinical_group"
    assert metadata_heatmap.customdata[0][0]["value"] == "high_risk"
    assert any(getattr(trace, "name", "") == "Clinical Group: High Risk" for trace in plotly_result.figure.data)

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["clinical_group"],
        metadata_palette=metadata_palette,
        backend="matplotlib",
        options=OncoplotOptions(metadata_position="top", mutation_legend_position="none"),
    )
    metadata_axis = matplotlib_result.figure.axes[0]
    legend_titles = [legend.get_title().get_text() for legend in matplotlib_result.figure.legends]
    legend_labels = [
        text.get_text() for legend in matplotlib_result.figure.legends for text in legend.get_texts()
    ]
    assert [label.get_text() for label in metadata_axis.get_yticklabels()] == ["Clinical Group"]
    assert "Clinical Group" in legend_titles
    assert "High Risk" in legend_labels

    hidden_title_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["clinical_group"],
        metadata_palette=metadata_palette,
        backend="matplotlib",
        options=OncoplotOptions(
            metadata_position="top",
            mutation_legend_position="none",
            show_legend_titles=False,
        ),
    )
    assert all(legend.get_title().get_text() == "" for legend in hidden_title_result.figure.legends)


def test_matplotlib_metadata_margin_expands_for_long_row_labels():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "days_to_last_followup": [120, 340, 560],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["days_to_last_followup"],
        backend="matplotlib",
        options=OncoplotOptions(
            width=1080,
            height=720,
            font_size_metadata=8,
            metadata_position="bottom",
            metadata_numeric_plot_type="bar",
        ),
    )

    assert result.figure.subplotpars.left >= 0.19


def test_matplotlib_gene_bar_axis_moves_above_bottom_metadata():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "group": ["A", "B", "A"],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        draw_gene_bar=True,
        backend="matplotlib",
        options=OncoplotOptions(
            metadata_position="bottom",
            show_gene_bar_axis=True,
        ),
    )
    gene_bar_axis = result.figure.axes[1]

    assert gene_bar_axis.xaxis.get_ticks_position() == "top"
    assert gene_bar_axis.xaxis.get_label_position() == "top"


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


def test_plotly_mutation_legend_none_keeps_metadata_legend():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="plotly",
        options=OncoplotOptions(mutation_legend_position="none", show_metadata_legends=True),
    )
    mutation_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "customdata", None)
        and isinstance(trace.customdata[0], dict)
        and trace.customdata[0].get("role") == "main_tile"
    ]
    metadata_legend_traces = [
        trace for trace in result.figure.data if str(getattr(trace, "legendgroup", "")).startswith("metadata:")
    ]
    assert mutation_traces
    assert all(trace.showlegend is False for trace in mutation_traces)
    assert metadata_legend_traces
    assert all(trace.showlegend is True for trace in metadata_legend_traces)
    assert result.figure.layout.showlegend is True


def test_plotly_legend_positions_are_applied():
    bottom = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="plotly",
        options=OncoplotOptions(mutation_legend_position="bottom"),
    )
    assert bottom.figure.layout.legend.orientation == "h"
    assert bottom.figure.layout.legend.xanchor == "center"
    assert bottom.figure.layout.margin.b == 90

    right = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="plotly",
        options=OncoplotOptions(mutation_legend_position="right"),
    )
    assert right.figure.layout.legend.orientation == "v"
    assert right.figure.layout.legend.x > 1
    assert right.figure.layout.margin.r == 150


def test_plotly_font_sizes_tile_linewidth_and_gene_bar_nticks_are_applied():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        backend="plotly",
        options=OncoplotOptions(
            show_sample_ids=True,
            show_x_label=True,
            show_y_label=True,
            font_size_samples=17,
            font_size_genes=19,
            font_size_x_label=23,
            font_size_y_label=29,
            font_size_gene_bar_axis=13,
            tile_linewidth=2.5,
            gene_bar_scale_n_breaks=4,
        ),
    )
    mutation_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "customdata", None)
        and isinstance(trace.customdata[0], dict)
        and trace.customdata[0].get("role") == "main_tile"
    ]
    assert mutation_traces
    assert mutation_traces[0].marker.line.width == 2.5
    assert result.figure.layout.xaxis.tickfont.size == 17
    assert result.figure.layout.yaxis.tickfont.size == 19
    assert result.figure.layout.xaxis.title.font.size == 23
    assert result.figure.layout.yaxis.title.font.size == 29
    assert result.figure.layout.xaxis2.nticks == 4
    assert result.figure.layout.xaxis2.tickfont.size == 13


def test_plotly_tmb_and_metadata_font_sizes_are_applied():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    metadata_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="plotly",
        options=OncoplotOptions(metadata_position="top", font_size_metadata=16),
    )
    assert metadata_result.figure.layout.yaxis.tickfont.size == 16

    tmb_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        backend="plotly",
        options=OncoplotOptions(show_tmb_y_label=True, font_size_tmb_axis=14),
    )
    assert tmb_result.figure.layout.yaxis.tickfont.size == 14
    assert tmb_result.figure.layout.yaxis.title.font.size == 14


def test_plotly_bar_axes_share_matrix_category_extents():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        draw_tmb_bar=True,
        backend="plotly",
        options=OncoplotOptions(log10_transform_tmb=False),
    )
    samples = result.prepared_data.samples
    genes = result.prepared_data.genes
    sample_range = (-0.5, len(samples) - 0.5)
    gene_range = (-0.5, len(genes) - 0.5)

    assert list(result.figure.layout.xaxis.categoryarray) == samples
    assert tuple(result.figure.layout.xaxis.range) == sample_range
    assert list(result.figure.layout.xaxis2.categoryarray) == samples
    assert tuple(result.figure.layout.xaxis2.range) == sample_range
    assert list(result.figure.layout.yaxis2.categoryarray) == list(reversed(genes))
    assert tuple(result.figure.layout.yaxis2.range) == gene_range
    assert list(result.figure.layout.yaxis3.categoryarray) == list(reversed(genes))
    assert tuple(result.figure.layout.yaxis3.range) == gene_range

    tmb_bars = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "bar" and trace.orientation is None
    ]
    gene_bars = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "bar" and trace.orientation == "h"
    ]
    assert tmb_bars and gene_bars
    assert all(trace.width == 1 for trace in tmb_bars)
    assert all(trace.width == 1 for trace in gene_bars)


def test_plotly_forces_all_gene_and_metadata_tick_labels():
    samples = [f"S{i:02d}" for i in range(1, 11)]
    genes = [f"GENE{i:02d}" for i in range(1, 26)]
    metadata_cols = [f"meta_{i}" for i in range(1, 6)]
    mutations = pd.DataFrame(
        {
            "sample": [samples[index % len(samples)] for index in range(len(genes))],
            "gene": genes,
            "type": ["Missense_Mutation"] * len(genes),
        }
    )
    metadata = pd.DataFrame(
        {
            "sample": samples,
            **{column: [f"value_{index % 3}" for index in range(len(samples))] for column in metadata_cols},
        }
    )

    result = oncoplot(
        mutations,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=metadata_cols,
        draw_gene_bar=True,
        draw_tmb_bar=True,
        top_n=25,
        backend="plotly",
        options=OncoplotOptions(log10_transform_tmb=False),
    )

    main_gene_axis = result.figure.layout.yaxis2
    gene_bar_axis = result.figure.layout.yaxis3
    metadata_axis = result.figure.layout.yaxis4
    expected_metadata_labels = ["Meta 1", "Meta 2", "Meta 3", "Meta 4", "Meta 5"]

    assert main_gene_axis.tickmode == "array"
    assert list(main_gene_axis.tickvals) == result.prepared_data.genes
    assert list(main_gene_axis.ticktext) == result.prepared_data.genes
    assert main_gene_axis.automargin is True
    assert gene_bar_axis.showticklabels is False
    assert list(gene_bar_axis.tickvals) == result.prepared_data.genes
    assert metadata_axis.tickmode == "array"
    assert list(metadata_axis.tickvals) == expected_metadata_labels
    assert list(metadata_axis.ticktext) == expected_metadata_labels
    assert metadata_axis.automargin is True


def test_matplotlib_tmb_axis_uses_matrix_sample_boundaries():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        backend="matplotlib",
        options=OncoplotOptions(log10_transform_tmb=False),
    )
    tmb_axis = result.figure.axes[0]
    main_axis = result.figure.axes[1]
    n_samples = len(result.prepared_data.samples)

    assert tmb_axis.get_xlim() == main_axis.get_xlim() == (0.0, float(n_samples))
    assert tmb_axis.patches
    assert {round(patch.get_width(), 6) for patch in tmb_axis.patches} == {1.0}
    assert min(round(patch.get_x(), 6) for patch in tmb_axis.patches) == 0.0
    assert max(round(patch.get_x() + patch.get_width(), 6) for patch in tmb_axis.patches) == float(n_samples)


def test_multi_hit_color_applies_to_defaults_and_explicit_palette_wins():
    data = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "gene": ["TP53", "TP53", "TP53"],
            "type": ["Missense_Mutation", "Nonsense_Mutation", "Splice_Site"],
        }
    )
    default_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="plotly",
        options=OncoplotOptions(multi_hit_color="#FF00FF"),
    )
    default_multi_hit = next(trace for trace in default_result.figure.data if trace.name == "Multi Hit")
    assert default_multi_hit.marker.color == "#FF00FF"

    explicit_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        palette={"Multi_Hit": "#00FF00", "Splice_Site": "#6A3D9A"},
        backend="plotly",
        options=OncoplotOptions(multi_hit_color="#FF00FF"),
    )
    explicit_multi_hit = next(trace for trace in explicit_result.figure.data if trace.name == "Multi Hit")
    assert explicit_multi_hit.marker.color == "#00FF00"


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


def test_matplotlib_static_legend_text_scales_with_plot_fonts():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "clinical_group": ["A", "B", "A"]})
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["clinical_group"],
        backend="matplotlib",
        options=OncoplotOptions(
            font_size_genes=32,
            font_size_metadata=24,
            mutation_legend_position="right",
            metadata_legend_position="right",
            show_metadata_legends=True,
        ),
    )
    legend_by_title = {legend.get_title().get_text(): legend for legend in result.figure.legends}

    mutation_legend = legend_by_title["Mutation Type"]
    metadata_legend = legend_by_title["Clinical Group"]
    assert mutation_legend.get_texts()[0].get_fontsize() >= 24
    assert mutation_legend.get_title().get_fontsize() >= 25
    assert metadata_legend.get_texts()[0].get_fontsize() >= 20
    assert metadata_legend.get_title().get_fontsize() >= 21


def test_matplotlib_large_figure_legend_text_matches_export_scale():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "clinical_group": ["A", "B", "A"],
            "er_status": ["Positive", "Negative", "Positive"],
            "pr_status": ["Positive", "Negative", "Negative"],
            "her2_status": ["Negative", "Negative", "Positive"],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["clinical_group", "er_status", "pr_status", "her2_status"],
        backend="matplotlib",
        options=OncoplotOptions(
            width=7200,
            height=3600,
            font_size_genes=34,
            font_size_metadata=24,
            mutation_legend_position="right",
            metadata_legend_position="right",
            show_metadata_legends=True,
        ),
    )
    legend_by_title = {legend.get_title().get_text(): legend for legend in result.figure.legends}

    assert legend_by_title["Mutation Type"].get_texts()[0].get_fontsize() >= 40
    assert legend_by_title["Clinical Group"].get_texts()[0].get_fontsize() >= 28
    assert {"Clinical Group", "Er Status", "Pr Status", "Her2 Status"}.issubset(legend_by_title)


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


def test_metadata_fallback_palette_wraps_for_many_categories():
    samples = [f"S{i:02d}" for i in range(1, 10)]
    mutations = pd.DataFrame(
        {
            "sample": samples,
            "gene": ["TP53"] * len(samples),
            "type": ["Missense_Mutation"] * len(samples),
        }
    )
    metadata = pd.DataFrame(
        {
            "sample": samples,
            "group": [f"C{i}" for i in range(len(samples))],
        }
    )

    result = oncoplot(
        mutations,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="plotly",
        options=OncoplotOptions(metadata_default_colors=("#111111", "#222222")),
    )

    legend_colors = {
        trace.name: trace.marker.color
        for trace in result.figure.data
        if str(getattr(trace, "legendgroup", "")).startswith("metadata:")
    }
    assert legend_colors["Group: C0"] == "#111111"
    assert legend_colors["Group: C1"] == "#222222"
    assert legend_colors["Group: C2"] == "#111111"
    assert legend_colors["Group: C3"] == "#222222"


def test_metadata_categorical_order_controls_fallback_colors_and_legends_in_both_backends():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "group": pd.Categorical(
                ["B", "A", "B"],
                categories=["A", "B", "C"],
                ordered=True,
            ),
        }
    )
    options = OncoplotOptions(
        metadata_default_colors=("#111111", "#222222"),
        mutation_legend_position="none",
    )

    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="plotly",
        options=options,
    )
    plotly_legend = [
        trace
        for trace in plotly_result.figure.data
        if str(getattr(trace, "legendgroup", "")).startswith("metadata:")
    ]
    assert [trace.name for trace in plotly_legend] == ["Group: A", "Group: B"]
    assert [trace.marker.color for trace in plotly_legend] == ["#111111", "#222222"]

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="matplotlib",
        options=options,
    )
    metadata_legend = next(legend for legend in matplotlib_result.figure.legends if legend.get_title().get_text() == "Group")
    assert [text.get_text() for text in metadata_legend.get_texts()] == ["A", "B"]
    legend_handles = getattr(metadata_legend, "legend_handles", None)
    if legend_handles is None:
        legend_handles = metadata_legend.legendHandles
    assert [to_hex(handle.get_facecolor()).lower() for handle in legend_handles] == [
        "#111111",
        "#222222",
    ]


def test_metadata_palette_mapping_order_is_used_when_no_explicit_or_dtype_order():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["B", "A", "B"]})
    metadata_palette = {"group": {"A": "#111111", "B": "#222222"}}

    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        metadata_palette=metadata_palette,
        backend="plotly",
        options=OncoplotOptions(mutation_legend_position="none"),
    )

    metadata_legend = [
        trace
        for trace in result.figure.data
        if str(getattr(trace, "legendgroup", "")).startswith("metadata:")
    ]
    assert [trace.name for trace in metadata_legend] == ["Group: A", "Group: B"]
    assert [trace.marker.color for trace in metadata_legend] == ["#111111", "#222222"]


def test_mutation_type_order_controls_legend_order_in_both_backends():
    data = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "gene": ["TP53", "TP53", "TP53"],
            "type": ["beta_type", "alpha_type", "gamma_type"],
        }
    )
    order = ["gamma_type", "alpha_type"]

    plotly_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        mutation_type_order=order,
        backend="plotly",
    )
    mutation_traces = [
        trace
        for trace in plotly_result.figure.data
        if getattr(trace, "customdata", None)
        and isinstance(trace.customdata[0], dict)
        and trace.customdata[0].get("role") == "main_tile"
    ]
    assert [trace.name for trace in mutation_traces] == ["Gamma Type", "Alpha Type", "Beta Type"]
    assert [trace.marker.color for trace in mutation_traces] == ["#A6CEE3", "#1F78B4", "#B2DF8A"]

    matplotlib_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        mutation_type_order=order,
        backend="matplotlib",
        options=OncoplotOptions(mutation_legend_position="right"),
    )
    mutation_legend = next(
        legend for legend in matplotlib_result.figure.legends if legend.get_title().get_text() == "Mutation Type"
    )
    assert [text.get_text() for text in mutation_legend.get_texts()] == [
        "Gamma Type",
        "Alpha Type",
        "Beta Type",
    ]


def test_mutation_palette_mapping_order_is_used_when_no_explicit_or_dtype_order():
    data = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "gene": ["TP53", "TP53", "TP53"],
            "type": ["beta_type", "alpha_type", "gamma_type"],
        }
    )
    palette = {
        "gamma_type": "#333333",
        "alpha_type": "#111111",
        "beta_type": "#222222",
    }

    plotly_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        palette=palette,
        backend="plotly",
    )
    mutation_traces = [
        trace
        for trace in plotly_result.figure.data
        if getattr(trace, "customdata", None)
        and isinstance(trace.customdata[0], dict)
        and trace.customdata[0].get("role") == "main_tile"
    ]
    assert [trace.name for trace in mutation_traces] == ["Gamma Type", "Alpha Type", "Beta Type"]

    matplotlib_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        palette=palette,
        backend="matplotlib",
        options=OncoplotOptions(mutation_legend_position="right"),
    )
    mutation_legend = next(
        legend for legend in matplotlib_result.figure.legends if legend.get_title().get_text() == "Mutation Type"
    )
    assert [text.get_text() for text in mutation_legend.get_texts()] == [
        "Gamma Type",
        "Alpha Type",
        "Beta Type",
    ]


def test_tmb_type_order_controls_stacked_trace_and_legend_order():
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": ["Beta", "Alpha", "Beta"],
            "mutations": [2, 5, 3],
        }
    )
    tmb_palette = {"Beta": "#222222", "Alpha": "#111111"}

    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette=tmb_palette,
        tmb_type_order=["Alpha", "Beta"],
        backend="plotly",
        options=OncoplotOptions(log10_transform_tmb=False, mutation_legend_position="right"),
    )
    tmb_traces = [
        trace for trace in plotly_result.figure.data if getattr(trace, "legendgroup", "") == "tmb"
    ]
    assert [trace.name for trace in tmb_traces] == ["TMB: Alpha", "TMB: Beta"]

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_tmb_bar=True,
        tmb_data=tmb,
        tmb_palette=tmb_palette,
        tmb_type_order=["Alpha", "Beta"],
        backend="matplotlib",
        options=OncoplotOptions(log10_transform_tmb=False, mutation_legend_position="right"),
    )
    tmb_legend = next(legend for legend in matplotlib_result.figure.legends if legend.get_title().get_text() == "TMB Type")
    assert [text.get_text() for text in tmb_legend.get_texts()] == ["Alpha", "Beta"]


def test_categorical_metadata_palette_string_names_render_in_both_backends():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    expected_a = to_hex(tol_colors[0]).lower()
    expected_b = to_hex(tol_colors[1]).lower()

    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        metadata_palette={"group": "tol_colors"},
        backend="plotly",
    )
    legend_colors = {
        trace.name: trace.marker.color
        for trace in plotly_result.figure.data
        if str(getattr(trace, "legendgroup", "")).startswith("metadata:")
    }
    assert legend_colors["Group: A"] == expected_a
    assert legend_colors["Group: B"] == expected_b

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        metadata_palette={"group": "tol_colors"},
        backend="matplotlib",
    )
    patch_colors = {
        to_hex(patch.get_facecolor()).lower()
        for axis in matplotlib_result.figure.axes
        for patch in axis.patches
    }
    assert expected_a in patch_colors
    assert expected_b in patch_colors


def test_categorical_metadata_palette_accepts_matplotlib_colormap_names():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    dark2 = colormaps.get_cmap("Dark2")
    expected_a = to_hex(dark2.colors[0]).lower()
    expected_b = to_hex(dark2.colors[1]).lower()

    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        metadata_palette={"group": "Dark2"},
        backend="plotly",
    )
    legend_colors = {
        trace.name: trace.marker.color
        for trace in result.figure.data
        if str(getattr(trace, "legendgroup", "")).startswith("metadata:")
    }
    assert legend_colors["Group: A"] == expected_a
    assert legend_colors["Group: B"] == expected_b


def test_numeric_metadata_supports_per_column_continuous_colormaps():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "score": [0.0, 0.5, 1.0],
            "purity": [0.0, 0.5, 1.0],
        }
    )
    metadata_palette = {"score": "viridis_greyzero", "purity": "Iridescent"}
    expected_score_zero = "#808080"
    expected_score_mid = to_hex(colormaps.get_cmap("viridis_greyzero")(0.5))
    expected_purity_mid = to_hex(LinearSegmentedColormap.from_list("iridescent_test", Iridescent)(0.5))

    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["score", "purity"],
        metadata_palette=metadata_palette,
        backend="plotly",
    )
    metadata_heatmap = next(
        trace
        for trace in plotly_result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and trace.customdata is not None
        and trace.customdata[0][0]["role"] == "metadata"
    )
    plotly_colors = {color.lower() for _stop, color in metadata_heatmap.colorscale}
    assert expected_score_zero in plotly_colors
    assert expected_score_mid in plotly_colors
    assert expected_purity_mid in plotly_colors
    colorbar_traces = [
        trace
        for trace in plotly_result.figure.data
        if getattr(getattr(trace, "marker", None), "showscale", False)
    ]
    assert [trace.marker.colorbar.title.text for trace in colorbar_traces] == ["Score", "Purity"]
    score_colorbar = colorbar_traces[0]
    score_colorbar_colors = {color.lower() for _stop, color in score_colorbar.marker.colorscale}
    assert expected_score_zero in score_colorbar_colors
    assert expected_score_mid in score_colorbar_colors
    assert score_colorbar.marker.cmin == 0.0
    assert score_colorbar.marker.cmax == 1.0

    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["score", "purity"],
        metadata_palette=metadata_palette,
        backend="matplotlib",
        options=OncoplotOptions(metadata_numeric_plot_type="heatmap"),
    )
    patch_colors = {
        to_hex(patch.get_facecolor()).lower()
        for axis in matplotlib_result.figure.axes
        for patch in axis.patches
    }
    assert expected_score_zero in patch_colors
    assert expected_score_mid in patch_colors
    assert expected_purity_mid in patch_colors


def test_metadata_palette_unknown_names_raise_clear_errors():
    categorical_metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    with pytest.raises(
        ValueError,
        match="Unknown metadata palette 'no_such_palette' for categorical metadata column 'group'",
    ):
        oncoplot(
            small_df(),
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            metadata=categorical_metadata,
            metadata_cols=["group"],
            metadata_palette={"group": "no_such_palette"},
            backend="plotly",
        )

    numeric_metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "score": [0.0, 0.5, 1.0]})
    with pytest.raises(
        ValueError,
        match="Unknown metadata palette 'no_such_palette' for numeric metadata column 'score'",
    ):
        oncoplot(
            small_df(),
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            metadata=numeric_metadata,
            metadata_cols=["score"],
            metadata_palette={"score": "no_such_palette"},
            backend="plotly",
        )


def test_plotly_numeric_metadata_colorbars_follow_metadata_legend_options():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "score": [1.0, 3.0, 5.0],
        }
    )
    hidden_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["score"],
        metadata_palette={"score": "viridis"},
        backend="plotly",
        options=OncoplotOptions(show_metadata_legends=False),
    )
    assert not any(
        getattr(getattr(trace, "marker", None), "showscale", False)
        for trace in hidden_result.figure.data
    )

    horizontal_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["score"],
        metadata_palette={"score": "viridis"},
        backend="plotly",
        options=OncoplotOptions(metadata_legend_orientation_heatmap="horizontal"),
    )
    colorbar_trace = next(
        trace
        for trace in horizontal_result.figure.data
        if getattr(getattr(trace, "marker", None), "showscale", False)
    )
    assert colorbar_trace.marker.colorbar.orientation == "h"
    assert horizontal_result.figure.layout.margin.b >= 135


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
