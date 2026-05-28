from pathlib import Path

import pandas as pd
import pytest
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, to_hex
from PIL import Image

from pyoncoplot import Iridescent, OncoplotOptions, oncoplot, tol_colors


def small_df():
    return pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S3"],
            "gene": ["TP53", "EGFR", "TP53", "PTEN"],
            "type": ["Missense_Mutation", "Frame_Shift_Del", "Nonsense_Mutation", "Splice_Site"],
            "vaf": [0.34, 0.21, 0.67, 0.48],
            "vaf_abs": [0.034, 0.021, 0.067, 0.048],
        }
    )


def matplotlib_colorbar_axes(figure, *labels):
    label_set = set(labels)
    return [
        axis
        for axis in figure.axes
        if axis.get_ylabel() in label_set or axis.get_xlabel() in label_set or axis.get_title() in label_set
    ]


def axis_colormap_colors(axis, fraction):
    return {
        to_hex(collection.cmap(fraction)).lower()
        for collection in axis.collections
        if getattr(collection, "cmap", None) is not None
    }


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


def test_plotly_continuous_variant_heatmap_and_gene_bar_legend():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_col="vaf",
        draw_gene_bar=True,
        backend="plotly",
        options=OncoplotOptions(width=700, height=450),
    )

    main_heatmap = next(
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
    )
    gene_bar_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "bar" and trace.orientation == "h"
    ]

    assert main_heatmap.showscale is True
    tp53_s1_payload = next(
        payload
        for row in main_heatmap.customdata
        for payload in row
        if payload.get("sample") == "S1" and payload.get("gene") == "TP53"
    )
    assert tp53_s1_payload["variant_value"] == pytest.approx(0.34)
    assert "Vaf" in main_heatmap.colorbar.title.text
    assert gene_bar_traces
    assert any(trace.showlegend is True for trace in gene_bar_traces)


def test_plotly_continuous_variant_heatmap_leaves_missing_values_blank():
    df = small_df()
    df.loc[0, "vaf"] = None
    result = oncoplot(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_col="vaf",
        backend="plotly",
        options=OncoplotOptions(width=700, height=450),
    )

    main_heatmap = next(
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
    )
    tp53_index = result.prepared_data.genes.index("TP53")
    s1_index = result.prepared_data.samples.index("S1")

    assert pd.isna(main_heatmap.z[tp53_index][s1_index])
    assert "variant_value" not in main_heatmap.customdata[tp53_index][s1_index]
    assert result.prepared_data.variant_value_min == pytest.approx(0.21)
    assert result.prepared_data.variant_value_max == pytest.approx(0.67)


def test_plotly_continuous_variant_heatmap_skips_colorbar_when_all_values_are_missing():
    df = small_df()
    df["vaf"] = pd.NA
    result = oncoplot(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_col="vaf",
        backend="plotly",
        options=OncoplotOptions(width=700, height=450),
    )

    main_heatmap = next(
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
    )

    assert main_heatmap.showscale is False
    assert result.prepared_data.variant_value_min is None
    assert result.prepared_data.variant_value_max is None


def test_plotly_multi_row_main_grid_renders_categorical_and_continuous_tracks():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf", "label": "VAF %"},
            {"kind": "variant_value", "column": "vaf_abs", "label": "VAF abs", "palette": "magma"},
        ],
        draw_gene_bar=True,
        backend="plotly",
        options=OncoplotOptions(width=760, height=500),
    )

    main_heatmaps = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
        and getattr(trace, "showscale", None) is True
    ]
    mutation_heatmaps = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
        and getattr(trace, "showlegend", None) is True
    ]
    gene_bar_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "bar" and trace.orientation == "h"
    ]

    assert result.prepared_data.main_grid_mode == "expanded"
    assert len(result.prepared_data.main_grid_rows) == len(result.prepared_data.genes) * 3
    assert len(main_heatmaps) == 2
    assert {trace.colorbar.title.text for trace in main_heatmaps} == {"VAF %", "VAF Abs"}
    tp53_s1_payload = next(
        payload
        for trace in main_heatmaps
        for row in trace.customdata
        for payload in row
        if payload.get("sample") == "S1"
        and payload.get("gene") == "TP53"
        and payload.get("variant_value_column") == "vaf"
    )
    tp53_s1_variant_hover = next(
        trace.text[row_index][sample_index]
        for trace in main_heatmaps
        for row_index, row in enumerate(trace.customdata)
        for sample_index, payload in enumerate(row)
        if payload.get("sample") == "S1"
        and payload.get("gene") == "TP53"
        and payload.get("variant_value_column") == "vaf"
    )
    tp53_s1_mutation_hover = next(
        trace.text[row_index][sample_index]
        for trace in mutation_heatmaps
        for row_index, row in enumerate(trace.customdata)
        for sample_index, payload in enumerate(row)
        if payload.get("sample") == "S1" and payload.get("gene") == "TP53"
    )
    assert tp53_s1_payload["variant_value"] == pytest.approx(0.34)
    assert tp53_s1_mutation_hover == (
        "Sample: S1<br>TP53: Missense_Mutation<br>VAF %: 0.34<br>VAF abs: 0.034"
    )
    assert tp53_s1_variant_hover == "Sample: S1<br>TP53: Missense_Mutation<br>VAF %: 0.34"
    assert "<strong>" not in tp53_s1_mutation_hover
    assert "<strong>" not in tp53_s1_variant_hover
    assert mutation_heatmaps
    assert any(trace.showlegend is True for trace in mutation_heatmaps)
    assert gene_bar_traces


def test_plotly_expanded_categorical_multi_hit_tiles_fill_cells():
    df = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "gene": ["TP53", "TP53", "TP53"],
            "type": ["Missense_Mutation", "Nonsense_Mutation", "Missense_Mutation"],
            "vaf": [0.2, 0.6, 0.4],
        }
    )
    result = oncoplot(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf", "label": "VAF %"},
        ],
        palette={
            "Missense_Mutation": "#00AA00",
            "Nonsense_Mutation": "#CC0000",
            "Multi_Hit": "#000000",
        },
        backend="plotly",
        options=OncoplotOptions(width=500, height=350),
    )

    expanded_marker_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "scatter"
        and getattr(trace, "mode", "") == "markers"
        and getattr(trace, "customdata", None)
    ]
    mutation_heatmap = next(
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "name", "") == "Multi Hit"
        and getattr(trace, "showlegend", None) is True
    )
    vaf_heatmap = next(
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "showscale", None) is True
        and trace.colorbar.title.text == "VAF %"
    )
    mutation_row = result.prepared_data.main_grid_rows[
        (result.prepared_data.main_grid_rows["Gene"] == "TP53")
        & (result.prepared_data.main_grid_rows["Kind"] == "mutation_type")
    ].iloc[0]
    vaf_row = result.prepared_data.main_grid_rows[
        (result.prepared_data.main_grid_rows["Gene"] == "TP53")
        & (result.prepared_data.main_grid_rows["Label"] == "VAF %")
    ].iloc[0]
    s1_index = result.prepared_data.samples.index("S1")

    assert expanded_marker_traces == []
    assert mutation_heatmap.showscale is False
    assert mutation_heatmap.z[int(mutation_row["RowIndex"])][s1_index] == pytest.approx(1.0)
    assert mutation_heatmap.customdata[int(mutation_row["RowIndex"])][s1_index]["mutation_type"] == "Multi_Hit"
    assert vaf_heatmap.z[int(vaf_row["RowIndex"])][s1_index] == pytest.approx(0.6)


def test_plotly_multi_row_main_grid_leaves_missing_values_blank_and_offsets_gene_labels():
    df = small_df()
    df.loc[0, "vaf"] = None
    result = oncoplot(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf", "label": "VAF %"},
            {"kind": "variant_value", "column": "vaf_abs", "label": "VAF abs"},
        ],
        backend="plotly",
        gene_name_x_offset=18,
        main_grid_rows_label_x_offset=12,
        options=OncoplotOptions(width=760, height=500, prettify_legend_titles=False),
    )

    main_heatmap = next(
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
        and trace.colorbar.title.text == "VAF %"
    )
    tp53_vaf_row = result.prepared_data.main_grid_rows[
        (result.prepared_data.main_grid_rows["Gene"] == "TP53")
        & (result.prepared_data.main_grid_rows["Label"] == "VAF %")
    ].iloc[0]
    s1_index = result.prepared_data.samples.index("S1")
    gene_label = next(annotation for annotation in result.figure.layout.annotations if annotation.text == "TP53")
    row_label = next(annotation for annotation in result.figure.layout.annotations if annotation.text == "VAF %")

    assert pd.isna(main_heatmap.z[int(tp53_vaf_row["RowIndex"])][s1_index])
    assert gene_label.xshift == pytest.approx(-72)
    assert row_label.xshift == pytest.approx(-20)


def test_plotly_variant_value_cols_shared_scale_uses_one_colorbar():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_cols=["vaf", "vaf_abs"],
        variant_value_scale="shared",
        backend="plotly",
        options=OncoplotOptions(width=760, height=500),
    )

    main_heatmaps = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
        and getattr(trace, "showscale", None) is True
    ]
    assert len(main_heatmaps) == 1
    assert main_heatmaps[0].showscale is True
    assert main_heatmaps[0].zmin == pytest.approx(0.021)
    assert main_heatmaps[0].zmax == pytest.approx(0.67)


def test_matplotlib_continuous_variant_heatmap_leaves_missing_values_blank():
    df = small_df()
    df.loc[0, "vaf"] = None
    result = oncoplot(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_col="vaf",
        backend="matplotlib",
        options=OncoplotOptions(width=700, height=450),
    )

    main_axis = result.figure.axes[0]
    assert result.backend == "matplotlib"
    assert result.prepared_data.tiles["VariantValue"].isna().any()
    assert main_axis.patches


def test_matplotlib_continuous_variant_heatmap_skips_colorbar_when_all_values_are_missing():
    df = small_df()
    df["vaf"] = pd.NA
    result = oncoplot(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_col="vaf",
        backend="matplotlib",
        options=OncoplotOptions(width=700, height=450),
    )

    assert result.prepared_data.variant_value_min is None
    assert result.prepared_data.variant_value_max is None
    assert matplotlib_colorbar_axes(result.figure, "Vaf") == []


def test_matplotlib_multi_row_main_grid_offsets_gene_labels_from_row_labels():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf", "label": "VAF %"},
        ],
        backend="matplotlib",
        gene_name_x_offset=18,
        main_grid_rows_label_x_offset=12,
        options=OncoplotOptions(width=700, height=450, prettify_legend_titles=False),
    )

    main_axis = result.figure.axes[0]
    gene_label = next(text for text in main_axis.texts if text.get_text() == "TP53")
    row_label = next(text for text in main_axis.texts if text.get_text() == "VAF %")

    assert gene_label.xyann[0] == pytest.approx(-72)
    assert row_label.xyann[0] == pytest.approx(-20)


def test_plotly_gene_bar_percent_mode_normalizes_each_gene():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        backend="plotly",
        options=OncoplotOptions(width=700, height=450, gene_bar_mode="percent"),
    )
    gene_bar_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "bar" and trace.orientation == "h"
    ]
    totals_by_gene = {gene: 0.0 for gene in result.prepared_data.genes}
    for trace in gene_bar_traces:
        for gene, value in zip(trace.y, trace.x):
            totals_by_gene[str(gene)] += float(value)

    assert all(total == pytest.approx(100.0) for total in totals_by_gene.values())
    assert result.figure.layout.xaxis2.title.text == "Mutation Type (%)"


def test_plotly_gene_bar_count_mode_uses_sample_counts_by_default():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        backend="plotly",
        options=OncoplotOptions(width=700, height=450),
    )
    gene_bar_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "bar" and trace.orientation == "h"
    ]
    totals_by_gene = {gene: 0.0 for gene in result.prepared_data.genes}
    for trace in gene_bar_traces:
        for gene, value in zip(trace.y, trace.x):
            totals_by_gene[str(gene)] += float(value)

    assert totals_by_gene == {"EGFR": 1.0, "PTEN": 1.0, "TP53": 2.0}
    assert result.figure.layout.xaxis2.title.text == "Samples"


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


def test_matplotlib_save_can_preserve_exact_figure_extent(tmp_path):
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="matplotlib",
        options=OncoplotOptions(width=700, height=450),
    )
    output = tmp_path / "exact-oncoplot.png"
    result.figure.set_size_inches(4, 3, forward=True)
    result.save(str(output), dpi=120, bbox_inches=None)

    try:
        with Image.open(output) as image:
            assert image.size == (480, 360)
    finally:
        import matplotlib.pyplot as plt

        plt.close(result.figure)


def test_matplotlib_continuous_variant_heatmap_and_save(tmp_path):
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_col="vaf",
        backend="matplotlib",
        options=OncoplotOptions(width=700, height=450),
    )
    output = tmp_path / "continuous-oncoplot.png"
    result.save(str(output), dpi=90)

    assert output.exists()
    assert output.stat().st_size > 0
    assert len(result.figure.axes) >= 2
    colorbar_axis = matplotlib_colorbar_axes(result.figure, "Vaf")[0]
    assert colorbar_axis.get_title() == "Vaf"
    assert colorbar_axis.get_ylabel() == ""
    assert colorbar_axis.get_position().width > colorbar_axis.get_position().height


def test_matplotlib_multi_row_main_grid_and_colorbars(tmp_path):
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_cols=["vaf", "vaf_abs"],
        draw_gene_bar=True,
        backend="matplotlib",
        options=OncoplotOptions(width=760, height=500),
    )
    output = tmp_path / "multi-row-oncoplot.png"
    result.save(str(output), dpi=90)

    assert output.exists()
    assert output.stat().st_size > 0
    assert result.prepared_data.main_grid_mode == "expanded"
    assert len(result.prepared_data.main_grid_rows) == len(result.prepared_data.genes) * 3
    main_position = result.figure.axes[0].get_position()
    gene_bar_position = result.figure.axes[1].get_position()
    colorbar_axes = matplotlib_colorbar_axes(result.figure, "Vaf", "Vaf Abs")
    assert [axis.get_title() for axis in colorbar_axes] == ["Vaf", "Vaf Abs"]
    assert all(axis.get_ylabel() == "" for axis in colorbar_axes)
    assert all(axis.get_position().width > axis.get_position().height for axis in colorbar_axes)
    assert gene_bar_position.y0 == pytest.approx(main_position.y0)
    assert gene_bar_position.height == pytest.approx(main_position.height)


def test_matplotlib_gene_bar_percent_mode_normalizes_each_gene():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        draw_gene_bar=True,
        backend="matplotlib",
        options=OncoplotOptions(width=700, height=450, gene_bar_mode="percent"),
    )
    gene_bar_axis = result.figure.axes[1]
    widths_by_y = {}
    for patch in gene_bar_axis.patches:
        y_key = round(patch.get_y(), 3)
        widths_by_y[y_key] = widths_by_y.get(y_key, 0.0) + float(patch.get_width())

    assert all(width == pytest.approx(100.0) for width in widths_by_y.values())
    assert gene_bar_axis.get_xlabel() == "Mutation Type (%)"


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
    assert metadata_heatmap.text[0][0] == "Sample: S1<br>Clinical Group: High Risk"
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


def test_plotly_bottom_metadata_hover_includes_sample_id():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="plotly",
        options=OncoplotOptions(metadata_position="bottom", mutation_legend_position="none"),
    )

    metadata_heatmap = next(
        trace
        for trace in result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "metadata"
    )

    assert metadata_heatmap.text[0][0] == "Sample: S1<br>Group: A"
    assert metadata_heatmap.customdata[0][0]["sample"] == "S1"


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


def test_matplotlib_expanded_main_grid_aligns_with_tmb_axis():
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf", "label": "VAF"},
        ],
        draw_tmb_bar=True,
        backend="matplotlib",
        options=OncoplotOptions(log10_transform_tmb=False),
    )
    tmb_axis = result.figure.axes[0]
    main_axis = result.figure.axes[1]
    n_samples = len(result.prepared_data.samples)

    assert result.prepared_data.main_grid_mode == "expanded"
    assert tmb_axis.get_xlim() == main_axis.get_xlim() == (0.0, float(n_samples))


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


def test_matplotlib_legend_offsets_move_only_targeted_legend():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    base = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="matplotlib",
        options=OncoplotOptions(
            mutation_legend_position="right",
            metadata_legend_position="right",
        ),
    )
    shifted = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        backend="matplotlib",
        options=OncoplotOptions(
            mutation_legend_position="right",
            metadata_legend_position="right",
            legend_offsets={"metadata:group": {"x": 0.08}},
        ),
    )
    base.figure.canvas.draw()
    shifted.figure.canvas.draw()

    base_legends = {legend.get_title().get_text(): legend for legend in base.figure.legends}
    shifted_legends = {legend.get_title().get_text(): legend for legend in shifted.figure.legends}
    assert shifted_legends["Group"].get_window_extent().x0 > base_legends["Group"].get_window_extent().x0
    assert shifted_legends["Mutation Type"].get_window_extent().x0 == pytest.approx(
        base_legends["Mutation Type"].get_window_extent().x0,
        abs=1,
    )


def test_plotly_legend_offsets_split_only_targeted_legend_group():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "group": ["A", "B", "A"],
            "status": ["High", "Low", "High"],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group", "status"],
        backend="plotly",
        options=OncoplotOptions(
            mutation_legend_position="right",
            metadata_legend_position="right",
            legend_offsets={"metadata:group": {"x": 0.05, "y": -0.02}},
        ),
    )
    group_traces = [
        trace for trace in result.figure.data if getattr(trace, "legendgroup", "") == "metadata:group"
    ]
    status_traces = [
        trace for trace in result.figure.data if getattr(trace, "legendgroup", "") == "metadata:status"
    ]
    mutation_traces = [
        trace
        for trace in result.figure.data
        if getattr(trace, "showlegend", None) is True
        and getattr(trace, "legendgroup", "") not in {"metadata:group", "metadata:status"}
    ]

    assert group_traces and all(trace.legend == "legend2" for trace in group_traces)
    assert status_traces and all(getattr(trace, "legend", None) is None for trace in status_traces)
    assert mutation_traces and all(getattr(trace, "legend", None) is None for trace in mutation_traces)
    assert result.figure.layout.legend2.x == pytest.approx(result.figure.layout.legend.x + 0.05)
    assert result.figure.layout.legend2.y == pytest.approx(result.figure.layout.legend.y - 0.02)


def test_legend_offsets_apply_to_variant_and_metadata_colorbars():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "score": [1.0, 3.0, 5.0]})
    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        variant_value_col="vaf",
        metadata=metadata,
        metadata_cols=["score"],
        backend="plotly",
        options=OncoplotOptions(
            metadata_legend_orientation_heatmap="horizontal",
            legend_offsets={
                "variant:vaf": {"x": 0.04, "y": 0.08},
                "metadata:score": {"x": -0.03, "y": 0.02},
            },
        ),
    )
    variant_heatmap = next(
        trace
        for trace in plotly_result.figure.data
        if getattr(trace, "type", "") == "heatmap"
        and getattr(trace, "customdata", None)
        and trace.customdata[0][0]["role"] == "main_tile"
    )
    metadata_colorbar = next(
        trace
        for trace in plotly_result.figure.data
        if getattr(getattr(trace, "marker", None), "showscale", False)
    )

    assert variant_heatmap.colorbar.x == pytest.approx(1.06)
    assert variant_heatmap.colorbar.y == pytest.approx(0.58)
    assert metadata_colorbar.marker.colorbar.x < 0.5
    assert metadata_colorbar.marker.colorbar.y == pytest.approx(-0.30)


def test_explicit_legend_font_sizes_and_character_limits_are_applied():
    data = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "gene": ["TP53", "TP53", "TP53"],
            "type": ["very_long_mutation_label", "another_long_mutation_label", "short"],
        }
    )
    matplotlib_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="matplotlib",
        options=OncoplotOptions(
            mutation_legend_position="right",
            font_size_legend_text=11,
            font_size_legend_title=13,
            legend_label_max_chars=10,
            legend_title_max_chars=8,
        ),
    )
    legend = matplotlib_result.figure.legends[0]
    assert legend.get_texts()[0].get_fontsize() == 11
    assert legend.get_title().get_fontsize() == 13
    assert legend.get_title().get_text() == "Mutat..."
    assert all(len(text.get_text()) <= 10 for text in legend.get_texts())

    plotly_result = oncoplot(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        backend="plotly",
        options=OncoplotOptions(
            font_size_legend_text=15,
            legend_label_max_chars=9,
        ),
    )
    mutation_traces = [
        trace
        for trace in plotly_result.figure.data
        if getattr(trace, "showlegend", None) is True
        and getattr(trace, "customdata", None)
        and trace.customdata[0]["role"] == "main_tile"
    ]
    assert plotly_result.figure.layout.legend.font.size == 15
    assert all(len(trace.name) <= 9 for trace in mutation_traces)


def test_title_and_subplot_title_options_apply_to_both_backends():
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]})
    options = OncoplotOptions(
        title_text="Cohort overview",
        main_subplot_title="Mutations",
        tmb_subplot_title="Burden",
        gene_bar_subplot_title="Counts",
        metadata_subplot_title="Clinical",
        font_size_title=18,
        font_size_subplot_title=11,
    )
    matplotlib_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        draw_gene_bar=True,
        draw_tmb_bar=True,
        backend="matplotlib",
        options=options,
    )
    assert matplotlib_result.figure._suptitle.get_text() == "Cohort overview"
    assert {axis.get_title() for axis in matplotlib_result.figure.axes}.issuperset(
        {"Mutations", "Burden", "Counts", "Clinical"}
    )

    plotly_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        draw_gene_bar=True,
        draw_tmb_bar=True,
        backend="plotly",
        options=options,
    )
    annotation_text = {annotation.text for annotation in plotly_result.figure.layout.annotations}
    assert plotly_result.figure.layout.title.text == "Cohort overview"
    assert {"Mutations", "Burden", "Counts", "Clinical"}.issubset(annotation_text)


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
    matplotlib_colorbars = matplotlib_colorbar_axes(matplotlib_result.figure, "Score", "Purity")
    assert [axis.get_title() for axis in matplotlib_colorbars] == ["Score", "Purity"]
    assert [axis.get_ylabel() for axis in matplotlib_colorbars] == ["", ""]
    assert all(axis.get_position().height > axis.get_position().width for axis in matplotlib_colorbars)
    assert all(axis.get_position().x0 > 0.80 for axis in matplotlib_colorbars)
    assert [text.get_text() for text in matplotlib_colorbars[0].get_yticklabels() if text.get_text()] == ["0", "1"]
    assert expected_score_zero in axis_colormap_colors(matplotlib_colorbars[0], 0.0)
    assert expected_score_mid in axis_colormap_colors(matplotlib_colorbars[0], 0.5)
    assert expected_purity_mid in axis_colormap_colors(matplotlib_colorbars[1], 0.5)


def test_matplotlib_numeric_metadata_colorbars_follow_legend_options():
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
        backend="matplotlib",
        options=OncoplotOptions(
            mutation_legend_position="none",
            show_metadata_legends=False,
        ),
    )
    assert not matplotlib_colorbar_axes(hidden_result.figure, "Score")
    assert any(axis.patches for axis in hidden_result.figure.axes)

    horizontal_result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["score"],
        metadata_palette={"score": "viridis"},
        backend="matplotlib",
        options=OncoplotOptions(
            mutation_legend_position="none",
            metadata_legend_orientation_heatmap="horizontal",
        ),
    )
    horizontal_colorbar = matplotlib_colorbar_axes(horizontal_result.figure, "Score")[0]
    assert horizontal_colorbar.get_xlabel() == "Score"
    assert horizontal_colorbar.get_position().width > horizontal_colorbar.get_position().height
    assert horizontal_result.figure.subplotpars.bottom >= 0.20


def test_matplotlib_numeric_metadata_bar_tracks_get_continuous_colorbars():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "score": [1.0, 3.0, 5.0],
        }
    )
    result = oncoplot(
        small_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["score"],
        metadata_palette={"score": "magma"},
        backend="matplotlib",
        options=OncoplotOptions(
            mutation_legend_position="none",
            metadata_numeric_plot_type="bar",
        ),
    )
    metadata_axis = next(
        axis
        for axis in result.figure.axes
        if [label.get_text() for label in axis.get_yticklabels()] == ["Score"]
    )
    metadata_text = {text.get_text() for text in metadata_axis.texts}
    assert {"1", "5"}.issubset(metadata_text)
    colorbar = matplotlib_colorbar_axes(result.figure, "Score")[0]
    assert colorbar.get_title() == "Score"
    assert colorbar.get_ylabel() == ""
    assert colorbar.get_position().height > colorbar.get_position().width
    assert [text.get_text() for text in colorbar.get_yticklabels() if text.get_text()] == ["1", "5"]


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
    with pytest.raises(ValueError, match="font_size_legend_text"):
        OncoplotOptions(font_size_legend_text=0)
    with pytest.raises(ValueError, match="legend_label_max_chars"):
        OncoplotOptions(legend_label_max_chars=0)
    with pytest.raises(ValueError, match="legend_offsets"):
        OncoplotOptions(legend_offsets={"metadata:group": {"z": 1}})
    with pytest.raises(ValueError, match="gene_name_x_offset"):
        OncoplotOptions(gene_name_x_offset=-1)
    with pytest.raises(ValueError, match="main_grid_rows_label_x_offset"):
        OncoplotOptions(main_grid_rows_label_x_offset=-1)
