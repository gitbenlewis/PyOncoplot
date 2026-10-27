"""Compatibility and eligible-population regressions from independent review."""
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pyoncoplot import OncoplotOptions, oncoplot, prepare_oncoplot_data
from pyoncoplot._options import coerce_options


def population(**overrides):
    params = dict(data=pd.DataFrame({"s": ["a", "b"], "g": ["G", "G"], "quality": [1, 0]}),
                  gene_col="g", sample_col="s",
                  metadata=pd.DataFrame({"s": ["a", "b"], "age": [20, None]}), metadata_cols=["age"],
                  filter_mutations_by_isin_lists={"quality": [1]},
                  metadata_filters={"age": {"min": 18, "missing": "error"}},
                  metadata_filter_mode="analysis_cohort")
    params.update(overrides)
    return prepare_oncoplot_data(**params)


def test_excluded_metadata_and_tmb_do_not_trigger_missing():
    assert population().samples == ["a"]
    result = population(prepare_tmb=True, show_all_samples=True,
        tmb_data=pd.DataFrame({"s": ["a", "b", "outside"], "tmb": [1, 2, 3]}))
    assert result.samples == ["a"]
    assert result.total_samples == 1


def test_retained_metadata_and_states_still_enforce_missing():
    with pytest.raises(ValueError, match="Missing metadata"):
        population(metadata_require_mutations=False)
    with pytest.raises(ValueError, match="Missing metadata"):
        population(cell_states=pd.DataFrame({"s": ["b"], "g": ["H"], "state": ["unresolved"]}))


def test_existing_sample_filter_also_precedes_missing():
    result = population(filter_mutations_by_isin_lists=None,
                        filter_samples_by_isin_lists={"quality": [1]})
    assert result.samples == ["a"]


def test_nested_mapping_proxies_materialized_independently():
    nested = {"min": 0}
    options = OncoplotOptions(metadata_numeric_scales=MappingProxyType({"age": MappingProxyType(nested)}))
    clone = coerce_options(options)
    clone.metadata_numeric_scales["age"]["min"] = 2
    assert nested == {"min": 0}
    assert coerce_options(OncoplotOptions(metadata_numeric_scales=MappingProxyType({}))).metadata_numeric_scales == {}
    mapped = coerce_options(MappingProxyType({"metadata_styles": MappingProxyType({
        "age": MappingProxyType({"labels": MappingProxyType({"A": "Label"})})})}))
    assert mapped.metadata_styles["age"]["labels"] == {"A": "Label"}


def test_subclass_fields_are_not_forwarded_or_mutated():
    @dataclass
    class Extended(OncoplotOptions):
        custom: dict = field(default_factory=lambda: {"value": [1]})
    original = Extended(width=800)
    clone = coerce_options(original)
    assert type(clone) is OncoplotOptions and clone.width == 800
    assert original.custom == {"value": [1]}
    original.width = float("nan")
    with pytest.raises(ValueError):
        coerce_options(original)


def test_decimal_angle_baseline_compatibility():
    result = oncoplot(pd.DataFrame({"s": ["a"], "g": ["G"]}), gene_col="g", sample_col="s",
        backend="matplotlib", options=OncoplotOptions(sample_id_angle=Decimal(45)))
    try:
        assert result.prepared_data.samples == ["a"]
        assert coerce_options(OncoplotOptions(sample_id_angle=Decimal(45))).sample_id_angle == Decimal(45)
    finally:
        plt.close(result.figure)
    for value in (Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ValueError):
            OncoplotOptions(sample_id_angle=value)


def test_literal_pr16_defaults_snapshot():
    # Captured from merged PR16, not generated from the implementation under test.
    expected = {'width': 1200, 'height': 650, 'selection_type': 'none', 'tmb_height_ratio': 0.14, 'gene_bar_width_ratio': 0.18, 'metadata_height_ratio': 0.18, 'metadata_position': 'bottom', 'buffer_metadata': 0.08, 'buffer_tmb': 0.08, 'buffer_gene_bar': 0.02, 'x_label': 'Sample', 'y_label': 'Gene', 'show_sample_ids': False, 'sample_id_position': 'bottom', 'sample_id_angle': 90, 'show_x_label': False, 'show_y_label': False, 'show_tmb_y_label': False, 'show_legend': True, 'show_legend_titles': True, 'mutation_legend_position': 'bottom', 'show_metadata_legends': True, 'metadata_legend_position': 'right', 'legend_key_size': 1.0, 'legend_offsets': {}, 'metadata_legend_nrow': None, 'metadata_legend_ncol': None, 'metadata_legend_key_size': 1.0, 'tile_height': 1.0, 'tile_width': 1.0, 'tile_linewidth': 0.25, 'row_separator_linewidth': 0.8, 'background_color': '#E5E5E5', 'unspecified_mutation_color': '#1A1A1A', 'multi_hit_color': 'black', 'font_size_x_label': 26, 'font_size_y_label': 26, 'font_size_genes': 12, 'font_size_samples': 9, 'font_size_gene_bar_axis': 10, 'font_size_tmb_axis': 10, 'font_size_metadata': 10, 'font_size_metadata_bar_numbers': 8, 'font_size_legend_text': None, 'font_size_legend_title': None, 'font_size_metadata_legend_text': None, 'font_size_metadata_legend_title': None, 'font_size_title': 14, 'font_size_subplot_title': 12, 'font_size_pathway': None, 'gene_name_x_offset': 0.0, 'main_grid_rows_label_x_offset': 10.0, 'font_family': 'Arial', 'gene_font_style': 'normal', 'sample_font_style': 'normal', 'font_style_metadata': 'normal', 'log10_transform_tmb': True, 'scientific_tmb': False, 'show_gene_bar_axis': True, 'show_tmb_axis': True, 'gene_bar_mode': 'counts', 'show_gene_bar_labels': False, 'gene_bar_label_round': 0, 'gene_bar_label_padding': 0.24, 'gene_bar_label_nudge': 0.0, 'gene_bar_scale_breaks': None, 'gene_bar_scale_n_breaks': None, 'pathway_text_color': 'white', 'pathway_background_color': '#1A1A1A', 'pathway_outline_color': 'black', 'pathway_text_angle': 0, 'prettify_legend_titles': True, 'prettify_legend_values': True, 'legend_label_max_chars': None, 'legend_title_max_chars': None, 'title_text': None, 'main_subplot_title': None, 'tmb_subplot_title': None, 'gene_bar_subplot_title': None, 'metadata_subplot_title': None, 'metadata_na_marker': '!', 'metadata_na_marker_size': 7, 'metadata_max_levels': 40, 'metadata_numeric_plot_type': 'heatmap', 'metadata_legend_orientation_heatmap': 'vertical', 'metadata_default_colors': ('#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494'), 'compact_layout': False, 'sample_id_padding': 4.0, 'tmb_bar_color': None, 'tmb_bar_width': 1.0, 'tmb_tick_interval': None, 'tmb_y_label': None, 'gene_bar_label_format': 'percent', 'tile_edgecolor': 'white', 'row_separator_color': '#666666', 'title_gap_points': 12.0, 'metadata_track_heights': {}, 'metadata_track_spacing': 0.0, 'metadata_numeric_scales': {}, 'legend_label_overrides': {}, 'legend_title_overrides': {}, 'gene_bar_color': None, 'gene_bar_x_label': None, 'legend_gap_points': 6.0, 'gene_bar_label_position': 'bar_end', 'tmb_y_label_angle': None, 'metadata_legend_order': None, 'metadata_legend_alignment': {}}
    current = OncoplotOptions()
    assert {name: getattr(current, name) for name in expected} == expected
    from pyoncoplot._utils import prettify
    assert current.prettify_function is prettify


def test_state_only_sample_uses_supplied_metadata_before_display_pruning():
    result = population(metadata=pd.DataFrame({"s": ["a", "b"], "age": [20, 30]}),
        cell_states=pd.DataFrame({"s": ["b"], "g": ["G"], "state": ["unresolved"]}))
    assert set(result.samples) == {"a", "b"}
    assert set(result.cell_states.Sample) == {"b"}
