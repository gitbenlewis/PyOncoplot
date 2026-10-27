"""Defensive validation and figure ownership regression tests."""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pyoncoplot import OncoplotOptions, oncoplot
from pyoncoplot._options import coerce_options


@pytest.mark.parametrize("field", ["width", "height", "font_size_genes", "sample_id_angle",
                                   "gene_bar_label_nudge", "tile_width", "buffer_tmb"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_finite_geometry(field, value):
    with pytest.raises(ValueError):
        OncoplotOptions(**{field: value})


def test_copy_and_nested_revalidation():
    options = OncoplotOptions(metadata_numeric_scales={"age": {"min": 0}})
    clone = coerce_options(options)
    clone.metadata_numeric_scales["age"]["min"] = 2
    assert options.metadata_numeric_scales["age"]["min"] == 0
    options.metadata_numeric_scales["age"]["min"] = float("nan")
    with pytest.raises(ValueError):
        coerce_options(options)


def test_api_does_not_mutate_caller_offsets():
    options = OncoplotOptions()
    result = oncoplot(pd.DataFrame({"sample": ["a"], "gene": ["G"]}),
                      gene_col="gene", sample_col="sample", options=options,
                      gene_name_x_offset=12, backend="matplotlib")
    try:
        assert options.gene_name_x_offset == 0
    finally:
        plt.close(result.figure)


def test_compact_preflight_does_not_allocate(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("figure allocated before incompatible options rejected")
    monkeypatch.setattr(plt, "figure", unexpected)
    with pytest.raises(ValueError, match="manual legend_offsets"):
        oncoplot(pd.DataFrame({"sample": ["a"], "gene": ["G"]}),
                 gene_col="gene", sample_col="sample", backend="matplotlib",
                 options={"compact_layout": True, "legend_offsets": {"mutation": {"x": 1}}})


def test_renderer_failure_closes_only_owned_figure(monkeypatch):
    import pyoncoplot._matplotlib as renderer
    existing = plt.figure()
    before = set(plt.get_fignums())
    def fail(*args, **kwargs):
        raise RuntimeError("injected rendering failure")
    monkeypatch.setattr(renderer, "_finish_compact_layout", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            oncoplot(pd.DataFrame({"sample": ["a"], "gene": ["G"]}),
                     gene_col="gene", sample_col="sample", backend="matplotlib",
                     options={"compact_layout": True})
        assert set(plt.get_fignums()) == before
    finally:
        plt.close(existing)


def test_all_options_documented():
    from dataclasses import fields
    from pathlib import Path
    reference = Path("docs/options-reference.md").read_text()
    assert all(chr(96) + f.name + chr(96) in reference for f in fields(OncoplotOptions))


def test_original_positional_field_prefix():
    from dataclasses import fields
    original = ['width', 'height', 'selection_type', 'tmb_height_ratio', 'gene_bar_width_ratio', 'metadata_height_ratio', 'metadata_position', 'buffer_metadata', 'buffer_tmb', 'buffer_gene_bar', 'x_label', 'y_label', 'show_sample_ids', 'sample_id_position', 'sample_id_angle', 'show_x_label', 'show_y_label', 'show_tmb_y_label', 'show_legend', 'show_legend_titles', 'mutation_legend_position', 'show_metadata_legends', 'metadata_legend_position', 'legend_key_size', 'legend_offsets', 'metadata_legend_nrow', 'metadata_legend_ncol', 'metadata_legend_key_size', 'tile_height', 'tile_width', 'tile_linewidth', 'row_separator_linewidth', 'background_color', 'unspecified_mutation_color', 'multi_hit_color', 'font_size_x_label', 'font_size_y_label', 'font_size_genes', 'font_size_samples', 'font_size_gene_bar_axis', 'font_size_tmb_axis', 'font_size_metadata', 'font_size_metadata_bar_numbers', 'font_size_legend_text', 'font_size_legend_title', 'font_size_metadata_legend_text', 'font_size_metadata_legend_title', 'font_size_title', 'font_size_subplot_title', 'font_size_pathway', 'gene_name_x_offset', 'main_grid_rows_label_x_offset', 'font_family', 'gene_font_style', 'sample_font_style', 'font_style_metadata', 'log10_transform_tmb', 'scientific_tmb', 'show_gene_bar_axis', 'show_tmb_axis', 'gene_bar_mode', 'show_gene_bar_labels', 'gene_bar_label_round', 'gene_bar_label_padding', 'gene_bar_label_nudge', 'gene_bar_scale_breaks', 'gene_bar_scale_n_breaks', 'pathway_text_color', 'pathway_background_color', 'pathway_outline_color', 'pathway_text_angle', 'prettify_legend_titles', 'prettify_legend_values', 'prettify_function', 'legend_label_max_chars', 'legend_title_max_chars', 'title_text', 'main_subplot_title', 'tmb_subplot_title', 'gene_bar_subplot_title', 'metadata_subplot_title', 'metadata_na_marker', 'metadata_na_marker_size', 'metadata_max_levels', 'metadata_numeric_plot_type', 'metadata_legend_orientation_heatmap', 'metadata_default_colors', 'compact_layout', 'sample_id_padding', 'tmb_bar_color', 'tmb_bar_width', 'tmb_tick_interval', 'tmb_y_label', 'gene_bar_label_format', 'tile_edgecolor', 'row_separator_color', 'title_gap_points', 'metadata_track_heights', 'metadata_track_spacing', 'metadata_numeric_scales', 'legend_label_overrides', 'legend_title_overrides', 'gene_bar_color', 'gene_bar_x_label', 'legend_gap_points', 'gene_bar_label_position', 'tmb_y_label_angle', 'metadata_legend_order', 'metadata_legend_alignment']
    assert [f.name for f in fields(OncoplotOptions)][:109] == original
    defaults = OncoplotOptions()
    positional = OncoplotOptions(*(getattr(defaults, name) for name in original))
    assert positional == defaults


def test_save_failure_closes_owned_figure():
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="save must include"):
        oncoplot(pd.DataFrame({"sample": ["a"], "gene": ["G"]}),
                 gene_col="gene", sample_col="sample", backend="matplotlib", save={})
    assert set(plt.get_fignums()) == before
