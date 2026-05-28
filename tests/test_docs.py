import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED_DOCS = [
    "index.md",
    "installation.md",
    "quickstart.md",
    "data-inputs.md",
    "api-reference.md",
    "options-reference.md",
    "palettes.md",
    "metadata-and-tmb.md",
    "pathways-and-sorting.md",
    "rendering-backends.md",
    "gallery.md",
    "examples/basic.md",
    "examples/metadata.md",
    "examples/brca-gallery.md",
    "examples/structural-variation-panel.md",
    "migration-from-ggoncoplot.md",
    "development.md",
    "troubleshooting.md",
]


def test_required_docs_exist_and_have_headings():
    for relative_path in REQUIRED_DOCS:
        path = DOCS / relative_path
        assert path.exists(), relative_path
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# "), relative_path
        assert "TODO" not in text


def test_docs_have_core_code_examples():
    pages_with_code = [
        "index.md",
        "installation.md",
        "quickstart.md",
        "data-inputs.md",
        "api-reference.md",
        "gallery.md",
        "examples/basic.md",
        "examples/metadata.md",
    ]
    for relative_path in pages_with_code:
        text = (DOCS / relative_path).read_text(encoding="utf-8")
        assert "```" in text, relative_path


def test_local_markdown_links_resolve():
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path or target_path.startswith("`"):
                continue
            resolved = (path.parent / target_path).resolve()
            assert resolved.exists(), f"{path.relative_to(ROOT)} links to missing {target}"


def test_current_docs_do_not_reference_old_example_oncoplots_tree():
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "example_oncoplots" not in text, path.relative_to(ROOT)


def test_options_reference_mentions_parity_options():
    text = (DOCS / "options-reference.md").read_text(encoding="utf-8")
    for option in [
        "pathway_text_angle",
        "gene_bar_label_padding",
        "gene_bar_label_nudge",
        "gene_bar_scale_breaks",
        "gene_bar_scale_n_breaks",
        "buffer_metadata",
        "buffer_tmb",
        "buffer_gene_bar",
        "legend_key_size",
        "legend_offsets",
        "legend_label_max_chars",
        "legend_title_max_chars",
        "metadata_legend_nrow",
        "metadata_legend_ncol",
        "metadata_legend_key_size",
        "font_size_legend_text",
        "font_size_legend_title",
        "font_size_metadata_legend_text",
        "font_size_metadata_legend_title",
        "font_size_title",
        "font_size_subplot_title",
        "font_size_pathway",
        "gene_name_x_offset",
        "title_text",
        "main_subplot_title",
        "tmb_subplot_title",
        "gene_bar_subplot_title",
        "metadata_subplot_title",
        "metadata_na_marker_size",
        "font_style_metadata",
        "font_size_metadata_bar_numbers",
        "metadata_legend_orientation_heatmap",
        "selection_type",
        "tile_height",
        "tile_width",
        "font_size_x_label",
        "font_size_y_label",
        "prettify_legend_titles",
        "prettify_legend_values",
        "prettify_function",
    ]:
        assert option in text


def test_docs_describe_tooltip_default_and_precise_parity_status():
    data_inputs = (DOCS / "data-inputs.md").read_text(encoding="utf-8")
    assert "defaults to `sample_col`" in data_inputs
    assert "defaults to the mutation type column" not in data_inputs
    assert "custom TMB input" in data_inputs

    migration = (DOCS / "migration-from-ggoncoplot.md").read_text(encoding="utf-8")
    assert "supported with renderer differences" in migration
    assert "custom TMB sample columns may appear in any position" in migration

    installation = (DOCS / "installation.md").read_text(encoding="utf-8")
    assert "Installing only `pytest` is not enough" in installation


def test_docs_describe_python_first_renderer_limits():
    options = (DOCS / "options-reference.md").read_text(encoding="utf-8")
    assert "`tile_width` and `tile_height` are Matplotlib/static-layout controls" in options
    assert "`tile_linewidth` applies to marker" in options
    assert "outlines in both renderers" in options
    assert "Font face style controls" in options
    assert "`metadata_numeric_plot_type=\"bar\"`" in options
    assert "Plotly uses one shared interactive legend" in options

    metadata_tmb = (DOCS / "metadata-and-tmb.md").read_text(encoding="utf-8")
    assert "coverage only when stacked subtype bars are" in metadata_tmb
    assert "rendered with `log10_transform_tmb=False`" in metadata_tmb

    palettes = (DOCS / "palettes.md").read_text(encoding="utf-8")
    assert "Default-generated mutation palettes use `OncoplotOptions.multi_hit_color`" in palettes


def test_docs_cover_remaining_parity_corrections():
    data_inputs = (DOCS / "data-inputs.md").read_text(encoding="utf-8")
    assert "more than one mutation row" in data_inputs
    assert "`metadata_require_mutations=False` is also set" in data_inputs
    assert "variant_value_missing=\"zero\"" in data_inputs
    assert "tile with only missing source values is" in data_inputs

    metadata_tmb = (DOCS / "metadata-and-tmb.md").read_text(encoding="utf-8")
    assert "metadata_require_mutations=True" in metadata_tmb
    assert "metadata_require_mutations=False" in metadata_tmb
    assert "both renderers collapse the" in metadata_tmb
    assert "emit a `UserWarning`" in metadata_tmb

    troubleshooting = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
    assert "prepare_oncoplot_data" in troubleshooting
    assert "Multi_Hit" in troubleshooting
    assert "displayed_types - set(palette)" in troubleshooting

    api_reference = (DOCS / "api-reference.md").read_text(encoding="utf-8")
    assert "Core Arguments" in api_reference
    assert "`PreparedOncoplotData` Fields" in api_reference
    assert "`tmb_sample_col`, `tmb_value_col`, `tmb_type_col`" in api_reference
    assert "main_grid_rows" in api_reference
    assert "variant_value_cols" in api_reference
    assert "variant_value_missing" in api_reference
    assert "gene_name_x_offset" in api_reference
    assert "Save Behavior" in api_reference

    rendering_backends = (DOCS / "rendering-backends.md").read_text(encoding="utf-8")
    assert "Backend Support Matrix" in rendering_backends
    assert "TMB subtype legends" in rendering_backends
    assert "`TMB: <subtype>`" in rendering_backends
