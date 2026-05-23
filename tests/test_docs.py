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
        "metadata_legend_nrow",
        "metadata_legend_ncol",
        "metadata_legend_key_size",
        "metadata_na_marker_size",
        "font_style_metadata",
        "font_size_metadata_bar_numbers",
        "metadata_legend_orientation_heatmap",
    ]:
        assert option in text
