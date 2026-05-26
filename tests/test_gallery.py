import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from python_refactor_goal_sources.recreate_gallery import (
    CONFIG_PATH,
    GALLERY_CONFIG,
    GALLERY_PRESETS,
    GENERATED_ROOT,
    GOAL_PLOTS,
    INPUTS,
    MULTIMODAL_MAX_MARKER_SCALE,
    MULTIMODAL_POINT_SIZE,
    MULTIMODAL_REFERENCE_OUTPUT_WIDTH,
    MULTIMODAL_SELECTED_LINEWIDTH,
    MULTIMODAL_SELECTED_POINT_SIZE,
    _load_brca,
    _multimodal_marker_style,
    _multimodal_panel_title_font_size,
    _multimodal_title_font_size,
    _options_from_params,
    _readme_title_font_size,
    _sample_order_by_mutation_rank,
    _weighted_row_starts,
    render_preset,
)


SOURCE_INFO = Path(__file__).resolve().parents[1] / "python_refactor_goal_sources" / "source_info_for_training.md"
OLD_GOAL_PLOT_NAMES = {
    "oncoplot.png",
    "customized_oncoplot_1.png",
    "customized_oncoplot_2.png",
    "73b90cf5-8b0b-4787-b38e-f0b1a2d6.png",
    "9c4c2e0f-74b4-4992-82be-57048160.png",
    "7a1ffb71-a4d9-4f42-8cf8-d992b037.png",
    "how-to-go-about-making-a-plot-like-this-v0-7ixn5xszvh3d1.webp",
    "vcf_sv.png",
}

EXCLUDED_IMPORTED_ASSET_PATTERNS = {
    "logo",
    "paper.pdf",
    "ggoncoplotTable",
    "paper.md",
    "paper.bib",
}


REQUIRED_FIXTURES = {
    "aml_mutations.tsv": {"sample", "gene", "mutation_type", "tooltip", "cluster"},
    "aml_metadata.tsv": {"sample", "FAB_classification", "days_to_last_followup", "Overall_Survival_Status"},
    "aml_tmb.tsv": {"sample", "mutation_type", "mutations"},
    "brca_mutations.tsv": {"sample", "gene", "mutation_type", "tooltip", "subtype"},
    "brca_metadata.tsv": {"sample", "Subtype", "ER_status", "PR_status", "HER2_status", "Age", "Age_years"},
    "brca_tmb.tsv": {"sample", "mutation_type", "mutations"},
    "cssc_mutations.tsv": {"sample", "gene", "alteration", "tooltip"},
    "cssc_tmb.tsv": {"sample", "alteration", "mutations"},
    "gbm_clinical_tracks.tsv": {"sample", "Recurrence", "Primary", "Sex", "GITS subtype R1"},
    "gbm_events.tsv": {"sample", "gene", "alteration", "tooltip"},
    "sv_depth.tsv": {"sample", "title", "position", "depth"},
    "sv_allele_fraction.tsv": {"sample", "position", "allele", "allele_fraction"},
    "sv_gene_models.tsv": {"gene", "start", "end", "strand"},
    "ggoncoplot_readme_mutations.tsv": {"sample", "gene", "mutation_type", "tooltip", "cohort"},
    "ggoncoplot_readme_metadata.tsv": {"sample", "Subtype", "ER_status", "PR_status", "HER2_status", "TMB_score"},
    "ggoncoplot_readme_tmb.tsv": {"sample", "mutation_type", "mutations"},
    "paper_multimodal_samples.tsv": {"sample", "selected"},
    "paper_multimodal_points.tsv": {"sample", "classification", "tsne_x", "tsne_y", "umap_x", "umap_y", "selected"},
    "paper_multimodal_events.tsv": {"sample", "gene", "alteration", "tooltip"},
    "paper_multimodal_clinical.tsv": {"sample", "Classification", "PR_status", "ER_status", "HER2_status"},
    "paper_multimodal_selection.tsv": {"vertex", "x", "y"},
    "ggoncoplot_comparison_table.tsv": {"package", "tidy input", "interactive", "linked selection", "metadata", "auto palette", "notes"},
}


@pytest.fixture(scope="module")
def rendered_clean_gallery(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("clean_gallery")
    for name in GALLERY_PRESETS:
        render_preset(name, output_dir)
    return output_dir


def test_synthetic_fixture_files_exist_and_load():
    for filename, required_columns in REQUIRED_FIXTURES.items():
        path = INPUTS / filename
        assert path.exists(), filename
        df = pd.read_csv(path, sep="\t")
        assert required_columns.issubset(df.columns), filename
        assert len(df) > 0, filename

    for filename in ("aml_palette.json", "brca_palette.json"):
        palette = json.loads((INPUTS / filename).read_text(encoding="utf-8"))
        assert "Missense_Mutation" in palette
        assert "Multi_Hit" in palette

    cssc_palette = json.loads((INPUTS / "cssc_palette.json").read_text(encoding="utf-8"))
    assert "missense_variant" in cssc_palette
    assert "complex_substitution" in cssc_palette

    gbm_palette = json.loads((INPUTS / "gbm_palette.json").read_text(encoding="utf-8"))
    assert {"tracks", "alterations"}.issubset(gbm_palette)
    assert "Classical" in gbm_palette["tracks"]
    assert "Mutation" in gbm_palette["alterations"]

    readme_palette = json.loads((INPUTS / "ggoncoplot_readme_palette.json").read_text(encoding="utf-8"))
    assert "missense" in readme_palette
    assert "Multi_Hit" in readme_palette

    multimodal_palette = json.loads((INPUTS / "paper_multimodal_palette.json").read_text(encoding="utf-8"))
    assert "Triple Negative" in multimodal_palette
    assert "Selected" in multimodal_palette


def test_goal_source_paths_and_numbered_reference_plots():
    assert INPUTS.name == "syntheitic_goal_data"
    assert GOAL_PLOTS.name == "goal_plots"
    assert GENERATED_ROOT.name == "generated_plots"
    assert GOAL_PLOTS.parent.name == "python_refactor_goal_sources"
    goal_plot_numbers = sorted(
        int(match.group(1))
        for path in GOAL_PLOTS.glob("goal_plot_*.png")
        if (match := re.fullmatch(r"goal_plot_(\d+)\.png", path.name))
    )
    assert goal_plot_numbers == list(range(1, max(goal_plot_numbers) + 1))
    assert max(goal_plot_numbers) >= 21
    assert not (OLD_GOAL_PLOT_NAMES & {path.name for path in GOAL_PLOTS.iterdir() if path.is_file()})
    assert not any(
        pattern in path.name
        for path in GOAL_PLOTS.iterdir()
        if path.is_file()
        for pattern in EXCLUDED_IMPORTED_ASSET_PATTERNS
    )


def test_goal_plots_are_valid_nonempty_pngs():
    for path in GOAL_PLOTS.glob("goal_plot_*.png"):
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.size[0] > 0
            assert image.size[1] > 0
        assert path.stat().st_size > 0


def test_source_info_mentions_every_goal_plot_and_sources_new_imports():
    text = SOURCE_INFO.read_text(encoding="utf-8")
    for pattern in EXCLUDED_IMPORTED_ASSET_PATTERNS:
        assert pattern not in text
    for path in GOAL_PLOTS.glob("goal_plot_*.png"):
        assert f"`{path.name}`" in text

    rows = {}
    for line in text.splitlines():
        match = re.match(r"^\| `goal_plot_(\d+)\.png` \| ([^|]+) \|", line)
        if match:
            rows[int(match.group(1))] = line

    assert rows
    for number, line in rows.items():
        if number >= 9:
            assert "| resolved |" in line
            assert "https://" in line
        else:
            assert "| unresolved |" not in line or number in {5, 6, 7}


def test_gallery_config_loads_and_declares_enabled_runs():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "gallery_params" in config
    runs = GALLERY_CONFIG["plot_runs"]
    required = {"renderer", "output_name", "goal_plot", "expected_size", "params"}
    assert {"aml", "brca", "cssc", "gbm", "sv", "readme", "multimodal", "comparison_table"}.issubset(GALLERY_CONFIG["input_files"])
    for name, run in runs.items():
        if not run.get("run", True):
            continue
        assert required.issubset(run), name

    clean_names = [preset.output_name for preset in GALLERY_PRESETS.values()]
    assert clean_names == [f"gen.goal_plot_{index}.png" for index in range(1, 22)]
    comparison_runs = GALLERY_CONFIG["comparison_runs"]
    assert comparison_runs["brca_large"]["output_name"] == "compare.goal_plot_4.png"
    assert comparison_runs["brca_large"]["expected_size"] == [1240, 398]
    assert comparison_runs["brca_compact_complex"]["output_name"] == "compare.goal_plot_5.png"
    assert comparison_runs["brca_compact_complex"]["expected_size"] == [1240, 398]


def test_brca_large_gallery_sample_order_uses_mutation_rank():
    genes = list(GALLERY_PRESETS["brca_large"].params["genes"])
    mutations, metadata, _, _ = _load_brca()
    order = _sample_order_by_mutation_rank(
        mutations,
        metadata,
        sample_col="sample",
        gene_col="gene",
        genes=genes,
    )
    metadata_order = metadata.sort_values(["Subtype", "Classification", "ER_status", "sample"], kind="stable")[
        "sample"
    ].astype(str).tolist()
    assert order != metadata_order

    weights = {gene: 2 ** (len(genes) - index - 1) for index, gene in enumerate(genes)}
    hits = (
        mutations[mutations["gene"].isin(genes)]
        .drop_duplicates(["sample", "gene"])
        .groupby("sample")["gene"]
        .apply(lambda values: sum(weights[gene] for gene in values.astype(str)))
    )
    scores = [int(hits.get(sample, 0)) for sample in order]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_brca_large_age_years_metadata_row_is_double_height():
    columns = list(GALLERY_PRESETS["brca_large"].params["metadata_cols"])
    row_starts, row_heights, total_height = _weighted_row_starts(columns, {"Age_years": 2.0})

    assert row_heights["Age_years"] == 2.0
    assert all(row_heights[column] == 1.0 for column in columns if column != "Age_years")
    assert total_height == len(columns) + 1
    assert row_starts["Age_years"] == len(columns) - 1


def test_multimodal_gallery_points_are_large_enough_for_exported_pngs():
    assert MULTIMODAL_POINT_SIZE >= 64
    assert MULTIMODAL_SELECTED_POINT_SIZE >= 140
    assert MULTIMODAL_SELECTED_LINEWIDTH >= 1.8

    reference_style = _multimodal_marker_style(
        {"figure_size": [MULTIMODAL_REFERENCE_OUTPUT_WIDTH / 100, 15.2]},
        dpi=100,
    )
    large_style = _multimodal_marker_style({"figure_size": [76.2, 52.04]}, dpi=100)
    assert reference_style["point_size"] == MULTIMODAL_POINT_SIZE
    assert large_style["point_size"] == MULTIMODAL_POINT_SIZE * MULTIMODAL_MAX_MARKER_SCALE**2
    assert large_style["selected_point_size"] == MULTIMODAL_SELECTED_POINT_SIZE * MULTIMODAL_MAX_MARKER_SCALE**2


def test_gallery_titles_use_python_branding_and_scale_for_large_exports():
    small_options = _options_from_params(GALLERY_PRESETS["ggoncoplot_readme_small"].params)
    large_options = _options_from_params(GALLERY_PRESETS["ggoncoplot_readme_basic"].params)
    configured_titles = {
        name: str(preset.params["title"])
        for name, preset in GALLERY_PRESETS.items()
        if "title" in preset.params
    }

    assert configured_titles["ggoncoplot_readme_small"] == "Pyoncoplot"
    assert configured_titles["ggoncoplot_readme_basic"] == "Pyoncoplot"
    assert configured_titles["ggoncoplot_readme_marginal"] == "Pyoncoplot"
    assert configured_titles["ggoncoplot_readme_metadata"] == "Pyoncoplot"
    assert configured_titles["ggoncoplot_package_mark"] == "Pyoncoplot"
    assert not any("ggoncoplot" in title.lower() for title in configured_titles.values())
    assert _readme_title_font_size(small_options) == 14
    assert _readme_title_font_size(large_options) >= 47
    assert _multimodal_title_font_size({"figure_size": [76.2, 52.04]}, dpi=100) >= 47
    assert _multimodal_panel_title_font_size({"figure_size": [76.2, 52.04]}, dpi=100) >= 27


def test_gallery_presets_write_expected_png_dimensions(rendered_clean_gallery):
    for name, preset in GALLERY_PRESETS.items():
        output = rendered_clean_gallery / preset.output_name
        assert output.exists()
        assert output.name.startswith("gen.goal_plot_")
        assert output.stat().st_size > 0
        with Image.open(output) as image:
            assert image.size == preset.expected_size


def test_new_gallery_outputs_are_generated_not_copied(rendered_clean_gallery):
    for index in range(9, 22):
        preset = next(preset for preset in GALLERY_PRESETS.values() if preset.output_name == f"gen.goal_plot_{index}.png")
        output = rendered_clean_gallery / preset.output_name
        goal = GOAL_PLOTS / f"goal_plot_{index}.png"
        assert output.read_bytes() != goal.read_bytes()


def test_new_gallery_outputs_have_expected_broad_features(rendered_clean_gallery):
    for index in (10, 11, 12, 14, 15, 17, 20, 21):
        preset = next(preset for preset in GALLERY_PRESETS.values() if preset.output_name == f"gen.goal_plot_{index}.png")
        output = rendered_clean_gallery / preset.output_name
        with Image.open(output) as image:
            assert image.size == preset.expected_size
            gray = image.convert("L")
            assert gray.getextrema()[0] != gray.getextrema()[1]
            width, height = image.size
            regions = [
                image.crop((0, 0, width // 2, height // 2)).convert("L"),
                image.crop((width // 2, 0, width, height // 2)).convert("L"),
                image.crop((0, height // 2, width // 2, height)).convert("L"),
                image.crop((width // 2, height // 2, width, height)).convert("L"),
            ]
            nonblank_regions = sum(region.getextrema()[0] != region.getextrema()[1] for region in regions)
            assert nonblank_regions >= 3


def test_accepted_aml_metadata_outputs_remain_clean_only_and_featureful(tmp_path):
    for name in ("aml_metadata_unsorted", "aml_metadata_sorted"):
        preset = GALLERY_PRESETS[name]
        assert preset.clean_only
        assert preset.feature_axes_min >= 4
        output = render_preset(name, tmp_path)
        with Image.open(output) as image:
            assert image.size == (1080, 720)
            regions = {
                "tmb": (120, 70, 760, 160),
                "matrix": (120, 190, 760, 500),
                "gene_bar": (760, 190, 900, 500),
                "metadata": (120, 500, 760, 650),
                "legend": (760, 120, 1060, 650),
            }
            for region in regions.values():
                cropped = image.crop(region).convert("L")
                extrema = cropped.getextrema()
                assert extrema[0] != extrema[1]


def test_brca_comparison_sheet_smoke(tmp_path):
    for name, output_name in {
        "brca_large": "compare.goal_plot_4.png",
        "brca_compact_complex": "compare.goal_plot_5.png",
    }.items():
        output = render_preset(name, tmp_path, style="comparison")
        assert output.exists()
        assert output.name == output_name
        assert output.stat().st_size > 0
        with Image.open(output) as image:
            assert image.size == (1240, 398)
