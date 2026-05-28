from collections import ChainMap

import pandas as pd
import pytest
from PIL import Image

from pyoncoplot import (
    OncoplotOptions,
    load_oncoplot_params,
    merge_oncoplot_params,
    oncoplot,
    prepare_oncoplot_data,
)
from pyoncoplot._matplotlib import render_matplotlib_oncoplot
from pyoncoplot._params import merge_params
from pyoncoplot._plotly import render_plotly_oncoplot


def small_df():
    return pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "gene": ["TP53", "EGFR", "PTEN"],
            "type": ["Missense_Mutation", "Frame_Shift_Del", "Nonsense_Mutation"],
            "vaf": [0.25, 0.50, 0.75],
            "vaf_abs": [0.025, 0.050, 0.075],
        }
    )


def _write_table(path, frame, **kwargs):
    frame.to_csv(path, index=False, **kwargs)
    return path


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


def test_oncoplot_params_options_can_be_overridden_with_params_argument():
    result = oncoplot(
        params={
            "data": small_df(),
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "backend": "plotly",
            "options": {"width": 400, "height": 300},
        },
        options={"width": 720, "height": 460},
    )

    assert result.figure.layout.width == 720
    assert result.figure.layout.height == 460


def test_oncoplot_save_mapping_writes_matplotlib_output(tmp_path):
    output = tmp_path / "saved-oncoplot.png"
    result = oncoplot(
        params={
            "data": small_df(),
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "backend": "matplotlib",
            "options": {"width": 480, "height": 360},
            "save": {"path": output, "dpi": 120},
        }
    )

    try:
        assert output.exists()
        with Image.open(output) as image:
            assert image.size == (480, 360)
        assert result.backend == "matplotlib"
    finally:
        import matplotlib.pyplot as plt

        plt.close(result.figure)


def test_oncoplot_params_accept_variant_value_heatmap_arguments():
    result = oncoplot(
        params={
            "data": small_df(),
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "variant_value_col": "vaf",
            "variant_value_agg": "max",
            "variant_value_missing": "blank",
            "variant_value_palette": "viridis",
            "backend": "plotly",
            "options": {"width": 720, "height": 460},
        }
    )

    assert result.prepared_data.variant_value_col == "vaf"
    assert result.prepared_data.variant_value_agg == "max"
    assert result.prepared_data.variant_value_missing == "blank"
    assert result.figure.layout.width == 720


def test_oncoplot_params_accept_multi_row_main_grid_arguments():
    result = oncoplot(
        params={
            "data": small_df(),
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "main_grid_rows": [
                {"kind": "mutation_type", "label": "Variant type"},
                {"kind": "variant_value", "column": "vaf", "label": "VAF"},
                {"kind": "variant_value", "column": "vaf_abs", "label": "VAF abs", "palette": "magma", "missing": "zero"},
            ],
            "variant_value_scale": "per_column",
            "gene_name_x_offset": 6,
            "main_grid_rows_label_x_offset": 12,
            "backend": "plotly",
        }
    )

    assert result.prepared_data.main_grid_mode == "expanded"
    assert result.prepared_data.main_grid_rows["Label"].drop_duplicates().tolist() == [
        "Variant type",
        "VAF",
        "VAF abs",
    ]
    assert result.prepared_data.main_grid_rows["VariantValueMissing"].dropna().drop_duplicates().tolist() == [
        "blank",
        "zero",
    ]
    assert next(annotation for annotation in result.figure.layout.annotations if annotation.text == "TP53").xshift == -60
    assert next(annotation for annotation in result.figure.layout.annotations if annotation.text == "VAF").xshift == -20
    assert result.prepared_data.main_grid_tiles["Kind"].tolist()


def test_oncoplot_params_accept_sample_and_mutation_filters():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "group": ["A", "B", "A"],
            "score": [1.0, 1.5, 3.0],
        }
    )

    result = oncoplot(
        params={
            "data": small_df(),
            "metadata": metadata,
            "metadata_cols": ["group", "score"],
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "filter_mutations_by_isin_lists": {"type": ["Missense_Mutation", "Nonsense_Mutation"]},
            "filter_mutations_by_greater_than": {"vaf": 0.20},
            "filter_mutations_by_less_than": {"vaf": 0.80},
            "filter_samples_by_isin_lists": {"group": ["A"]},
            "filter_samples_by_greater_than": {"score": 0.50},
            "filter_samples_by_less_than": {"score": 2.00},
            "backend": "plotly",
        }
    )

    assert result.prepared_data.samples == ["S1"]
    assert result.prepared_data.genes == ["TP53"]


def test_merge_oncoplot_params_supports_unpacked_kwargs_override():
    merged = merge_oncoplot_params(
        {
            "data": small_df(),
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "top_n": 1,
            "backend": "plotly",
            "options": {"width": 400, "height": 300},
        },
        top_n=2,
        options={"width": 750, "height": 500},
    )

    result = oncoplot(**merged)

    assert merged["options"] == {"width": 750, "height": 500}
    assert len(result.prepared_data.genes) == 2
    assert result.figure.layout.width == 750


def test_oncoplot_accepts_unpacked_params_without_overrides():
    result = oncoplot(
        **{
            "data": small_df(),
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "backend": "plotly",
            "options": {"width": 680, "height": 420},
        }
    )

    assert result.figure.layout.width == 680
    assert result.figure.layout.height == 420


def test_chainmap_supports_unpacked_params_with_overrides():
    params = {
        "data": small_df(),
        "gene_col": "gene",
        "sample_col": "sample",
        "mutation_type_col": "type",
        "backend": "plotly",
        "options": {"width": 400, "height": 300},
    }

    result = oncoplot(**ChainMap({"options": {"width": 760, "height": 510}}, params))

    assert result.figure.layout.width == 760
    assert result.figure.layout.height == 510


def test_oncoplot_params_accept_data_and_metadata_paths(tmp_path):
    data_path = _write_table(tmp_path / "mutations.csv", small_df())
    metadata_path = _write_table(
        tmp_path / "metadata.csv",
        pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]}),
    )

    result = oncoplot(
        params={
            "data": data_path,
            "metadata": metadata_path,
            "metadata_cols": ["group"],
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "backend": "plotly",
        }
    )

    assert result.prepared_data.metadata is not None
    assert result.prepared_data.metadata_cols == ["group"]


def test_oncoplot_params_accept_tmb_and_pathway_paths(tmp_path):
    data_path = _write_table(tmp_path / "mutations.csv", small_df())
    tmb_path = _write_table(
        tmp_path / "tmb.csv",
        pd.DataFrame({"sample": ["S1", "S2", "S3"], "mutations": [4, 5, 6]}),
    )
    pathway_path = _write_table(
        tmp_path / "pathway.csv",
        pd.DataFrame({"gene": ["TP53", "EGFR", "PTEN"], "pathway": ["Cell cycle", "RTK", "PI3K"]}),
    )

    result = oncoplot(
        params={
            "data": data_path,
            "tmb_data": tmb_path,
            "pathway": pathway_path,
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "draw_tmb_bar": True,
            "include_genes": ["TP53", "EGFR", "PTEN"],
            "backend": "matplotlib",
        }
    )

    assert result.prepared_data.tmb_is_custom is True
    assert result.prepared_data.pathway_by_gene == {"TP53": "Cell cycle", "EGFR": "RTK", "PTEN": "PI3K"}


def test_load_oncoplot_params_extracts_nested_key_and_resolves_paths(tmp_path):
    data_dir = tmp_path / "files"
    data_dir.mkdir()
    _write_table(data_dir / "mutations.csv", small_df())
    _write_table(
        data_dir / "metadata.csv",
        pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["A", "B", "A"]}),
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """
datasets:
  m15:
    plot1_params:
      data: files/mutations.csv
      metadata: files/metadata.csv
      metadata_cols: [group]
      gene_col: gene
      sample_col: sample
      mutation_type_col: type
      variant_value_cols: [vaf, vaf_abs]
      variant_value_scale: shared
      backend: plotly
      top_n: 1
""",
        encoding="utf-8",
    )

    params = load_oncoplot_params(config, key="datasets.m15.plot1_params")
    assert isinstance(params["data"], pd.DataFrame)
    result = oncoplot(params=config, params_key="datasets.m15.plot1_params", top_n=2)

    assert len(result.prepared_data.genes) == 2
    assert result.prepared_data.metadata is not None
    assert result.prepared_data.variant_value_cols == ["vaf", "vaf_abs"]
    assert result.prepared_data.main_grid_mode == "expanded"


def test_table_read_spec_passes_read_csv_kwargs(tmp_path):
    table_path = _write_table(tmp_path / "mutations.txt", small_df(), sep="\t")

    result = oncoplot(
        params={
            "data": {"path": table_path, "sep": "\t"},
            "gene_col": "gene",
            "sample_col": "sample",
            "mutation_type_col": "type",
            "backend": "plotly",
        }
    )

    assert set(result.prepared_data.samples) == {"S1", "S2", "S3"}


def test_config_loading_errors_are_clear(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("datasets:\n  m15:\n    plot1_params: 5\n", encoding="utf-8")
    with pytest.raises(KeyError, match="datasets.m16.plot1_params"):
        load_oncoplot_params(config, key="datasets.m16.plot1_params")
    with pytest.raises(TypeError, match="must resolve to a mapping"):
        load_oncoplot_params(config, key="datasets.m15.plot1_params")

    root = tmp_path / "root.yaml"
    root.write_text("- not\n- mapping\n", encoding="utf-8")
    with pytest.raises(TypeError, match="root must be a mapping"):
        load_oncoplot_params(root)

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        oncoplot(
            params={
                "data": tmp_path / "missing.csv",
                "gene_col": "gene",
                "sample_col": "sample",
            }
        )

    with pytest.raises(ValueError, match="must include a 'path'"):
        oncoplot(
            params={
                "data": {"sep": "\t"},
                "gene_col": "gene",
                "sample_col": "sample",
            }
        )


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
