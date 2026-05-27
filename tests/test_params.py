from collections import ChainMap

import pandas as pd
import pytest

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
