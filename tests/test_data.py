import math
from pathlib import Path

import pandas as pd
import pytest

from pyoncoplot import (
    check_valid_dataframe_columns,
    identify_top_genes,
    prepare_oncoplot_data,
    rank_genes_by_pathway,
    score_sample_by_gene_rank,
)


def mutation_df():
    return pd.DataFrame(
        {
            "sample": ["S1", "S1", "S1", "S2", "S2", "S3", "S4", "S5"],
            "gene": ["TP53", "TP53", "EGFR", "TP53", "PTEN", "PTEN", "EGFR", "ALK"],
            "type": [
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Missense_Mutation",
                "Missense_Mutation",
                "Frame_Shift_Del",
                "Frame_Shift_Del",
                "Missense_Mutation",
                "Splice_Site",
            ],
            "tooltip": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )


def test_check_valid_dataframe_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    check_valid_dataframe_columns(df, ["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        check_valid_dataframe_columns(df, ["missing"])


def test_identify_top_genes_counts_distinct_samples_and_breaks_ties():
    df = mutation_df()
    assert identify_top_genes(df, gene_col="gene", sample_col="sample", top_n=3) == [
        "EGFR",
        "PTEN",
        "TP53",
    ]


def test_identify_top_genes_can_return_ties_and_all_genes():
    df = mutation_df()
    tied = identify_top_genes(
        df,
        gene_col="gene",
        sample_col="sample",
        top_n=2,
        return_extra_genes_if_tied=True,
    )
    assert tied == ["EGFR", "PTEN", "TP53"]
    assert len(identify_top_genes(df, "gene", "sample", top_n=math.inf)) == 4


def test_score_sample_by_gene_rank_matches_r_examples():
    assert score_sample_by_gene_rank(["TERT", "EGFR", "PTEN", "BRCA2"], ["EGFR", "BRCA2"], [1, 2]) == 3
    assert score_sample_by_gene_rank(["TERT", "EGFR", "PTEN", "IDH1"], ["EGFR", "BRCA2"], [1, 2]) == 1
    assert score_sample_by_gene_rank(["TERT", "IDH1", "PTEN", "BRCA2"], ["EGFR", "BRCA2"], [1, 2]) == 2


def test_prepare_oncoplot_data_collapses_multi_hits_and_tooltips():
    prepared = prepare_oncoplot_data(
        mutation_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        tooltip_col="tooltip",
        top_n=3,
    )
    hit = prepared.tiles[(prepared.tiles["Sample"].astype(str) == "S1") & (prepared.tiles["Gene"].astype(str) == "TP53")].iloc[0]
    assert hit["MutationType"] == "Multi_Hit"
    assert hit["MutationCount"] == 2
    assert hit["Tooltip"] == "Sample: S1<br>a<br>b"
    assert prepared.genes == ["EGFR", "PTEN", "TP53"]
    assert prepared.samples[0] == "S1"
    assert set(prepared.mutation_counts.columns) == {"Gene", "MutationType", "Count"}


def test_prepare_oncoplot_data_generates_default_mutation_tooltips():
    prepared = prepare_oncoplot_data(
        mutation_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        include_genes=["TP53", "EGFR"],
    )

    multi_hit = prepared.tiles[
        (prepared.tiles["Sample"].astype(str) == "S1")
        & (prepared.tiles["Gene"].astype(str) == "TP53")
    ].iloc[0]
    single_hit = prepared.tiles[
        (prepared.tiles["Sample"].astype(str) == "S1")
        & (prepared.tiles["Gene"].astype(str) == "EGFR")
    ].iloc[0]

    assert multi_hit["Tooltip"] == (
        "Sample: S1<br>TP53: Missense_Mutation<br>TP53: Nonsense_Mutation"
    )
    assert single_hit["Tooltip"] == "Sample: S1<br>EGFR: Missense_Mutation"


def test_prepare_oncoplot_data_default_expanded_tooltips_include_variant_values():
    df = mutation_df()
    df["vaf_pct"] = [12, 42, 20, 35, 51, 62, 25, 15]

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        include_genes=["TP53", "EGFR"],
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf_pct", "label": "VAF %"},
        ],
    )
    tp53_s1 = prepared.main_grid_tiles[
        (prepared.main_grid_tiles["Sample"] == "S1")
        & (prepared.main_grid_tiles["Gene"] == "TP53")
    ].sort_values("TrackIndex")

    assert tp53_s1.iloc[0]["Tooltip"] == (
        "Sample: S1<br>TP53: Missense_Mutation<br>TP53: Nonsense_Mutation<br>VAF %: 42"
    )
    assert tp53_s1.iloc[1]["Tooltip"] == (
        "Sample: S1<br>TP53: Missense_Mutation<br>TP53: Nonsense_Mutation<br>VAF %: 42"
    )


def test_prepare_oncoplot_data_aggregates_variant_values_for_collapsed_tiles():
    df = mutation_df()
    df["vaf"] = [0.12, 0.42, 0.20, 0.35, 0.51, 0.62, 0.25, 0.15]
    expected = {
        "max": 0.42,
        "mean": 0.27,
        "median": 0.27,
        "min": 0.12,
    }

    for agg, value in expected.items():
        prepared = prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            tooltip_col="tooltip",
            include_genes=["TP53", "EGFR"],
            variant_value_col="vaf",
            variant_value_agg=agg,
        )
        hit = prepared.tiles[
            (prepared.tiles["Sample"].astype(str) == "S1")
            & (prepared.tiles["Gene"].astype(str) == "TP53")
        ].iloc[0]

        assert hit["MutationType"] == "Multi_Hit"
        assert hit["VariantValue"] == pytest.approx(value)
        assert prepared.variant_value_col == "vaf"
        assert prepared.variant_value_agg == agg
        assert prepared.variant_value_min is not None
        assert prepared.variant_value_max is not None


def test_prepare_oncoplot_data_blanks_missing_variant_values_by_default():
    df = mutation_df()
    df["vaf"] = [pd.NA, 0.42, 0.20, 0.35, 0.51, 0.62, pd.NA, 0.15]

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        tooltip_col="tooltip",
        include_genes=["TP53", "EGFR"],
        variant_value_col="vaf",
        variant_value_agg="mean",
    )

    tp53_s1 = prepared.tiles[
        (prepared.tiles["Sample"].astype(str) == "S1")
        & (prepared.tiles["Gene"].astype(str) == "TP53")
    ].iloc[0]
    egfr_s4 = prepared.tiles[
        (prepared.tiles["Sample"].astype(str) == "S4")
        & (prepared.tiles["Gene"].astype(str) == "EGFR")
    ].iloc[0]

    assert tp53_s1["VariantValue"] == pytest.approx(0.42)
    assert math.isnan(float(egfr_s4["VariantValue"]))
    assert prepared.variant_value_missing == "blank"
    assert prepared.variant_value_min == pytest.approx(0.20)
    assert prepared.variant_value_max == pytest.approx(0.42)


def test_prepare_oncoplot_data_can_fill_missing_variant_values_with_zero():
    df = mutation_df()
    df["vaf"] = [None, 0.42, 0.20, 0.35, 0.51, 0.62, None, 0.15]

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        tooltip_col="tooltip",
        include_genes=["TP53", "EGFR"],
        variant_value_col="vaf",
        variant_value_agg="mean",
        variant_value_missing="zero",
    )

    tp53_s1 = prepared.tiles[
        (prepared.tiles["Sample"].astype(str) == "S1")
        & (prepared.tiles["Gene"].astype(str) == "TP53")
    ].iloc[0]
    egfr_s4 = prepared.tiles[
        (prepared.tiles["Sample"].astype(str) == "S4")
        & (prepared.tiles["Gene"].astype(str) == "EGFR")
    ].iloc[0]

    assert tp53_s1["VariantValue"] == pytest.approx(0.21)
    assert egfr_s4["VariantValue"] == pytest.approx(0.0)
    assert prepared.variant_value_missing == "zero"


def test_prepare_oncoplot_data_builds_multi_row_main_grid_tracks():
    df = mutation_df()
    df["vaf_pct"] = [12, 42, 20, 35, 51, 62, 25, 15]
    df["vaf_abs"] = [0.12, 0.42, 0.20, 0.35, 0.51, 0.62, 0.25, 0.15]

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        tooltip_col="tooltip",
        include_genes=["TP53", "EGFR"],
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf_pct", "label": "VAF %", "agg": "mean"},
            {"kind": "variant_value", "column": "vaf_abs", "label": "VAF abs", "agg": "max"},
        ],
    )

    assert prepared.main_grid_mode == "expanded"
    assert prepared.variant_value_col is None
    assert len(prepared.main_grid_rows) == len(prepared.genes) * 3
    assert prepared.main_grid_rows["Label"].drop_duplicates().tolist() == ["Variant type", "VAF %", "VAF abs"]

    tp53_s1 = prepared.main_grid_tiles[
        (prepared.main_grid_tiles["Sample"] == "S1") & (prepared.main_grid_tiles["Gene"] == "TP53")
    ].sort_values("TrackIndex")
    assert tp53_s1["Kind"].tolist() == ["mutation_type", "variant_value", "variant_value"]
    assert tp53_s1.iloc[0]["MutationType"] == "Multi_Hit"
    assert tp53_s1.iloc[1]["VariantValue"] == pytest.approx(27.0)
    assert tp53_s1.iloc[2]["VariantValue"] == pytest.approx(0.42)


def test_prepare_oncoplot_data_variant_value_cols_use_global_missing_policy():
    df = mutation_df()
    df["vaf_pct"] = [None, 42, 20, 35, 51, 62, None, 15]
    df["delta_vaf_pct"] = [None, 8, 2, 1, -3, 4, None, 9]

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        include_genes=["TP53", "EGFR"],
        variant_value_cols=["vaf_pct", "delta_vaf_pct"],
        variant_value_agg="mean",
        variant_value_missing="zero",
    )

    tp53_s1 = prepared.main_grid_tiles[
        (prepared.main_grid_tiles["Sample"] == "S1") & (prepared.main_grid_tiles["Gene"] == "TP53")
    ].sort_values("TrackIndex")
    egfr_s4 = prepared.main_grid_tiles[
        (prepared.main_grid_tiles["Sample"] == "S4") & (prepared.main_grid_tiles["Gene"] == "EGFR")
    ].sort_values("TrackIndex")

    assert tp53_s1["VariantValue"].tolist()[1:] == pytest.approx([21.0, 4.0])
    assert egfr_s4["VariantValue"].tolist()[1:] == pytest.approx([0.0, 0.0])
    assert set(prepared.main_grid_rows["VariantValueMissing"].dropna()) == {"zero"}


def test_prepare_oncoplot_data_main_grid_rows_override_missing_policy():
    df = mutation_df()
    df["vaf_pct"] = [None, 42, 20, 35, 51, 62, None, 15]
    df["vaf_abs"] = [None, 0.42, 0.20, 0.35, 0.51, 0.62, None, 0.15]

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        include_genes=["TP53", "EGFR"],
        variant_value_agg="mean",
        main_grid_rows=[
            {"kind": "mutation_type", "label": "Variant type"},
            {"kind": "variant_value", "column": "vaf_pct", "label": "VAF %", "missing": "zero"},
            {"kind": "variant_value", "column": "vaf_abs", "label": "VAF abs"},
        ],
    )

    tp53_s1 = prepared.main_grid_tiles[
        (prepared.main_grid_tiles["Sample"] == "S1") & (prepared.main_grid_tiles["Gene"] == "TP53")
    ].sort_values("TrackIndex")
    egfr_s4 = prepared.main_grid_tiles[
        (prepared.main_grid_tiles["Sample"] == "S4") & (prepared.main_grid_tiles["Gene"] == "EGFR")
    ].sort_values("TrackIndex")

    assert tp53_s1["VariantValue"].tolist()[1:] == pytest.approx([21.0, 0.42])
    assert egfr_s4.iloc[1]["VariantValue"] == pytest.approx(0.0)
    assert math.isnan(float(egfr_s4.iloc[2]["VariantValue"]))
    assert prepared.main_grid_rows["VariantValueMissing"].dropna().drop_duplicates().tolist() == ["zero", "blank"]


def test_prepare_oncoplot_data_variant_value_cols_convenience_and_shared_scale():
    df = mutation_df()
    df["vaf_pct"] = [12, 42, 20, 35, 51, 62, 25, 15]
    df["delta_vaf_pct"] = [-5, 8, 2, 1, -3, 4, 7, 9]

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        include_genes=["TP53", "EGFR"],
        variant_value_cols=["vaf_pct", "delta_vaf_pct"],
        variant_value_scale="shared",
    )

    assert prepared.variant_value_cols == ["vaf_pct", "delta_vaf_pct"]
    assert prepared.main_grid_rows["Label"].drop_duplicates().tolist() == [
        "Variant type",
        "vaf_pct",
        "delta_vaf_pct",
    ]
    value_rows = prepared.main_grid_rows[prepared.main_grid_rows["Kind"] == "variant_value"]
    assert set(value_rows["VariantValueMin"].dropna()) == {1.0}
    assert set(value_rows["VariantValueMax"].dropna()) == {42.0}
    assert set(value_rows["ScaleGroup"]) == {"variant_value_shared"}


def test_prepare_oncoplot_data_validates_variant_value_inputs():
    df = mutation_df()
    df["vaf"] = [0.12, 0.42, 0.20, 0.35, 0.51, 0.62, 0.25, 0.15]

    with pytest.raises(ValueError, match="variant_value_agg"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            variant_value_col="vaf",
            variant_value_agg="sum",
        )

    non_numeric = df.copy()
    non_numeric["vaf"] = non_numeric["vaf"].astype(str)
    with pytest.raises(ValueError, match="numeric"):
        prepare_oncoplot_data(
            non_numeric,
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            variant_value_col="vaf",
        )

    with pytest.raises(ValueError, match="variant_value_missing"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            variant_value_col="vaf",
            variant_value_missing="drop",
        )

    with pytest.raises(ValueError, match="variant_value_scale"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            variant_value_cols=["vaf"],
            variant_value_scale="global",
        )

    with pytest.raises(ValueError, match="Row-level palettes"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            main_grid_rows=[
                {"kind": "variant_value", "column": "vaf", "palette": "magma"},
            ],
            variant_value_scale="shared",
        )

    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            variant_value_col="vaf",
            variant_value_cols=["vaf"],
        )


def test_prepare_oncoplot_data_validates_empty_ids_and_metadata_duplicates():
    df = mutation_df()
    bad = df.copy()
    bad.loc[0, "sample"] = ""
    with pytest.raises(ValueError, match="Sample column"):
        prepare_oncoplot_data(bad, gene_col="gene", sample_col="sample")

    metadata = pd.DataFrame({"sample": ["S1", "S1"], "sex": ["F", "M"]})
    with pytest.raises(ValueError, match="unique"):
        prepare_oncoplot_data(df, gene_col="gene", sample_col="sample", metadata=metadata)


def test_prepare_oncoplot_data_derives_metadata_from_mutation_data():
    df = mutation_df()
    df["clinical_group"] = df["sample"].map({"S1": "A", "S2": "B", "S3": "A", "S4": "C", "S5": "B"})
    df["purity"] = df["sample"].map({"S1": 0.72, "S2": 0.61, "S3": 0.83, "S4": 0.55, "S5": 0.68})

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata_cols=["clinical_group", "purity"],
        show_all_samples=True,
    )

    assert prepared.metadata is not None
    assert prepared.metadata_cols == ["clinical_group", "purity"]
    assert list(prepared.metadata.columns) == ["Sample", "clinical_group", "purity"]
    assert prepared.metadata["Sample"].astype(str).tolist() == prepared.samples
    assert len(prepared.metadata) == df["sample"].nunique()
    assert prepared.metadata_tracks is not None
    assert [track.column for track in prepared.metadata_tracks] == ["clinical_group", "purity"]
    assert [track.kind for track in prepared.metadata_tracks] == ["categorical", "numeric"]


def test_prepare_oncoplot_data_rejects_conflicting_derived_metadata_values():
    df = mutation_df()
    df["clinical_group"] = "A"
    df.loc[1, "clinical_group"] = "B"

    with pytest.raises(ValueError, match="unique"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            metadata_cols=["clinical_group"],
        )


def test_mutation_filters_are_row_level_and_affect_top_gene_ranking():
    df = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "gene": ["AAA", "AAA", "AAA", "BBB", "BBB", "CCC"],
            "type": ["keep", "drop", "drop", "keep", "keep", "keep"],
            "vaf": [0.90, 0.90, 0.90, 0.70, 0.80, 0.20],
        }
    )

    unfiltered = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        top_n=1,
    )
    filtered = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        top_n=1,
        filter_mutations_by_isin_lists={"type": ["keep"]},
        filter_mutations_by_greater_than={"vaf": 0.50},
        filter_mutations_by_less_than={"vaf": 0.95},
    )

    assert unfiltered.genes == ["AAA"]
    assert filtered.genes == ["BBB"]
    assert filtered.samples == ["S4", "S5"]
    assert set(filtered.tiles["Gene"].astype(str)) == {"BBB"}


def test_sample_filters_use_metadata_with_and_semantics():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3", "S4", "S5"],
            "group": ["A", "B", "A", "A", "C"],
            "age": [45, 60, 61, 72, 55],
        }
    )

    prepared = prepare_oncoplot_data(
        mutation_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group", "age"],
        top_n=None,
        filter_samples_by_isin_lists={"group": ["A"]},
        filter_samples_by_greater_than={"age": 50},
        filter_samples_by_less_than={"age": 70},
    )

    assert prepared.samples == ["S3"]
    assert prepared.genes == ["PTEN"]
    assert prepared.total_samples == 1
    assert prepared.metadata["Sample"].astype(str).tolist() == ["S3"]


def test_sample_filter_source_resolution_prefers_metadata_over_main_data():
    df = mutation_df()
    df["group"] = df["sample"].map(
        {
            "S1": "main_drop",
            "S2": "main_keep",
            "S3": "main_drop",
            "S4": "main_drop",
            "S5": "main_drop",
        }
    )
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3", "S4", "S5"],
            "group": ["meta_keep", "meta_drop", "meta_drop", "meta_drop", "meta_drop"],
        }
    )

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        top_n=None,
        filter_samples_by_isin_lists={"group": ["meta_keep"]},
    )

    assert prepared.samples == ["S1"]
    assert set(prepared.tiles["Sample"].astype(str)) == {"S1"}


def test_main_data_sample_filters_require_one_row_to_satisfy_all_filters_then_keep_sample_rows():
    df = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S2", "S3"],
            "gene": ["TP53", "PTEN", "TP53", "PTEN", "EGFR"],
            "type": [
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Missense_Mutation",
            ],
            "vaf": [0.20, 0.95, 0.90, 0.10, 0.40],
        }
    )

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        top_n=None,
        filter_samples_by_isin_lists={"type": ["Missense_Mutation"]},
        filter_samples_by_greater_than={"vaf": 0.80},
    )

    assert prepared.samples == ["S2"]
    assert set(prepared.tiles["Sample"].astype(str)) == {"S2"}
    assert set(prepared.tiles["Gene"].astype(str)) == {"PTEN", "TP53"}


def test_mutation_filters_run_before_main_data_sample_filters():
    df = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S2"],
            "gene": ["TP53", "EGFR", "TP53", "PTEN"],
            "status": ["keep", "drop", "keep", "keep"],
            "cohort": ["other", "target", "target", "other"],
        }
    )

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="status",
        top_n=None,
        filter_mutations_by_isin_lists={"status": ["keep"]},
        filter_samples_by_isin_lists={"cohort": ["target"]},
    )

    assert prepared.samples == ["S2"]
    assert set(prepared.tiles["Gene"].astype(str)) == {"PTEN", "TP53"}
    assert "S1" not in set(prepared.tiles["Sample"].astype(str))


def test_derived_metadata_omits_filter_only_main_data_columns():
    df = mutation_df()
    df["clinical_group"] = df["sample"].map({"S1": "A", "S2": "B", "S3": "A", "S4": "C", "S5": "B"})
    df["cohort"] = df["sample"].map({"S1": "keep", "S2": "drop", "S3": "keep", "S4": "drop", "S5": "drop"})

    prepared = prepare_oncoplot_data(
        df,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata_cols=["clinical_group"],
        show_all_samples=True,
        filter_samples_by_isin_lists={"cohort": ["keep"]},
    )

    assert set(prepared.samples) == {"S1", "S3"}
    assert prepared.metadata_cols == ["clinical_group"]
    assert list(prepared.metadata.columns) == ["Sample", "clinical_group"]


def test_filter_validation_errors_are_clear():
    df = mutation_df()

    with pytest.raises(TypeError, match="mapping"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            filter_mutations_by_isin_lists=["type"],
        )

    with pytest.raises(TypeError, match="sequence"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            filter_samples_by_isin_lists={"type": "Missense_Mutation"},
        )

    with pytest.raises(ValueError, match="Could not find column: missing"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            filter_mutations_by_isin_lists={"missing": ["x"]},
        )

    with pytest.raises(ValueError, match="numeric filters"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            filter_mutations_by_greater_than={"type": 1},
        )

    with pytest.raises(ValueError, match="not NaN"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            filter_samples_by_less_than={"type": math.nan},
        )

    with pytest.raises(ValueError, match="zero mutation rows"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            filter_mutations_by_isin_lists={"type": ["absent"]},
        )

    with pytest.raises(ValueError, match="zero samples"):
        prepare_oncoplot_data(
            df,
            gene_col="gene",
            sample_col="sample",
            filter_samples_by_isin_lists={"type": ["absent"]},
        )


def test_ampersand_delimited_so_terms_are_rejected():
    df = pd.DataFrame(
        {
            "sample": ["S1"],
            "gene": ["TP53"],
            "type": ["missense_variant&intron_variant"],
        }
    )
    with pytest.raises(ValueError, match="ampersand"):
        prepare_oncoplot_data(df, gene_col="gene", sample_col="sample", mutation_type_col="type")


def test_pathway_ranking_and_reserved_other_validation():
    pathway = pd.DataFrame({"gene": ["TP53", "PTEN", "EGFR"], "pathway": ["B", "A", "A"]})
    assert rank_genes_by_pathway(pathway, gene_ranks=["TP53", "EGFR", "PTEN"], pathway_ranks=["A", "B"]) == [
        "EGFR",
        "PTEN",
        "TP53",
    ]

    bad_pathway = pd.DataFrame({"gene": ["TP53"], "pathway": ["Other"]})
    with pytest.raises(ValueError, match="reserved"):
        prepare_oncoplot_data(mutation_df(), gene_col="gene", sample_col="sample", pathway=bad_pathway)


def test_prepared_pathway_groups_include_other_for_unmapped_genes():
    pathway = pd.DataFrame({"gene": ["TP53", "PTEN"], "pathway": ["Cell cycle", "PI3K"]})
    prepared = prepare_oncoplot_data(
        mutation_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        include_genes=["TP53", "PTEN", "EGFR"],
        pathway=pathway,
    )
    assert prepared.pathway_by_gene == {"TP53": "Cell cycle", "PTEN": "PI3K", "EGFR": "Other"}
    assert [group.name for group in prepared.pathway_groups] == ["Cell cycle", "PI3K", "Other"]
    assert set(prepared.tiles["Pathway"].astype(str)) == {"Cell cycle", "PI3K", "Other"}


def test_metadata_sorting_and_tmb_preparation():
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3", "S4", "S5"],
            "group": ["B", "A", "A", "B", "C"],
        }
    )
    prepared = prepare_oncoplot_data(
        mutation_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_sort_cols=["group"],
        metadata_sort_by="alphabetical",
        metadata_sort_desc=False,
        prepare_tmb=True,
        show_all_samples=True,
    )
    assert prepared.samples[:2] == ["S2", "S3"]
    assert prepared.tmb is not None
    assert prepared.tmb_value_col == "Mutations"
    assert prepared.tmb_totals is not None
    assert prepared.tmb_type_counts is not None
    assert prepared.metadata_tracks is not None
    assert prepared.metadata_tracks[0].levels == ["A", "B", "C"]


def test_category_levels_follow_pandas_categorical_dtype_order():
    data = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "gene": ["TP53", "TP53", "TP53"],
            "type": pd.Categorical(
                ["beta_type", "alpha_type", "gamma_type"],
                categories=["gamma_type", "beta_type", "alpha_type", "unused_type"],
                ordered=True,
            ),
        }
    )
    metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "group": pd.Categorical(
                ["late", "early", "middle"],
                categories=["early", "middle", "late", "unused"],
                ordered=True,
            ),
        }
    )
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": pd.Categorical(
                ["subclonal", "clonal", "subclonal"],
                categories=["clonal", "subclonal", "unused"],
                ordered=True,
            ),
            "mutations": [2, 5, 3],
        }
    )

    prepared = prepare_oncoplot_data(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        tmb_data=tmb,
        prepare_tmb=True,
        show_all_samples=True,
    )

    assert prepared.mutation_type_levels == ["gamma_type", "beta_type", "alpha_type"]
    assert prepared.mutation_type_order_source == "categorical"
    assert prepared.metadata_tracks is not None
    assert prepared.metadata_tracks[0].levels == ["early", "middle", "late"]
    assert prepared.metadata_tracks[0].level_order_source == "categorical"
    assert prepared.tmb_type_levels == ["clonal", "subclonal"]
    assert prepared.tmb_type_order_source == "categorical"
    assert list(prepared.tmb_type_counts.columns) == ["clonal", "subclonal"]


def test_explicit_category_orders_prepend_present_levels_and_append_missing_observed_levels():
    data = pd.DataFrame(
        {
            "sample": ["S1", "S2", "S3"],
            "gene": ["TP53", "TP53", "TP53"],
            "type": ["beta_type", "alpha_type", "gamma_type"],
        }
    )
    metadata = pd.DataFrame({"sample": ["S1", "S2", "S3"], "group": ["B", "A", "C"]})
    tmb = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2"],
            "tmb_type": ["Beta", "Alpha", "Gamma"],
            "mutations": [2, 5, 3],
        }
    )

    prepared = prepare_oncoplot_data(
        data,
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        metadata=metadata,
        metadata_cols=["group"],
        mutation_type_order=["gamma_type", "beta_type", "unused_type"],
        metadata_category_orders={"group": ["C", "B", "unused"]},
        tmb_type_order=["Gamma", "Alpha", "unused"],
        tmb_data=tmb,
        prepare_tmb=True,
        show_all_samples=True,
    )

    assert prepared.mutation_type_levels == ["gamma_type", "beta_type", "alpha_type"]
    assert prepared.metadata_tracks is not None
    assert prepared.metadata_tracks[0].levels == ["C", "B", "A"]
    assert prepared.tmb_type_levels == ["Gamma", "Alpha", "Beta"]


def test_explicit_category_orders_reject_duplicates():
    with pytest.raises(ValueError, match="mutation_type_order"):
        prepare_oncoplot_data(
            mutation_df(),
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            mutation_type_order=["A", "A"],
        )

    with pytest.raises(ValueError, match="metadata_category_orders"):
        prepare_oncoplot_data(
            mutation_df(),
            gene_col="gene",
            sample_col="sample",
            metadata=pd.DataFrame({"sample": ["S1"], "group": ["A"]}),
            metadata_category_orders={"group": ["A", "A"]},
        )

    with pytest.raises(ValueError, match="tmb_type_order"):
        prepare_oncoplot_data(
            mutation_df(),
            gene_col="gene",
            sample_col="sample",
            mutation_type_col="type",
            tmb_type_order=["A", "A"],
        )


def test_custom_tmb_sample_column_does_not_need_to_be_first():
    tmb = pd.DataFrame(
        {
            "mutations": [5, 7],
            "sample": ["S1", "S2"],
        }
    )
    prepared = prepare_oncoplot_data(
        mutation_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        tmb_data=tmb,
        prepare_tmb=True,
    )
    assert prepared.tmb_sample_col == "sample"
    assert prepared.tmb_value_col == "mutations"
    assert prepared.tmb_totals is not None
    assert prepared.tmb_totals.loc["S1"] == 5.0
    assert prepared.tmb_totals.loc["S2"] == 7.0


def test_show_all_samples_includes_custom_tmb_only_samples():
    tmb = pd.DataFrame(
        {
            "mutations": [5, 11],
            "sample": ["S1", "TMB_ONLY"],
        }
    )
    prepared = prepare_oncoplot_data(
        mutation_df(),
        gene_col="gene",
        sample_col="sample",
        mutation_type_col="type",
        tmb_data=tmb,
        prepare_tmb=True,
        show_all_samples=True,
        total_samples="all",
    )
    assert "TMB_ONLY" in prepared.samples
    assert prepared.tmb_totals is not None
    assert prepared.tmb_totals.loc["TMB_ONLY"] == 11.0
    assert prepared.total_samples == len(prepared.samples)


def test_gbm_fixture_top_gene_order_matches_ggoncoplot_snapshot():
    fixture = (
        Path(__file__).resolve().parents[1]
        / "python_refactor_goal_sources"
        / "ggoncoplot"
        / "inst"
        / "testdata"
        / "GBM_tcgamutations_mc3_maf.csv.gz"
    )
    if not fixture.exists():
        pytest.skip("Nested ggoncoplot source fixture is not available.")
    gbm = pd.read_csv(fixture)
    assert identify_top_genes(gbm, "Hugo_Symbol", "Tumor_Sample_Barcode", top_n=10) == [
        "PTEN",
        "TP53",
        "TTN",
        "EGFR",
        "MUC16",
        "FLG",
        "NF1",
        "RYR2",
        "ATRX",
        "PIK3R1",
    ]
