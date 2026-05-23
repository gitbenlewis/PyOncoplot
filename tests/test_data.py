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
    assert hit["Tooltip"] == "<strong>S1</strong><br>a<br>b"
    assert prepared.genes == ["EGFR", "PTEN", "TP53"]
    assert prepared.samples[0] == "S1"
    assert set(prepared.mutation_counts.columns) == {"Gene", "MutationType", "Count"}


def test_prepare_oncoplot_data_validates_empty_ids_and_metadata_duplicates():
    df = mutation_df()
    bad = df.copy()
    bad.loc[0, "sample"] = ""
    with pytest.raises(ValueError, match="Sample column"):
        prepare_oncoplot_data(bad, gene_col="gene", sample_col="sample")

    metadata = pd.DataFrame({"sample": ["S1", "S1"], "sex": ["F", "M"]})
    with pytest.raises(ValueError, match="unique"):
        prepare_oncoplot_data(df, gene_col="gene", sample_col="sample", metadata=metadata)


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
