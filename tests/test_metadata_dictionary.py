"""Source-scoped style and explicit cohort semantics."""
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyoncoplot import oncoplot, prepare_oncoplot_data, OncoplotOptions


@pytest.fixture
def cohort():
    data = pd.DataFrame({"sample": ["a", "b", "c", "d"], "gene": ["G", "G", "H", "H"],
                         "event": ["A"] * 4})
    meta = pd.DataFrame({"sample": ["a", "b", "c"], "treatment": ["A", "B", None],
                         "response": ["A", "A", "B"], "age": [18, 65, 40]})
    return dict(data=data, gene_col="gene", sample_col="sample", mutation_type_col="event",
                metadata=meta, metadata_cols=["treatment", "response", "age"], top_n=None)


def test_empty_filters_and_no_input_mutation(cohort):
    before = deepcopy(cohort)
    a = prepare_oncoplot_data(**cohort)
    b = prepare_oncoplot_data(**cohort, metadata_filters={})
    assert a.samples == b.samples and a.total_samples == b.total_samples
    assert_frame_equal(a.tiles, b.tiles)
    assert_frame_equal(cohort["data"], before["data"])
    assert_frame_equal(cohort["metadata"], before["metadata"])


def test_inclusive_and_missing_scope(cohort):
    p = prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort",
        metadata_filters={"treatment": {"include": ["A"], "missing": "include"},
                          "age": {"min": 18, "max": 40}})
    assert set(p.samples) == {"a", "c"}
    assert p.total_samples == 2
    assert len(p.tiles) == 2


def test_missing_absent_table_samples(cohort):
    p = prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort",
        metadata_filters={"treatment": {"include": ["A"], "missing": "include"}})
    assert set(p.samples) == {"a", "c", "d"}
    with pytest.raises(ValueError, match="Missing metadata"):
        prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort",
                             metadata_filters={"treatment": {"missing": "error"}})


@pytest.mark.parametrize("rules", [{"min": float("inf")}, {"wat": 1}, {"include": "A"},
                                   {"include": [["A"]]}, {"missing": "bad"}, {"min": 70, "max": 2}])
def test_filter_validation(cohort, rules):
    with pytest.raises(ValueError):
        prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort", metadata_filters={"age": rules})


def test_mode_zero_unknown_and_categorical(cohort):
    for extra in ({"metadata_filters": {"age": {"min": 2}}},
                  {"metadata_filter_mode": "display_only"},
                  {"metadata_filter_mode": "analysis_cohort", "metadata_filters": {"unknown": {}}},
                  {"metadata_filter_mode": "analysis_cohort", "metadata_filters": {"age": {"min": 999}}},
                  {"metadata_filter_mode": "analysis_cohort", "metadata_filters": {"treatment": {"min": 0}}}):
        with pytest.raises(ValueError):
            prepare_oncoplot_data(**cohort, **extra)


def test_reference_denominators_and_external_rows(cohort):
    filters = dict(metadata_filter_mode="analysis_cohort", metadata_filters={"age": {"min": 18, "max": 18}})
    p = prepare_oncoplot_data(**cohort, **filters, gene_denominators={"G": 100, "H": 200})
    assert p.row_summary.loc["G", "Denominator"] == 100
    assert p.row_summary.loc["G", "Count"] == 1
    rows = pd.DataFrame({"gene": ["G", "H"], "count": [50, 75], "denominator": [100, 200]})
    p = prepare_oncoplot_data(**cohort, **filters, row_summaries=rows)
    assert p.row_summary.loc["G", "Count"] == 50
    assert p.row_summary.loc["G", "Denominator"] == 100


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_source_labels_do_not_collide_or_mutate_science(cohort, backend):
    result = oncoplot(**cohort, backend=backend, options={
        "legend_label_overrides": {"A": "Global A"},
        "metadata_styles": {"treatment": {"labels": {"A": "Treatment A"}},
                            "response": {"labels": {"A": "Response A"}}}})
    try:
        if backend == "matplotlib":
            labels = [t.get_text() for legend in result.figure.legends for t in legend.get_texts()]
        else:
            labels = [trace.name for trace in result.figure.data if trace.name]
        assert any("Treatment A" in label for label in labels)
        assert any("Response A" in label for label in labels)
        assert any("Global A" in label for label in labels)
        assert set(result.prepared_data.tiles.MutationType) == {"A"}
    finally:
        if backend == "matplotlib":
            plt.close(result.figure)


def test_nested_style_mutation_revalidated():
    options = OncoplotOptions(metadata_styles={"x": {"legend": {"ncol": 2}}})
    options.metadata_styles["x"]["legend"]["ncol"] = 0
    from pyoncoplot._options import coerce_options
    with pytest.raises(ValueError):
        coerce_options(options)


@pytest.mark.parametrize("total", ["any_mutations", "all", "oncoplot"])
def test_states_tmb_and_total_population(cohort, total):
    p = prepare_oncoplot_data(**cohort, total_samples=total, prepare_tmb=True,
        tmb_data=pd.DataFrame({"sample": ["a", "b", "c", "d"], "burden": [10, 20, 30, 40]}),
        cell_states=pd.DataFrame({"sample": ["b", "c"], "gene": ["H", "G"], "state": ["unresolved"] * 2}),
        sample_order=["d", "c", "b", "a"],
        metadata_filter_mode="analysis_cohort", metadata_filters={"age": {"max": 40}})
    assert p.samples == ["c", "a"]
    assert p.total_samples == 2
    assert set(p.cell_states.Sample) == {"c"}
    assert set(p.tmb[p.tmb_sample_col].astype(str)) == {"a", "c"}
    assert sorted(p.tmb_totals.tolist()) == [10, 30]


@pytest.mark.parametrize("value", [None, "B"])
def test_duplicate_metadata_conflicts(cohort, value):
    duplicate = cohort["metadata"].iloc[[0]].copy()
    duplicate["treatment"] = value
    cohort["metadata"] = pd.concat([cohort["metadata"], duplicate])
    with pytest.raises(ValueError, match="unique identifiers"):
        prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort", metadata_filters={"age": {"min": 0}})


def test_dictionary_instance_yaml_parity(cohort, tmp_path):
    import yaml
    style = {"metadata_styles": {"treatment": {"labels": {"A": "Treatment A"}}}}
    filters = dict(metadata_filter_mode="analysis_cohort", metadata_filters={"age": {"max": 40}})
    config = {**cohort, **filters, "backend": "matplotlib", "options": style}
    for key in ("data", "metadata"):
        path = tmp_path / (key + ".csv")
        config[key].to_csv(path, index=False)
        config[key] = str(path)
    path = tmp_path / "plot.yaml"
    path.write_text(yaml.safe_dump(config))
    results = [oncoplot(**cohort, **filters, backend="matplotlib", options=style),
               oncoplot(params={**cohort, **filters, "backend": "matplotlib", "options": OncoplotOptions(**style)}),
               oncoplot(params=path)]
    try:
        for result in results[1:]:
            assert_frame_equal(results[0].prepared_data.tiles, result.prepared_data.tiles)
            assert result.prepared_data.samples == results[0].prepared_data.samples
    finally:
        for result in results:
            plt.close(result.figure)


def test_nested_layout_precedence(cohort):
    result = oncoplot(**cohort, backend="matplotlib", options={
        "compact_layout": True, "metadata_legend_position": "bottom",
        "metadata_legend_ncol": {"treatment": 1, "response": 2},
        "metadata_styles": {"treatment": {"legend": {"ncol": 2, "alignment": "top"}}}})
    try:
        legends = {l.get_title().get_text(): l for l in result.figure.legends}
        assert legends["Treatment"]._ncols == 2
        assert legends["Response"]._ncols == 2
    finally:
        plt.close(result.figure)


@pytest.mark.parametrize("backend,compact", [("plotly", False), ("matplotlib", False)])
def test_unsupported_nested_layout(cohort, backend, compact):
    with pytest.raises(NotImplementedError, match="metadata_styles"):
        oncoplot(**cohort, backend=backend, options={"compact_layout": compact,
            "metadata_styles": {"treatment": {"legend": {"ncol": 1}}}})


def test_unknown_styles_even_hidden(cohort):
    with pytest.raises(ValueError, match="unknown metadata"):
        oncoplot(**cohort, options={"show_metadata_legends": False,
            "metadata_styles": {"unknown": {"labels": {"A": "A"}}}})


def test_filtered_input_is_not_mutated(cohort):
    before = deepcopy(cohort)
    prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort",
                          metadata_filters={"age": {"max": 40}})
    assert_frame_equal(cohort["data"], before["data"])
    assert_frame_equal(cohort["metadata"], before["metadata"])


def test_numeric_colorbar_counts_are_noops(cohort):
    results = [oncoplot(**cohort, backend="matplotlib", options={"compact_layout": True,
        "metadata_legend_position": "bottom", "metadata_styles": styles})
        for styles in ({}, {"age": {"legend": {"ncol": 5, "nrow": 2}}})]
    try:
        assert [list(ax.get_position().bounds) for ax in results[0].figure.axes] == [list(ax.get_position().bounds) for ax in results[1].figure.axes]
    finally:
        for result in results:
            plt.close(result.figure)


def test_filter_does_not_add_metadata_only_samples(cohort):
    extra = pd.DataFrame({"sample": ["extra"], "treatment": ["A"], "response": ["A"], "age": [20]})
    cohort["metadata"] = pd.concat([cohort["metadata"], extra], ignore_index=True)
    p = prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort", metadata_filters={"age": {"min": 18}})
    assert "extra" not in p.samples
    assert p.total_samples == 3


def test_exclude_and_existing_sample_filters_intersect(cohort):
    p = prepare_oncoplot_data(**cohort, metadata_filter_mode="analysis_cohort",
        metadata_filters={"age": {"min": 18, "max": 65}, "treatment": {"exclude": ["B"], "missing": "include"}},
        filter_samples_by_isin_lists={"response": ["A"]})
    assert p.samples == ["a"]


def test_equal_display_labels_preserve_categories(cohort):
    result = oncoplot(**cohort, backend="matplotlib", options={
        "metadata_styles": {"treatment": {"labels": {"A": "Same", "B": "Same"}}}})
    try:
        legend = next(l for l in result.figure.legends if l.get_title().get_text() == "Treatment")
        assert [t.get_text() for t in legend.get_texts()].count("Same") == 2
        assert set(result.prepared_data.metadata.treatment.dropna()) == {"A", "B"}
    finally:
        plt.close(result.figure)
