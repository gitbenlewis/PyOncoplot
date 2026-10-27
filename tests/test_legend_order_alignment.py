"""Issue 15: renderer-measured legend packing without data changes."""
from collections import ChainMap
from dataclasses import fields
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pandas.testing import assert_frame_equal
from pyoncoplot import OncoplotOptions
from pyoncoplot._matplotlib import render_matplotlib_oncoplot
from pyoncoplot._plotly import render_plotly_oncoplot
from test_publication_layout import axes_by_name

spec = importlib.util.spec_from_file_location("order_alignment", Path(__file__).parents[1] / "docs/examples/legend_order_alignment.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def boxes(result):
    fig = result.figure
    renderer = fig.canvas.get_renderer()
    axes = axes_by_name(fig)
    categorical = fig.legends[-2:]
    numeric = next(ax for ax in fig.axes if ax not in axes.values())
    return [categorical[0].get_window_extent(renderer), numeric.get_tightbbox(renderer),
            categorical[1].get_window_extent(renderer)]


@pytest.mark.parametrize("angle", [0, 90])
def test_order_top_bounds_exports_and_science(angle, tmp_path):
    r = example.publication_example(overrides={"tmb_y_label_angle": angle})
    base = example.publication_example(overrides={"metadata_legend_order": None, "metadata_legend_alignment": {}})
    a, b = r.prepared_data, base.prepared_data
    assert a.metadata_cols == b.metadata_cols == ["response", "measurement", "treatment"]
    assert (a.samples, a.genes, a.total_samples, a.row_summary_external) == (b.samples, b.genes, b.total_samples, b.row_summary_external)
    for name in ("tiles", "metadata", "mutation_counts", "tmb", "row_summary", "cell_states", "main_grid_rows"):
        if getattr(a, name) is None:
            assert getattr(b, name) is None
        else:
            assert_frame_equal(getattr(a, name), getattr(b, name))
    assert [[t.get_text() for t in legend.get_texts()] for legend in r.figure.legends[1:]] == [["A", "B", "C"], ["Stable", "Response"]]
    geometry = boxes(r)
    assert [box.y1 for box in geometry] == pytest.approx([geometry[0].y1] * 3)
    assert geometry[0].height > geometry[2].height
    for first, second in zip(geometry, geometry[1:]):
        assert second.x0 - first.x1 == pytest.approx(r.figure.dpi / 72)
    renderer = r.figure.canvas.get_renderer()
    all_boxes = geometry + [legend.get_window_extent(renderer) for legend in r.figure.legends[:1]]
    all_boxes += [ax.get_tightbbox(renderer) for ax in axes_by_name(r.figure).values()]
    for i, box in enumerate(all_boxes):
        assert 0 <= box.x0 < box.x1 <= r.figure.bbox.x1
        assert 0 <= box.y0 < box.y1 <= r.figure.bbox.y1
        assert all(not box.overlaps(other) for other in all_boxes[i+1:])
    for suffix, signature in (("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")):
        path = tmp_path / f"ordered-{angle}.{suffix}"
        r.save(path, bbox_inches=None)
        assert path.read_bytes().startswith(signature)
    plt.close("all")


def test_partial_order_and_alignment_fallback():
    r = example.publication_example(overrides={"metadata_legend_order": ["measurement"],
        "metadata_legend_alignment": ChainMap({"response": "top"}), "show_legend": False})
    fig = r.figure
    renderer = fig.canvas.get_renderer()
    numeric = next(ax for ax in fig.axes if ax.get_label() not in ("main", "gene_bar", "metadata", "tmb"))
    cb = numeric.get_tightbbox(renderer)
    response, treatment = [l.get_window_extent(renderer) for l in fig.legends]
    assert cb.x1 < response.x0 < treatment.x0
    assert response.y1 == pytest.approx(treatment.y1)
    assert (cb.y0 + cb.y1) / 2 == pytest.approx((treatment.y0 + treatment.y1) / 2)
    plt.close("all")


def test_gap_control_and_ratio_when_clearance_permits():
    tight = example.publication_example(overrides={"show_legend": False, "tmb_height_ratio": .12})
    loose = example.publication_example(overrides={"show_legend": False, "legend_gap_points": 12, "tmb_height_ratio": .24})
    a, b = boxes(tight), boxes(loose)
    assert a[1].x0 - a[0].x1 < b[1].x0 - b[0].x1
    for result, ratio in ((tight, .12), (loose, .24)):
        axes = axes_by_name(result.figure)
        grid = axes["tmb"].get_subplotspec().get_gridspec()
        # No label minimum should inflate an already sufficient requested weight.
        assert grid.get_height_ratios()[0] == pytest.approx(1.2 * ratio / .14)
        assert axes["tmb"].yaxis.label.get_window_extent(result.figure.canvas.get_renderer()).height < axes["tmb"].bbox.height
    plt.close("all")


@pytest.mark.parametrize("kwargs", [
    {"metadata_legend_order": ["response", "response"]}, {"metadata_legend_order": "response"},
    {"metadata_legend_order": [1]}, {"metadata_legend_alignment": "top"},
    {"metadata_legend_alignment": {"response": "bottom"}}, {"metadata_legend_alignment": {1: "top"}},
    {"metadata_legend_alignment": {"response": []}},
])
def test_invalid_values(kwargs):
    with pytest.raises(ValueError, match="metadata_legend"):
        OncoplotOptions(**kwargs)


@pytest.mark.parametrize("renderer", [render_matplotlib_oncoplot, render_plotly_oncoplot])
@pytest.mark.parametrize("kwargs", [{"metadata_legend_order": ["Observed\nmeasurement"]},
    {"metadata_legend_alignment": {"metadata:response": "top"}}])
def test_unknown_source_keys_even_hidden(renderer, kwargs):
    prepared = example.publication_example().prepared_data
    with pytest.raises(ValueError, match="unknown metadata columns"):
        renderer(prepared, palette={}, options={"show_metadata_legends": False, **kwargs})
    plt.close("all")


@pytest.mark.parametrize("renderer", [render_matplotlib_oncoplot, render_plotly_oncoplot])
@pytest.mark.parametrize("kwargs", [{"metadata_legend_order": []}, {"metadata_legend_alignment": {"response": "center"}}])
def test_unsupported_layouts_are_explicit(renderer, kwargs):
    prepared = example.publication_example().prepared_data
    with pytest.raises(NotImplementedError, match="metadata_legend"):
        renderer(prepared, palette={}, options=kwargs)
    plt.close("all")


def test_options_appended_and_defaults_unchanged():
    assert [f.name for f in fields(OncoplotOptions)][107:109] == ["metadata_legend_order", "metadata_legend_alignment"]
    assert OncoplotOptions().metadata_legend_order is None
    assert OncoplotOptions().metadata_legend_alignment == {}


@pytest.mark.parametrize("titles", [True, False])
@pytest.mark.parametrize("width", [960, 1800])
def test_wrapped_order_and_alignment_with_or_without_titles(titles, width):
    r = example.publication_example(overrides={"width": width, "height": 1100,
        "show_legend_titles": titles, "metadata_legend_alignment": {"treatment": "top", "measurement": "top", "response": "top"}})
    geometry = boxes(r)
    # Read shelves top-to-bottom and then left-to-right, using full rendered bounds.
    assert sorted(range(3), key=lambda i: (-round(geometry[i].y1, 4), geometry[i].x0)) == [0, 1, 2]
    for i, box in enumerate(geometry):
        assert all(not box.overlaps(other) for other in geometry[i+1:])
        assert 0 <= box.x0 < box.x1 <= r.figure.bbox.x1
        assert 0 <= box.y0 < box.y1 <= r.figure.bbox.y1
    plt.close("all")


@pytest.mark.parametrize("angle", [0, 90])
def test_histogram_text_minimum_is_tight_not_default_panel(angle):
    r = example.publication_example(overrides={"tmb_height_ratio": .001, "tmb_y_label_angle": angle})
    ax = axes_by_name(r.figure)["tmb"]
    renderer = r.figure.canvas.get_renderer()
    label_height = ax.yaxis.label.get_window_extent(renderer).height
    tick_boxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels()]
    needed = max(label_height + 2, max((t.height + 2 for t in tick_boxes), default=0))
    assert label_height <= ax.bbox.height < needed + 5
    plt.close("all")


def test_order_only_keeps_centered_alignment():
    r = example.publication_example(overrides={"metadata_legend_alignment": {}})
    geometry = boxes(r)
    assert [(box.y0 + box.y1) / 2 for box in geometry] == pytest.approx([(geometry[0].y0 + geometry[0].y1) / 2] * 3)
    plt.close("all")
