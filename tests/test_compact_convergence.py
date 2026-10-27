"""Public issue-15 geometry remains stable while redundant passes stop."""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pytest
from docs.examples.legend_order_alignment import publication_example


@pytest.mark.parametrize("angle,expected", [(None, 11), (0, 13), (90, 14)])
def test_converged_draw_counts(monkeypatch, angle, expected):
    calls = []
    original = FigureCanvasAgg.draw
    def draw(canvas, *args, **kwargs):
        calls.append(None)
        return original(canvas, *args, **kwargs)
    monkeypatch.setattr(FigureCanvasAgg, "draw", draw)
    result = publication_example(overrides={"tmb_y_label_angle": angle})
    try:
        assert len(calls) <= expected
        # Existing publication geometry tests cover clipping and shared alignment.
        assert result.prepared_data.total_samples == 4
    finally:
        plt.close(result.figure)
