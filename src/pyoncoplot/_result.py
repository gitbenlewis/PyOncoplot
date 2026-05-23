"""Result wrapper returned by the public API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ._data import PreparedOncoplotData


@dataclass
class OncoplotResult:
    """Wrapper around backend-specific figure objects."""

    figure: Any
    backend: str
    prepared_data: PreparedOncoplotData
    copy_on_click: str = "sample"

    def show(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.figure, "show"):
            return self.figure.show(*args, **kwargs)
        raise TypeError("The current figure object does not provide a show() method.")

    def to_html(self, *args: Any, **kwargs: Any) -> str:
        if self.backend != "plotly":
            raise TypeError("to_html() is only available for Plotly-backed oncoplots.")

        include_plotlyjs = kwargs.pop("include_plotlyjs", "cdn")
        full_html = kwargs.pop("full_html", True)
        html = self.figure.to_html(
            *args,
            include_plotlyjs=include_plotlyjs,
            full_html=full_html,
            **kwargs,
        )
        script = """
<script>
(function () {
  function pointMeta(point) {
    var value = point && point.customdata;
    if (Array.isArray(value)) value = value[0];
    return value && typeof value === 'object' ? value : {copy_value: value};
  }
  function copyValue(point) {
    var meta = pointMeta(point);
    if (meta.copy_value !== undefined) return meta.copy_value;
    return "";
  }
  function sampleFromMeta(meta) {
    if (!meta || typeof meta !== 'object') return null;
    return meta.sample || null;
  }
  function flattenCustomdata(customdata) {
    if (!Array.isArray(customdata)) return [];
    if (Array.isArray(customdata[0])) return customdata.flat();
    return customdata;
  }
  function applyLinkedSelection(graph, samples) {
    if (!samples || !samples.size || !graph || !graph.data) return;
    graph.data.forEach(function (trace, traceIndex) {
      var customdata = flattenCustomdata(trace.customdata);
      if (!customdata.length) return;
      var points = [];
      customdata.forEach(function (meta, pointIndex) {
        var sample = sampleFromMeta(meta);
        if (sample && samples.has(String(sample))) points.push(pointIndex);
      });
      if (points.length) {
        Plotly.restyle(graph, {selectedpoints: [points]}, [traceIndex]);
      }
    });
  }
  function attachClipboard() {
    var graph = document.querySelector('.plotly-graph-div');
    if (!graph || !graph.on) return;
    graph.on('plotly_click', function (eventData) {
      if (!eventData || !eventData.points || !eventData.points.length) return;
      var value = copyValue(eventData.points[0]);
      if (value === undefined || value === null || value === "") return;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(String(value));
      }
    });
    graph.on('plotly_selected', function (eventData) {
      if (!eventData || !eventData.points || !eventData.points.length) return;
      var samples = new Set();
      eventData.points.forEach(function (point) {
        var sample = sampleFromMeta(pointMeta(point));
        if (sample) samples.add(String(sample));
      });
      applyLinkedSelection(graph, samples);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachClipboard);
  } else {
    attachClipboard();
  }
})();
</script>
"""
        if "</body>" in html:
            return html.replace("</body>", script + "\n</body>")
        return html + script

    def save(self, path: str, **kwargs: Any) -> None:
        target = Path(path)
        suffix = target.suffix.lower()
        if self.backend == "plotly":
            if suffix in {"", ".html", ".htm"}:
                target.write_text(self.to_html(**kwargs), encoding="utf-8")
            else:
                self.figure.write_image(str(target), **kwargs)
            return

        if hasattr(self.figure, "savefig"):
            self.figure.savefig(str(target), bbox_inches="tight", **kwargs)
            return

        raise TypeError("The current figure object cannot be saved by pyoncoplot.")
