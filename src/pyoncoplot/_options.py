"""Visual and behavioral options for oncoplot rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from copy import deepcopy
from numbers import Number
import math
from collections.abc import Mapping
from typing import Any, Callable, Literal, Optional, Sequence

from ._utils import prettify


@dataclass
class OncoplotOptions:
    """Configuration shared by the Plotly and Matplotlib renderers."""

    width: int = 1200
    height: int = 650
    selection_type: Literal["none", "multiple", "single"] = "none"

    tmb_height_ratio: float = 0.14
    gene_bar_width_ratio: float = 0.18
    metadata_height_ratio: float = 0.18
    metadata_position: Literal["bottom", "top"] = "bottom"
    buffer_metadata: float = 0.08
    buffer_tmb: float = 0.08
    buffer_gene_bar: float = 0.02

    x_label: str = "Sample"
    y_label: str = "Gene"
    show_sample_ids: bool = False
    sample_id_position: Literal["bottom", "top"] = "bottom"
    sample_id_angle: float = 90
    show_x_label: bool = False
    show_y_label: bool = False
    show_tmb_y_label: bool = False
    show_legend: bool = True
    show_legend_titles: bool = True
    mutation_legend_position: Literal["bottom", "right", "none"] = "bottom"
    show_metadata_legends: bool = True
    metadata_legend_position: Literal["right", "bottom"] = "right"
    legend_key_size: float = 1.0
    legend_offsets: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    metadata_legend_nrow: Optional[int | Mapping[str, int]] = None
    metadata_legend_ncol: Optional[int | Mapping[str, int]] = None
    metadata_legend_key_size: float = 1.0

    tile_height: float = 1.0
    tile_width: float = 1.0
    tile_linewidth: float = 0.25
    row_separator_linewidth: float = 0.8
    background_color: str = "#E5E5E5"
    unspecified_mutation_color: str = "#1A1A1A"
    multi_hit_color: str = "black"

    font_size_x_label: float = 26
    font_size_y_label: float = 26
    font_size_genes: float = 12
    font_size_samples: float = 9
    font_size_gene_bar_axis: float = 10
    font_size_tmb_axis: float = 10
    font_size_metadata: float = 10
    font_size_metadata_bar_numbers: float = 8
    font_size_legend_text: Optional[float] = None
    font_size_legend_title: Optional[float] = None
    font_size_metadata_legend_text: Optional[float] = None
    font_size_metadata_legend_title: Optional[float] = None
    font_size_title: float = 14
    font_size_subplot_title: float = 12
    font_size_pathway: Optional[float] = None
    gene_name_x_offset: float = 0.0
    main_grid_rows_label_x_offset: float = 10.0
    font_family: str = "Arial"
    gene_font_style: Literal["normal", "italic", "bold", "bold_italic"] = "normal"
    sample_font_style: Literal["normal", "italic", "bold", "bold_italic"] = "normal"
    font_style_metadata: Literal["normal", "italic", "bold", "bold_italic"] = "normal"

    log10_transform_tmb: bool = True
    scientific_tmb: bool = False
    show_gene_bar_axis: bool = True
    show_tmb_axis: bool = True
    gene_bar_mode: Literal["counts", "percent"] = "counts"
    show_gene_bar_labels: bool = False
    gene_bar_label_round: int = 0
    gene_bar_label_padding: float = 0.24
    gene_bar_label_nudge: float = 0.0
    gene_bar_scale_breaks: Optional[Sequence[float]] = None
    gene_bar_scale_n_breaks: Optional[int] = None

    pathway_text_color: str = "white"
    pathway_background_color: str = "#1A1A1A"
    pathway_outline_color: str = "black"
    pathway_text_angle: float = 0

    prettify_legend_titles: bool = True
    prettify_legend_values: bool = True
    prettify_function: Callable[[str], str] = prettify
    legend_label_max_chars: Optional[int] = None
    legend_title_max_chars: Optional[int] = None
    title_text: Optional[str] = None
    main_subplot_title: Optional[str] = None
    tmb_subplot_title: Optional[str] = None
    gene_bar_subplot_title: Optional[str] = None
    metadata_subplot_title: Optional[str] = None

    metadata_na_marker: str = "!"
    metadata_na_marker_size: float = 7
    metadata_max_levels: int = 40
    metadata_numeric_plot_type: Literal["bar", "heatmap"] = "heatmap"
    metadata_legend_orientation_heatmap: Literal["vertical", "horizontal"] = "vertical"
    metadata_default_colors: Sequence[str] = field(
        default_factory=lambda: (
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
        )
    )

    # Publication layout controls; lengths ending in _points are typographic points.
    compact_layout: bool = False
    sample_id_padding: float = 4.0
    tmb_bar_color: Optional[str] = None
    tmb_bar_width: float = 1.0  # sample units; spacing is 1 - width
    tmb_tick_interval: Optional[int] = None
    tmb_y_label: Optional[str] = None
    gene_bar_label_format: Literal["percent", "count", "count_percent"] = "percent"
    tile_edgecolor: str = "white"
    row_separator_color: str = "#666666"
    title_gap_points: float = 12.0
    metadata_track_heights: Mapping[str, float] = field(default_factory=dict)
    metadata_track_spacing: float = 0.0
    metadata_numeric_scales: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    legend_label_overrides: Mapping[str, str] = field(default_factory=dict)
    legend_title_overrides: Mapping[str, str] = field(default_factory=dict)
    gene_bar_color: Optional[str] = None
    gene_bar_x_label: Optional[str] = None

    # Automatic compact spacing; append to preserve positional compatibility.
    legend_gap_points: float = 6.0

    gene_bar_label_position: Literal["bar_end", "column"] = "bar_end"

    tmb_y_label_angle: Optional[float] = None

    metadata_legend_order: Optional[Sequence[str]] = None
    metadata_legend_alignment: Mapping[str, Literal["top", "center"]] = field(default_factory=dict)

    metadata_styles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata_styles, Mapping):
            raise ValueError("metadata_styles must be a mapping by original source column.")
        for source, style in self.metadata_styles.items():
            if not isinstance(source, str) or not isinstance(style, Mapping) or set(style) - {"legend", "labels"}:
                raise ValueError("metadata_styles accepts source mappings containing legend and labels only.")
            labels = style.get("labels", {})
            if not isinstance(labels, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items()):
                raise ValueError("metadata_styles labels must map original string values to strings.")
            legend = style.get("legend", {})
            if not isinstance(legend, Mapping) or set(legend) - {"ncol", "nrow", "alignment"}:
                raise ValueError("metadata_styles legend accepts ncol, nrow, alignment only.")
            for key in ("ncol", "nrow"):
                if key in legend and (isinstance(legend[key], bool) or not isinstance(legend[key], int) or legend[key] < 1):
                    raise ValueError("metadata_styles legend counts must be positive integers.")
            if "alignment" in legend and legend["alignment"] not in ("top", "center"):
                raise ValueError("metadata_styles legend alignment must be top or center.")
        # Reject nonfinite scalar geometry before allocating backend artists.
        numeric = {f.name for f in fields(self) if isinstance(f.default, (int, float))
                   and not isinstance(f.default, bool)}
        numeric.update({"font_size_legend_text", "font_size_legend_title",
                        "font_size_metadata_legend_text", "font_size_metadata_legend_title",
                        "font_size_pathway", "tmb_y_label_angle", "gene_bar_scale_n_breaks",
                        "legend_label_max_chars", "legend_title_max_chars"})
        for name in numeric:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Number)
                                      or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number.")
            if name.startswith("font_size_") and value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0.")
        for name in ("tile_height", "tile_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0.")
        if self.metadata_legend_order is not None:
            order = self.metadata_legend_order
            if (isinstance(order, (str, bytes)) or not isinstance(order, Sequence)
                    or any(not isinstance(key, str) for key in order)):
                raise ValueError("metadata_legend_order must be a sequence of original metadata column names.")
            if len(set(order)) != len(order):
                raise ValueError("metadata_legend_order must not contain duplicate columns.")
        if (not isinstance(self.metadata_legend_alignment, Mapping)
                or any(not isinstance(key, str) or value not in ("top", "center")
                       for key, value in self.metadata_legend_alignment.items())):
            raise ValueError("metadata_legend_alignment must map original metadata column names to 'top' or 'center'.")
        if self.tmb_y_label_angle is not None and (
            isinstance(self.tmb_y_label_angle, bool)
            or not isinstance(self.tmb_y_label_angle, (int, float))
            or not math.isfinite(self.tmb_y_label_angle)
        ):
            raise ValueError("tmb_y_label_angle must be a finite number or None.")
        if self.gene_bar_label_position not in {"bar_end", "column"}:
            raise ValueError("gene_bar_label_position must be bar_end or column.")
        if isinstance(self.legend_gap_points, bool):
            raise ValueError("legend_gap_points must be a finite nonnegative number.")
        if not isinstance(self.compact_layout, bool):
            raise ValueError("compact_layout must be a boolean.")
        if self.tmb_y_label is not None and not isinstance(self.tmb_y_label, str):
            raise ValueError("tmb_y_label must be a string or None.")
        for name in ("sample_id_padding", "title_gap_points", "legend_gap_points", "metadata_track_spacing",
                     "tile_linewidth", "row_separator_linewidth"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if not isinstance(self.tmb_bar_width, (int, float)) or not math.isfinite(self.tmb_bar_width) or not 0 < self.tmb_bar_width <= 1:
            raise ValueError("tmb_bar_width must be in (0, 1].")
        if self.tmb_tick_interval is not None and (
            isinstance(self.tmb_tick_interval, bool) or not isinstance(self.tmb_tick_interval, int)
            or self.tmb_tick_interval < 1
        ):
            raise ValueError("tmb_tick_interval must be a positive integer.")
        if self.gene_bar_label_format not in {"percent", "count", "count_percent"}:
            raise ValueError("gene_bar_label_format must be percent, count, or count_percent.")
        for name in ("metadata_track_heights", "metadata_numeric_scales", "legend_label_overrides", "legend_title_overrides"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping.")
        for value in self.metadata_track_heights.values():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("metadata_track_heights values must be finite and > 0.")
        for labels in (self.legend_label_overrides, self.legend_title_overrides):
            if any(not isinstance(v, str) for v in labels.values()):
                raise ValueError("legend overrides must map source labels to strings.")
        for column, scale in self.metadata_numeric_scales.items():
            if not isinstance(scale, Mapping) or set(scale) - {"min", "max", "ticks", "missing_color", "show_missing_marker", "colorbar", "show_title"}:
                raise ValueError(f"Invalid metadata_numeric_scales entry for {column!r}.")
            if "show_missing_marker" in scale and not isinstance(scale["show_missing_marker"], bool):
                raise ValueError("show_missing_marker must be a boolean.")
            if "show_title" in scale and not isinstance(scale["show_title"], bool):
                raise ValueError("metadata numeric show_title must be a boolean.")
            colorbar = scale.get("colorbar", {})
            if not isinstance(colorbar, Mapping) or set(colorbar) - {"length", "thickness", "label_pad"}:
                raise ValueError("colorbar accepts length, thickness, label_pad in points.")
            for key, value in colorbar.items():
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (key != "label_pad" and value == 0)):
                    raise ValueError("colorbar sizes must be finite and positive (label_pad may be zero).")
            for bound in ("min", "max"):
                if bound in scale and (not isinstance(scale[bound], (int, float)) or not math.isfinite(scale[bound])):
                    raise ValueError("metadata scale bounds must be finite numbers.")
            if "min" in scale and "max" in scale and scale["min"] >= scale["max"]:
                raise ValueError("metadata scale min must be less than max.")
            if "ticks" in scale:
                ticks = scale["ticks"]
                if (not isinstance(ticks, Sequence) or isinstance(ticks, str) or not ticks
                    or any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in ticks)
                    or any(a >= b for a, b in zip(ticks, ticks[1:]))):
                    raise ValueError("metadata scale ticks must be finite, strictly increasing numbers.")
        from matplotlib.colors import is_color_like
        colors = [self.tile_edgecolor, self.row_separator_color]
        if self.gene_bar_x_label is not None and not isinstance(self.gene_bar_x_label, str):
            raise ValueError("gene_bar_x_label must be a string or None.")
        if self.gene_bar_color is not None:
            colors.append(self.gene_bar_color)
        if self.tmb_bar_color is not None:
            colors.append(self.tmb_bar_color)
        colors.extend(s["missing_color"] for s in self.metadata_numeric_scales.values() if "missing_color" in s)
        if any(not is_color_like(color) for color in colors):
            raise ValueError("Publication layout colors must be valid colors.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        if self.selection_type not in {"none", "multiple", "single"}:
            raise ValueError("selection_type must be one of: 'none', 'multiple', 'single'.")
        if self.metadata_position not in {"bottom", "top"}:
            raise ValueError("metadata_position must be 'bottom' or 'top'.")
        if self.sample_id_position not in {"bottom", "top"}:
            raise ValueError("sample_id_position must be 'bottom' or 'top'.")
        if self.metadata_numeric_plot_type not in {"bar", "heatmap"}:
            raise ValueError("metadata_numeric_plot_type must be 'bar' or 'heatmap'.")
        if self.gene_bar_mode not in {"counts", "percent"}:
            raise ValueError("gene_bar_mode must be 'counts' or 'percent'.")
        if self.mutation_legend_position not in {"bottom", "right", "none"}:
            raise ValueError("mutation_legend_position must be one of: 'bottom', 'right', 'none'.")
        if self.metadata_legend_position not in {"right", "bottom"}:
            raise ValueError("metadata_legend_position must be 'right' or 'bottom'.")
        if self.metadata_legend_orientation_heatmap not in {"vertical", "horizontal"}:
            raise ValueError("metadata_legend_orientation_heatmap must be 'vertical' or 'horizontal'.")
        for name in ("gene_font_style", "sample_font_style", "font_style_metadata"):
            if getattr(self, name) not in {"normal", "italic", "bold", "bold_italic"}:
                raise ValueError(f"{name} must be one of: normal, italic, bold, bold_italic.")
        for name in ("tmb_height_ratio", "gene_bar_width_ratio", "metadata_height_ratio"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in the interval [0, 1).")
        for name in (
            "buffer_metadata",
            "buffer_tmb",
            "buffer_gene_bar",
            "legend_key_size",
            "metadata_legend_key_size",
            "metadata_na_marker_size",
            "font_size_metadata_bar_numbers",
            "gene_bar_label_padding",
            "gene_name_x_offset",
            "main_grid_rows_label_x_offset",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
        for name in ("font_size_title", "font_size_subplot_title"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        for name in (
            "font_size_legend_text",
            "font_size_legend_title",
            "font_size_metadata_legend_text",
            "font_size_metadata_legend_title",
            "font_size_pathway",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when supplied.")
        for name in ("legend_label_max_chars", "legend_title_max_chars"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when supplied.")
        if not isinstance(self.legend_offsets, Mapping):
            raise ValueError("legend_offsets must be a mapping of legend keys to x/y offsets.")
        normalized_offsets: dict[str, dict[str, float]] = {}
        for key, offsets in self.legend_offsets.items():
            if not isinstance(offsets, Mapping):
                raise ValueError("legend_offsets values must be mappings with optional 'x' and 'y' entries.")
            unknown = set(offsets) - {"x", "y"}
            if unknown:
                raise ValueError("legend_offsets entries may only contain 'x' and 'y'.")
            try:
                normalized_offsets[str(key)] = {
                    "x": float(offsets.get("x", 0.0)),
                    "y": float(offsets.get("y", 0.0)),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError("legend_offsets x/y values must be numeric.") from exc
        if any(not math.isfinite(value) for offsets in normalized_offsets.values() for value in offsets.values()):
            raise ValueError("legend_offsets x/y values must be finite.")
        self.legend_offsets = normalized_offsets
        if self.gene_bar_width_ratio >= 0.95:
            raise ValueError("gene_bar_width_ratio must be less than 0.95.")
        if self.tmb_height_ratio + self.metadata_height_ratio >= 0.95:
            raise ValueError("tmb_height_ratio + metadata_height_ratio must be less than 0.95.")
        if self.gene_bar_label_round < 0:
            raise ValueError("gene_bar_label_round must be >= 0.")
        if self.metadata_max_levels < 1:
            raise ValueError("metadata_max_levels must be >= 1.")
        for name in ("metadata_legend_nrow", "metadata_legend_ncol"):
            value = getattr(self, name)
            counts = value.values() if isinstance(value, Mapping) else [value]
            if isinstance(value, Mapping) and any(not isinstance(key, str) for key in value):
                raise ValueError(f"{name} keys must be original metadata column names (strings).")
            if any(count is None or isinstance(count, bool) or not isinstance(count, int) or count < 1
                   for count in counts) and value is not None:
                raise ValueError(f"{name} must be a positive integer or a mapping of column names to positive integers.")
        if self.gene_bar_scale_n_breaks is not None and self.gene_bar_scale_n_breaks < 1:
            raise ValueError("gene_bar_scale_n_breaks must be >= 1 when supplied.")
        if self.gene_bar_scale_breaks is not None:
            try:
                values = [float(value) for value in self.gene_bar_scale_breaks]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("gene_bar_scale_breaks must be finite.")
            except (TypeError, ValueError) as exc:
                raise ValueError("gene_bar_scale_breaks must be a sequence of numeric values.") from exc


def _copy_option_value(value):
    """Materialize arbitrary Mapping containers before copying nested values."""
    if isinstance(value, Mapping):
        return {deepcopy(key): _copy_option_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_option_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_option_value(item) for item in value)
    return deepcopy(value)


def coerce_options(options: Optional[OncoplotOptions | Mapping[str, Any]]) -> OncoplotOptions:
    if options is None:
        return OncoplotOptions()
    if isinstance(options, Mapping):
        return OncoplotOptions(**_copy_option_value(options))
    if not isinstance(options, OncoplotOptions):
        raise TypeError("options must be an OncoplotOptions instance, a mapping, or None.")
    # Rendering consumes the base contract. Subclass-only state remains with the
    # caller; never forward extra dataclass fields to the base constructor.
    return OncoplotOptions(**{f.name: _copy_option_value(getattr(options, f.name))
                              for f in fields(OncoplotOptions)})


def validate_metadata_legend_columns(options: OncoplotOptions, columns: Sequence[str]) -> None:
    """Validate source names even when legends are hidden or the backend rejects mappings."""
    for name in ("metadata_styles", "metadata_legend_ncol", "metadata_legend_nrow", "metadata_legend_alignment", "metadata_legend_order"):
        value = getattr(options, name)
        if isinstance(value, Mapping) or (name == "metadata_legend_order" and value is not None):
            unknown = set(value) - set(columns)
            if unknown:
                raise ValueError(f"{name} references unknown metadata columns: {sorted(unknown)}")
