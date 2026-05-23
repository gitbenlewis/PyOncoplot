"""Visual and behavioral options for oncoplot rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    metadata_legend_nrow: Optional[int] = None
    metadata_legend_ncol: Optional[int] = None
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
    font_family: str = "Arial"
    gene_font_style: Literal["normal", "italic", "bold", "bold_italic"] = "normal"
    sample_font_style: Literal["normal", "italic", "bold", "bold_italic"] = "normal"
    font_style_metadata: Literal["normal", "italic", "bold", "bold_italic"] = "normal"

    log10_transform_tmb: bool = True
    scientific_tmb: bool = False
    show_gene_bar_axis: bool = True
    show_tmb_axis: bool = True
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

    metadata_na_marker: str = "!"
    metadata_na_marker_size: float = 7
    metadata_max_levels: int = 6
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

    def __post_init__(self) -> None:
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
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
        if self.gene_bar_width_ratio >= 0.95:
            raise ValueError("gene_bar_width_ratio must be less than 0.95.")
        if self.tmb_height_ratio + self.metadata_height_ratio >= 0.95:
            raise ValueError("tmb_height_ratio + metadata_height_ratio must be less than 0.95.")
        if self.gene_bar_label_round < 0:
            raise ValueError("gene_bar_label_round must be >= 0.")
        if self.metadata_max_levels < 1:
            raise ValueError("metadata_max_levels must be >= 1.")
        for name in ("metadata_legend_nrow", "metadata_legend_ncol", "gene_bar_scale_n_breaks"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when supplied.")
        if self.gene_bar_scale_breaks is not None:
            try:
                [float(value) for value in self.gene_bar_scale_breaks]
            except (TypeError, ValueError) as exc:
                raise ValueError("gene_bar_scale_breaks must be a sequence of numeric values.") from exc


def coerce_options(options: Optional[OncoplotOptions | Mapping[str, Any]]) -> OncoplotOptions:
    if options is None:
        return OncoplotOptions()
    if isinstance(options, Mapping):
        return OncoplotOptions(**dict(options))
    if not isinstance(options, OncoplotOptions):
        raise TypeError("options must be an OncoplotOptions instance, a mapping, or None.")
    return options
