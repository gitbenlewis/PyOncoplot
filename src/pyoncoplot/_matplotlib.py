"""Matplotlib renderer for static oncoplots."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._data import PreparedOncoplotData
from ._metadata_colors import (
    categorical_metadata_palette,
    coerce_continuous_colormap,
    metadata_palette_spec,
    numeric_metadata_colormap_spec,
    numeric_metadata_endpoint_colors,
    sample_numeric_metadata_colormap,
)
from ._options import OncoplotOptions, coerce_options
from ._params import merge_params
from ._utils import reorder_by_priority


MATPLOTLIB_RENDER_PARAM_KEYS = {
    "prepared",
    "palette",
    "tmb_palette",
    "metadata_palette",
    "variant_value_palette",
    "options",
    "draw_gene_bar",
    "draw_tmb_bar",
}

MATPLOTLIB_RENDER_DEFAULTS: dict[str, Any] = {
    "tmb_palette": None,
    "metadata_palette": None,
    "variant_value_palette": "viridis",
    "draw_gene_bar": False,
    "draw_tmb_bar": False,
}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Patch, Rectangle
    except ImportError as exc:
        raise ImportError("Matplotlib rendering requires the 'matplotlib' package.") from exc
    return plt, ListedColormap, GridSpec, Patch, Rectangle


def _font_kwargs(style: str) -> Dict[str, str]:
    if style == "bold":
        return {"fontweight": "bold"}
    if style == "italic":
        return {"fontstyle": "italic"}
    if style == "bold_italic":
        return {"fontweight": "bold", "fontstyle": "italic"}
    return {}


def _legend_font_size(base: float, reference: float, scale: float = 0.75) -> float:
    return max(base, reference * scale)


def _legend_label(value: object, options: OncoplotOptions) -> str:
    text = "Unspecified" if pd.isna(value) else str(value)
    return options.prettify_function(text) if options.prettify_legend_values else text


def _legend_title(value: object, options: OncoplotOptions) -> str:
    text = "" if value is None else str(value)
    return options.prettify_function(text) if options.prettify_legend_titles else text


def _ordered_levels(
    levels: Optional[Sequence[object]],
    order_source: Optional[str],
    palette: Optional[Mapping[str, object]] = None,
) -> list[str]:
    level_names = [str(level) for level in levels or []]
    if order_source == "observed" and isinstance(palette, Mapping):
        return reorder_by_priority(level_names, [str(level) for level in palette])
    return level_names


def _metadata_track(prepared: PreparedOncoplotData, column: object):
    column_name = str(column)
    for track in prepared.metadata_tracks or []:
        if track.column == column_name:
            return track
    return None


def _metadata_levels(
    prepared: PreparedOncoplotData,
    column: object,
    values: pd.Series,
    palette_spec: object,
) -> list[str]:
    track = _metadata_track(prepared, column)
    if track is None:
        return [str(value) for value in pd.unique(values.dropna())]
    return _ordered_levels(track.levels, track.level_order_source, palette_spec if isinstance(palette_spec, Mapping) else None)


def _mutation_levels(prepared: PreparedOncoplotData, palette: Mapping[str, str]) -> list[str]:
    if prepared.mutation_type_levels:
        levels = _ordered_levels(
            prepared.mutation_type_levels,
            prepared.mutation_type_order_source,
            palette,
        )
    else:
        levels = [str(value) for value in pd.unique(prepared.tiles["MutationType"].dropna())]
    observed = set(prepared.tiles["MutationType"].dropna().astype(str))
    return [level for level in levels if level in observed]


def _tmb_levels(
    prepared: PreparedOncoplotData,
    palette: Mapping[str, str],
) -> list[str]:
    if prepared.tmb is None or prepared.tmb_type_col is None:
        return []
    if prepared.tmb_type_levels:
        levels = _ordered_levels(
            prepared.tmb_type_levels,
            prepared.tmb_type_order_source,
            palette,
        )
    else:
        levels = [str(value) for value in pd.unique(prepared.tmb[prepared.tmb_type_col].dropna())]
    observed = set(prepared.tmb[prepared.tmb_type_col].dropna().astype(str))
    return [level for level in levels if level in observed]


def _left_margin_for_metadata(
    prepared: PreparedOncoplotData,
    options: OncoplotOptions,
    *,
    draw_metadata: bool,
    fig_width: float,
) -> float:
    if not draw_metadata or not prepared.metadata_cols:
        return 0.08

    labels = [_legend_title(column, options) for column in prepared.metadata_cols]
    max_label_len = max((len(label) for label in labels), default=0)
    label_width_inches = max_label_len * options.font_size_metadata / 72 * 0.62
    margin = 0.04 + label_width_inches / max(fig_width, 1)
    return min(0.30, max(0.14, margin))


def _legend_stack_step(
    handle_count: int,
    fontsize: float,
    *,
    figure_height: float,
    include_title: bool = True,
    max_step: float = 0.34,
) -> float:
    line_count = handle_count + (1 if include_title else 0)
    line_height = fontsize / 72 / max(figure_height, 1) * 1.35
    return min(max_step, 0.025 + line_height * line_count)


def _should_show_tmb_legend(
    prepared: PreparedOncoplotData,
    options: OncoplotOptions,
    render_stacked: bool,
    mutation_palette: Optional[Mapping[str, str]] = None,
    tmb_palette: Optional[Mapping[str, str]] = None,
    mutation_legend_visible: bool = True,
) -> bool:
    show = bool(
        prepared.tmb_is_custom
        and render_stacked
        and options.show_legend
        and options.mutation_legend_position != "none"
    )
    if not show or mutation_palette is None:
        return show
    if prepared.tmb is None or prepared.tmb_type_col is None or prepared.tiles.empty:
        return show
    if not mutation_legend_visible:
        return show

    tmb_categories = prepared.tmb_type_levels or [
        str(value)
        for value in pd.unique(prepared.tmb[prepared.tmb_type_col])
        if not pd.isna(value)
    ]
    mutation_categories = set(prepared.tiles["MutationType"].dropna().astype(str))
    if not tmb_categories or not set(tmb_categories).issubset(mutation_categories):
        return show

    tmb_colors = tmb_palette or mutation_palette
    duplicates_mutation_legend = all(
        tmb_colors.get(category) == mutation_palette.get(category)
        for category in tmb_categories
    )
    return not duplicates_mutation_legend


def _validate_metadata_levels(prepared: PreparedOncoplotData, options: OncoplotOptions) -> None:
    for track in prepared.metadata_tracks or []:
        if track.kind == "categorical" and len(track.levels) > options.metadata_max_levels:
            raise ValueError(
                f"Metadata column {track.column!r} has {len(track.levels)} levels, "
                f"which exceeds metadata_max_levels={options.metadata_max_levels}."
            )


def _draw_main(
    ax,
    prepared: PreparedOncoplotData,
    palette: Mapping[str, str],
    variant_value_palette: object,
    options: OncoplotOptions,
):
    _plt, _ListedColormap, _GridSpec, _Patch, Rectangle = _require_matplotlib()
    continuous_cmap = None
    continuous_norm = None
    if prepared.variant_value_col is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        min_value = prepared.variant_value_min if prepared.variant_value_min is not None else 0.0
        max_value = prepared.variant_value_max if prepared.variant_value_max is not None else 1.0
        if np.isclose(min_value, max_value):
            padding = max(0.5, abs(float(min_value)) * 0.05)
            min_value = float(min_value) - padding
            max_value = float(max_value) + padding
        continuous_cmap = coerce_continuous_colormap(variant_value_palette, prepared.variant_value_col)
        continuous_norm = Normalize(vmin=float(min_value), vmax=float(max_value))
        if options.show_legend and options.mutation_legend_position != "none":
            mappable = ScalarMappable(norm=continuous_norm, cmap=continuous_cmap)
            mappable.set_array([])
            colorbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.025, pad=0.012)
            colorbar.set_label(_legend_title(prepared.variant_value_col, options), fontsize=options.font_size_metadata)
            colorbar.ax.tick_params(labelsize=options.font_size_metadata)
    n_genes = len(prepared.genes)
    n_samples = len(prepared.samples)
    pathway_width = 0.95 if prepared.pathway_groups else 0
    ax.set_xlim(-pathway_width, n_samples)
    ax.set_ylim(0, n_genes)
    ax.invert_yaxis()
    ax.set_facecolor(options.background_color)

    for gene_index, gene in enumerate(prepared.genes):
        for sample_index, _sample in enumerate(prepared.samples):
            ax.add_patch(
                Rectangle(
                    (sample_index, gene_index),
                    options.tile_width,
                    options.tile_height,
                    facecolor=options.background_color,
                    edgecolor="white",
                    linewidth=options.tile_linewidth,
                )
            )

    sample_pos = {sample: index for index, sample in enumerate(prepared.samples)}
    gene_pos = {gene: index for index, gene in enumerate(prepared.genes)}
    for _index, row in prepared.tiles.iterrows():
        sample = str(row["Sample"])
        gene = str(row["Gene"])
        if sample not in sample_pos or gene not in gene_pos:
            continue
        if continuous_cmap is not None and continuous_norm is not None:
            color = continuous_cmap(continuous_norm(float(row["VariantValue"])))
        else:
            mutation_type = row["MutationType"]
            color = options.unspecified_mutation_color if pd.isna(mutation_type) else palette.get(str(mutation_type), options.unspecified_mutation_color)
        ax.add_patch(
            Rectangle(
                (sample_pos[sample], gene_pos[gene]),
                options.tile_width,
                options.tile_height,
                facecolor=color,
                edgecolor="white",
                linewidth=options.tile_linewidth,
            )
        )

    ax.hlines(
        np.arange(n_genes + 1),
        xmin=0,
        xmax=n_samples,
        colors="#666666",
        linewidth=options.row_separator_linewidth,
        alpha=0.75,
    )
    if prepared.pathway_groups:
        for group in prepared.pathway_groups:
            ax.add_patch(
                Rectangle(
                    (-pathway_width, group.start),
                    pathway_width,
                    group.end - group.start + 1,
                    facecolor=options.pathway_background_color,
                    edgecolor=options.pathway_outline_color,
                    linewidth=max(0.4, options.row_separator_linewidth),
                    clip_on=False,
                )
            )
            ax.text(
                -pathway_width / 2,
                group.start + (group.end - group.start + 1) / 2,
                group.name,
                ha="center",
                va="center",
                rotation=options.pathway_text_angle,
                color=options.pathway_text_color,
                fontsize=max(7, options.font_size_genes * 0.75),
                clip_on=False,
            )
            ax.hlines(
                [group.start, group.end + 1],
                xmin=-pathway_width,
                xmax=n_samples,
                colors=options.pathway_outline_color,
                linewidth=max(0.4, options.row_separator_linewidth),
                alpha=0.85,
            )
    ax.set_yticks(np.arange(n_genes) + 0.5)
    ax.set_yticklabels(prepared.genes, fontsize=options.font_size_genes)
    for label in ax.get_yticklabels():
        label.update(_font_kwargs(options.gene_font_style))
    if options.show_sample_ids:
        ax.set_xticks(np.arange(n_samples) + 0.5)
        ax.set_xticklabels(prepared.samples, rotation=options.sample_id_angle, fontsize=options.font_size_samples)
        if options.sample_id_position == "top":
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
        else:
            ax.xaxis.tick_bottom()
            ax.xaxis.set_label_position("bottom")
        for label in ax.get_xticklabels():
            label.update(_font_kwargs(options.sample_font_style))
    else:
        ax.set_xticks([])
    ax.set_xlabel(options.x_label if options.show_x_label else "", fontsize=options.font_size_x_label)
    ax.set_ylabel(options.y_label if options.show_y_label else "", fontsize=options.font_size_y_label)


def _draw_gene_bar(ax, prepared: PreparedOncoplotData, palette: Mapping[str, str], options: OncoplotOptions):
    tiles = prepared.tiles
    if tiles.empty:
        return
    y_positions = np.arange(len(prepared.genes)) + 0.5
    left = np.zeros(len(prepared.genes))
    mutation_groups = [
        (mutation_type, tiles[tiles["MutationType"].astype(str) == mutation_type])
        for mutation_type in _mutation_levels(prepared, palette)
    ]
    unspecified = tiles[tiles["MutationType"].isna()]
    if not unspecified.empty:
        mutation_groups.append((np.nan, unspecified))
    total_counts = tiles.groupby("Gene", observed=False).size().reindex(prepared.genes, fill_value=0).to_numpy(dtype=float)
    for mutation_type, group in mutation_groups:
        counts = group.groupby("Gene", observed=False).size().reindex(prepared.genes, fill_value=0).to_numpy(dtype=float)
        if options.gene_bar_mode == "percent":
            values = np.divide(counts, total_counts, out=np.zeros_like(counts), where=total_counts > 0) * 100
        else:
            values = counts
        color = options.unspecified_mutation_color if pd.isna(mutation_type) else palette.get(str(mutation_type), options.unspecified_mutation_color)
        ax.barh(y_positions, values, left=left, height=0.85, color=color)
        left = left + values
    if options.show_gene_bar_labels:
        totals = total_counts
        denominator = max(prepared.total_samples, 1)
        max_total = 100 if options.gene_bar_mode == "percent" else max(left.max(), 1)
        for y_position, total in zip(y_positions, totals):
            label_x = (100 if options.gene_bar_mode == "percent" else total)
            ax.text(
                label_x + max_total * options.gene_bar_label_padding + options.gene_bar_label_nudge,
                y_position,
                f"{round(total / denominator * 100, options.gene_bar_label_round):g}%",
                va="center",
                ha="left",
                fontsize=options.font_size_gene_bar_axis,
            )
        ax.set_xlim(0, max(left.max() * (1 + options.gene_bar_label_padding + 0.08), left.max() + 1))
    elif options.gene_bar_mode == "percent":
        ax.set_xlim(0, 100)
    ax.set_ylim(0, len(prepared.genes))
    ax.invert_yaxis()
    ax.set_yticks([])
    if options.show_gene_bar_axis:
        ax.set_xlabel(
            "Mutation Type (%)" if options.gene_bar_mode == "percent" else "Samples",
            fontsize=options.font_size_gene_bar_axis,
        )
        if options.gene_bar_scale_breaks is not None:
            ax.set_xticks(list(options.gene_bar_scale_breaks))
        elif options.gene_bar_scale_n_breaks is not None:
            ax.locator_params(axis="x", nbins=options.gene_bar_scale_n_breaks)
    else:
        ax.set_xticks([])
    if prepared.pathway_groups:
        for group in prepared.pathway_groups:
            ax.hlines(
                [group.start, group.end + 1],
                xmin=0,
                xmax=max(left.max(), 1),
                colors=options.pathway_outline_color,
                linewidth=max(0.4, options.row_separator_linewidth),
                alpha=0.65,
            )


def _draw_tmb(
    ax,
    prepared: PreparedOncoplotData,
    mutation_palette: Mapping[str, str],
    tmb_palette: Optional[Mapping[str, str]],
    options: OncoplotOptions,
    mutation_legend_visible: bool = True,
) -> List[object]:
    if prepared.tmb is None or prepared.tmb_sample_col is None or prepared.tmb_value_col is None:
        return []
    _plt, _ListedColormap, _GridSpec, Patch, _Rectangle = _require_matplotlib()
    sample_col = prepared.tmb_sample_col
    x_positions = np.arange(len(prepared.samples)) + 0.5
    render_stacked = prepared.tmb_render_stacked and not options.log10_transform_tmb and prepared.tmb_type_col is not None
    show_tmb_legend = _should_show_tmb_legend(
        prepared,
        options,
        render_stacked,
        mutation_palette,
        tmb_palette,
        mutation_legend_visible=mutation_legend_visible,
    )
    handles: List[object] = []
    if prepared.tmb_render_stacked and options.log10_transform_tmb:
        warnings.warn(
            "log10_transform_tmb=True disables stacked TMB rendering; totals are rendered instead.",
            stacklevel=2,
        )
    if render_stacked:
        bottoms = np.zeros(len(prepared.samples))
        type_col = prepared.tmb_type_col
        palette = tmb_palette or mutation_palette
        for mutation_type in _tmb_levels(prepared, palette):
            group = prepared.tmb[prepared.tmb[type_col].astype(str) == mutation_type]
            values = (
                group.groupby(sample_col, observed=False)[prepared.tmb_value_col]
                .sum()
                .reindex(prepared.samples, fill_value=0)
                .to_numpy(dtype=float)
            )
            color = "#4D4D4D" if pd.isna(mutation_type) else palette.get(str(mutation_type), "#4D4D4D")
            ax.bar(x_positions, values, bottom=bottoms, color=color, width=1.0, linewidth=0)
            if show_tmb_legend:
                handles.append(Patch(color=color, label=_legend_label(mutation_type, options)))
            bottoms = bottoms + values
    else:
        totals = prepared.tmb.groupby(sample_col, observed=False)[prepared.tmb_value_col].sum().reindex(prepared.samples, fill_value=0)
        values = totals.to_numpy(dtype=float)
        if options.log10_transform_tmb:
            values = np.log10(np.maximum(values, 1))
        ax.bar(x_positions, values, color="#4D4D4D", width=1.0, linewidth=0)
    ax.set_xlim(0, len(prepared.samples))
    ax.set_xticks([])
    if options.show_tmb_y_label:
        axis_label = prepared.tmb_value_col
        if options.log10_transform_tmb:
            axis_label = f"log10 {axis_label}"
        ax.set_ylabel(axis_label, fontsize=options.font_size_tmb_axis)
    ax.tick_params(axis="y", labelsize=options.font_size_tmb_axis)
    if options.scientific_tmb:
        from matplotlib.ticker import FormatStrFormatter

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    if not options.show_tmb_axis:
        ax.set_yticks([])
    return handles


def _draw_metadata(
    ax,
    prepared: PreparedOncoplotData,
    options: OncoplotOptions,
    metadata_palette: Optional[Mapping[str, Any]] = None,
) -> List[tuple[str, List[object]]]:
    if prepared.metadata is None or not prepared.metadata_cols:
        return []
    _plt, _ListedColormap, _GridSpec, Patch, Rectangle = _require_matplotlib()
    metadata = prepared.metadata.set_index("Sample")
    legends: List[tuple[str, List[object]]] = []
    ax.set_xlim(0, len(prepared.samples))
    ax.set_ylim(0, len(prepared.metadata_cols))
    ax.invert_yaxis()
    ax.set_facecolor("white")
    metadata_palette = metadata_palette or {}

    for row_index, column in enumerate(prepared.metadata_cols):
        display_column = _legend_title(column, options)
        values = metadata[column].reindex(prepared.samples)
        is_numeric = pd.api.types.is_numeric_dtype(values)
        min_value = float(values.min(skipna=True)) if is_numeric and values.notna().any() else 0.0
        max_value = float(values.max(skipna=True)) if is_numeric and values.notna().any() else 1.0
        span = max(max_value - min_value, 1e-9)
        palette_spec = metadata_palette_spec(metadata_palette, column)
        numeric_colormap = numeric_metadata_colormap_spec(palette_spec) if is_numeric else None
        if is_numeric and options.metadata_numeric_plot_type == "bar":
            ax.add_patch(
                Rectangle(
                    (0, row_index),
                    len(prepared.samples),
                    1,
                    facecolor="#F4F4F4",
                    edgecolor="white",
                    linewidth=0,
                )
            )
            for col_index, sample in enumerate(prepared.samples):
                value = values.get(sample, np.nan)
                if pd.isna(value):
                    ax.add_patch(Rectangle((col_index, row_index), 1, 1, facecolor="#D9D9D9", edgecolor="white", linewidth=0.1))
                    ax.text(
                        col_index + 0.5,
                        row_index + 0.55,
                        options.metadata_na_marker,
                        ha="center",
                        va="center",
                        fontsize=options.metadata_na_marker_size,
                        color="#333333",
                    )
                    continue
                frac = (float(value) - min_value) / span
                color = (
                    sample_numeric_metadata_colormap(numeric_colormap, value, min_value, max_value, column)
                    if numeric_colormap is not None
                    else "#7F7F7F"
                )
                ax.add_patch(
                    Rectangle(
                        (col_index, row_index + (1 - frac)),
                        1,
                        frac,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.1,
                    )
                )
            ax.text(
                len(prepared.samples) + 0.12,
                row_index + 0.25,
                f"{max_value:g}",
                ha="left",
                va="center",
                fontsize=options.font_size_metadata_bar_numbers,
                clip_on=False,
            )
            ax.text(
                len(prepared.samples) + 0.12,
                row_index + 0.78,
                f"{min_value:g}",
                ha="left",
                va="center",
                fontsize=options.font_size_metadata_bar_numbers,
                clip_on=False,
            )
            if numeric_colormap is not None:
                legend_low, legend_high = numeric_metadata_endpoint_colors(numeric_colormap, column)
                legends.append(
                    (
                        display_column,
                        [
                            Patch(color=legend_low, label=f"{min_value:g}"),
                            Patch(color=legend_high, label=f"{max_value:g}"),
                        ],
                    )
                )
            else:
                legends.append((display_column, [Patch(color="#7F7F7F", label=f"{min_value:g}-{max_value:g}")]))
            continue

        if is_numeric:
            numeric_values = values.astype(float)
            min_value = float(numeric_values.min(skipna=True)) if numeric_values.notna().any() else 0.0
            max_value = float(numeric_values.max(skipna=True)) if numeric_values.notna().any() else 1.0
            span = max(max_value - min_value, 1e-9)
            colours = options.metadata_default_colors
            for col_index, sample in enumerate(prepared.samples):
                value = numeric_values.get(sample, np.nan)
                if pd.isna(value):
                    color = "#D9D9D9"
                elif numeric_colormap is not None:
                    color = sample_numeric_metadata_colormap(numeric_colormap, value, min_value, max_value, column)
                else:
                    bucket = min(len(colours) - 1, int((float(value) - min_value) / span * len(colours)))
                    color = colours[bucket]
                ax.add_patch(Rectangle((col_index, row_index), 1, 1, facecolor=color, edgecolor="white", linewidth=0.1))
                if pd.isna(value):
                    ax.text(
                        col_index + 0.5,
                        row_index + 0.55,
                        options.metadata_na_marker,
                        ha="center",
                        va="center",
                        fontsize=options.metadata_na_marker_size,
                        color="#333333",
                    )
            legend_low, legend_high = (
                numeric_metadata_endpoint_colors(numeric_colormap, column)
                if numeric_colormap is not None
                else (options.metadata_default_colors[0], options.metadata_default_colors[-1])
            )
            legends.append(
                (
                    display_column,
                    [
                        Patch(color=legend_low, label=f"{min_value:g}"),
                        Patch(color=legend_high, label=f"{max_value:g}"),
                    ],
                )
            )
            continue

        levels = _metadata_levels(prepared, column, values, palette_spec)
        column_palette = categorical_metadata_palette(palette_spec, column, levels)
        for level_index, level in enumerate(levels):
            column_palette.setdefault(level, options.metadata_default_colors[level_index % len(options.metadata_default_colors)])
        column_palette = {level: column_palette[level] for level in levels}
        for col_index, sample in enumerate(prepared.samples):
            value = values.get(sample, np.nan)
            color = "#D9D9D9" if pd.isna(value) else column_palette[str(value)]
            ax.add_patch(Rectangle((col_index, row_index), 1, 1, facecolor=color, edgecolor="white", linewidth=0.1))
            if pd.isna(value):
                ax.text(
                    col_index + 0.5,
                    row_index + 0.55,
                    options.metadata_na_marker,
                    ha="center",
                    va="center",
                    fontsize=options.metadata_na_marker_size,
                    color="#333333",
                )
        handles = [
            Patch(color=color, label=_legend_label(level, options))
            for level, color in column_palette.items()
        ]
        if values.isna().any():
            handles.append(Patch(color="#D9D9D9", label=options.metadata_na_marker))
        legends.append((display_column, handles))

    ax.set_yticks(np.arange(len(prepared.metadata_cols)))
    ax.set_yticklabels(
        [_legend_title(column, options) for column in prepared.metadata_cols],
        fontsize=options.font_size_metadata,
    )
    for label in ax.get_yticklabels():
        label.update(_font_kwargs(options.font_style_metadata))
    ax.set_xticks([])
    return legends


def _add_static_legends(
    figure,
    mutation_handles,
    tmb_handles,
    metadata_legends,
    options: OncoplotOptions,
) -> None:
    large_figure = options.width >= 1600 or options.height >= 1200
    mutation_base = 10 if large_figure else 8
    metadata_base = 9 if large_figure else 7
    mutation_scale = 1.2 if large_figure else 0.75
    metadata_scale = 1.2 if large_figure else 0.85
    title_padding = 4 if large_figure else 1
    figure_height = figure.get_figheight()
    mutation_fontsize = _legend_font_size(mutation_base, options.font_size_genes, scale=mutation_scale)
    mutation_title_fontsize = mutation_fontsize + title_padding
    metadata_fontsize = _legend_font_size(metadata_base, options.font_size_metadata, scale=metadata_scale)
    metadata_title_fontsize = metadata_fontsize + title_padding
    if options.show_legend and mutation_handles and options.mutation_legend_position == "bottom":
        figure.legend(
            handles=mutation_handles,
            loc="lower center",
            bbox_to_anchor=(0.42, 0.01),
            ncol=min(5, len(mutation_handles)),
            frameon=True,
            fontsize=mutation_fontsize,
            title="Mutation Type" if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )
    if options.show_legend and tmb_handles and options.mutation_legend_position == "bottom":
        figure.legend(
            handles=tmb_handles,
            loc="lower right",
            bbox_to_anchor=(0.99, 0.01),
            ncol=min(4, len(tmb_handles)),
            frameon=True,
            fontsize=mutation_fontsize,
            title="TMB Type" if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )

    right_y = 0.96
    if options.show_legend and mutation_handles and options.mutation_legend_position == "right":
        figure.legend(
            handles=mutation_handles,
            loc="upper left",
            bbox_to_anchor=(0.74, right_y),
            ncol=1,
            frameon=False,
            fontsize=mutation_fontsize,
            title="Mutation Type" if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )
        right_y -= _legend_stack_step(len(mutation_handles), mutation_fontsize, figure_height=figure_height)
    if options.show_legend and tmb_handles and options.mutation_legend_position == "right":
        figure.legend(
            handles=tmb_handles,
            loc="upper left",
            bbox_to_anchor=(0.74, right_y),
            ncol=1,
            frameon=False,
            fontsize=mutation_fontsize,
            title="TMB Type" if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )
        right_y -= _legend_stack_step(len(tmb_handles), mutation_fontsize, figure_height=figure_height)

    if not options.show_metadata_legends or not metadata_legends:
        return

    if options.metadata_legend_position == "bottom":
        handles = []
        for title, title_handles in metadata_legends:
            handles.extend(title_handles[: max(1, min(3, len(title_handles)))])
        if handles:
            ncol = options.metadata_legend_ncol or 3
            figure.legend(
                handles=handles,
                loc="lower right",
                bbox_to_anchor=(0.99, 0.01),
                ncol=ncol,
                frameon=True,
                fontsize=metadata_fontsize,
                handlelength=options.metadata_legend_key_size,
                handleheight=options.metadata_legend_key_size,
            )
        return

    y = right_y
    for title, handles in metadata_legends:
        if not handles:
            continue
        ncol = options.metadata_legend_ncol or 1
        if options.metadata_legend_nrow:
            ncol = max(ncol, int(np.ceil(len(handles) / options.metadata_legend_nrow)))
        figure.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.74, y),
            ncol=ncol,
            frameon=False,
            fontsize=metadata_fontsize,
            title=title if options.show_legend_titles else None,
            title_fontsize=metadata_title_fontsize,
            handlelength=options.metadata_legend_key_size,
            handleheight=options.metadata_legend_key_size,
        )
        y -= _legend_stack_step(len(handles), metadata_fontsize, figure_height=figure_height)
        if y < 0.05:
            break


def render_matplotlib_oncoplot(
    prepared: Optional[PreparedOncoplotData] = None,
    *,
    params: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> object:
    """Render a static Matplotlib oncoplot."""

    supplied = merge_params(params, allowed_keys=MATPLOTLIB_RENDER_PARAM_KEYS, context="matplotlib renderer", **kwargs)
    merged = {**MATPLOTLIB_RENDER_DEFAULTS, **supplied}
    if prepared is None:
        prepared = merged.pop("prepared", None)
    else:
        merged.pop("prepared", None)
    if prepared is None:
        raise TypeError("render_matplotlib_oncoplot requires prepared data as the first argument or params['prepared'].")
    if "palette" not in merged:
        raise TypeError("render_matplotlib_oncoplot requires 'palette'.")
    if "options" not in merged:
        raise TypeError("render_matplotlib_oncoplot requires 'options'.")

    palette = merged["palette"]
    tmb_palette = merged["tmb_palette"]
    metadata_palette = merged["metadata_palette"]
    variant_value_palette = merged["variant_value_palette"]
    options = coerce_options(merged["options"])
    draw_gene_bar = merged["draw_gene_bar"]
    draw_tmb_bar = merged["draw_tmb_bar"]

    plt, _ListedColormap, GridSpec, Patch, _Rectangle = _require_matplotlib()
    draw_metadata = prepared.metadata is not None and bool(prepared.metadata_cols)
    if draw_metadata:
        _validate_metadata_levels(prepared, options)
    rows = []
    if options.metadata_position == "top" and draw_metadata:
        rows.append(("metadata", max(1, len(prepared.metadata_cols or []))))
    if draw_tmb_bar:
        rows.append(("tmb", 1.2))
    rows.append(("main", max(3, len(prepared.genes) * 0.55)))
    if options.metadata_position == "bottom" and draw_metadata:
        rows.append(("metadata", max(1, len(prepared.metadata_cols or []))))

    fig_width = max(8, options.width / 120)
    fig_height = max(5, options.height / 120)
    figure = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    ncols = 2 if draw_gene_bar else 1
    width_ratios = [1 - options.gene_bar_width_ratio, options.gene_bar_width_ratio] if draw_gene_bar else [1]
    grid = GridSpec(
        len(rows),
        ncols,
        figure=figure,
        height_ratios=[weight for _name, weight in rows],
        width_ratios=width_ratios,
        hspace=max(0.02, min(0.6, max(options.buffer_tmb, options.buffer_metadata))),
        wspace=max(0.01, min(0.4, options.buffer_gene_bar)),
    )

    axes = {}
    for row_index, (name, _weight) in enumerate(rows):
        axes[name] = figure.add_subplot(grid[row_index, 0])
        if name == "main" and draw_gene_bar:
            axes["gene_bar"] = figure.add_subplot(grid[row_index, 1])

    tmb_handles = []
    mutation_legend_visible = prepared.variant_value_col is None or draw_gene_bar
    if draw_tmb_bar and "tmb" in axes:
        tmb_handles = _draw_tmb(
            axes["tmb"],
            prepared,
            palette,
            tmb_palette,
            options,
            mutation_legend_visible=mutation_legend_visible,
        )
    metadata_legends = []
    if draw_metadata and "metadata" in axes:
        metadata_legends = _draw_metadata(axes["metadata"], prepared, options, metadata_palette=metadata_palette)
    _draw_main(axes["main"], prepared, palette, variant_value_palette, options)
    if draw_gene_bar and "gene_bar" in axes:
        _draw_gene_bar(axes["gene_bar"], prepared, palette, options)
        if draw_metadata and options.metadata_position == "bottom" and options.show_gene_bar_axis:
            axes["gene_bar"].xaxis.tick_top()
            axes["gene_bar"].xaxis.set_label_position("top")

    mutation_handles = []
    if (
        options.show_legend
        and options.mutation_legend_position != "none"
        and (prepared.variant_value_col is None or draw_gene_bar)
    ):
        mutation_handles = [
            Patch(color=palette.get(name, options.unspecified_mutation_color), label=_legend_label(name, options))
            for name in _mutation_levels(prepared, palette)
        ]
    has_bottom_legend = bool((mutation_handles or tmb_handles) and options.mutation_legend_position == "bottom")
    has_right_legend = bool(
        ((mutation_handles or tmb_handles) and options.mutation_legend_position == "right")
        or (metadata_legends and options.show_metadata_legends and options.metadata_legend_position == "right")
    )
    variant_colorbar_active = (
        prepared.variant_value_col is not None
        and options.show_legend
        and options.mutation_legend_position != "none"
    )
    _add_static_legends(figure, mutation_handles, tmb_handles, metadata_legends, options)
    figure.subplots_adjust(
        left=_left_margin_for_metadata(prepared, options, draw_metadata=draw_metadata, fig_width=fig_width),
        right=0.70 if has_right_legend else (0.88 if variant_colorbar_active else 0.98),
        top=0.92,
        bottom=0.18 if has_bottom_legend else 0.08,
        hspace=max(0.02, min(0.6, max(options.buffer_tmb, options.buffer_metadata))),
        wspace=max(0.01, min(0.4, options.buffer_gene_bar)),
    )
    return figure
