"""Matplotlib renderer for static oncoplots."""

from __future__ import annotations

from dataclasses import dataclass
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

RIGHT_LEGEND_LEFT = 0.74
RIGHT_LEGEND_RIGHT = 0.98


@dataclass(frozen=True)
class _MetadataLegendSpec:
    title: str
    key: str
    handles: Optional[List[object]] = None
    colormap: object = None
    min_value: float = 0.0
    max_value: float = 1.0

    @property
    def is_numeric(self) -> bool:
        return self.colormap is not None


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


def _truncate_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


def _legend_label(value: object, options: OncoplotOptions) -> str:
    text = "Unspecified" if pd.isna(value) else str(value)
    return options.prettify_function(text) if options.prettify_legend_values else text


def _legend_title(value: object, options: OncoplotOptions) -> str:
    text = "" if value is None else str(value)
    return options.prettify_function(text) if options.prettify_legend_titles else text


def _legend_entry_label(value: object, options: OncoplotOptions) -> str:
    return _truncate_text(_legend_label(value, options), options.legend_label_max_chars)


def _legend_entry_title(value: object, options: OncoplotOptions) -> str:
    return _truncate_text(_legend_title(value, options), options.legend_title_max_chars)


def _legend_offset(options: OncoplotOptions, key: str, fallback_key: Optional[str] = None) -> tuple[float, float]:
    offsets = options.legend_offsets
    offset = offsets.get(key)
    if offset is None and fallback_key is not None:
        offset = offsets.get(fallback_key)
    if offset is None:
        return 0.0, 0.0
    return float(offset.get("x", 0.0)), float(offset.get("y", 0.0))


def _offset_anchor(
    anchor: tuple[float, float],
    options: OncoplotOptions,
    key: str,
    fallback_key: Optional[str] = None,
) -> tuple[float, float]:
    x_offset, y_offset = _legend_offset(options, key, fallback_key)
    return anchor[0] + x_offset, anchor[1] + y_offset


def _offset_bounds(
    bounds: Sequence[float],
    options: OncoplotOptions,
    key: str,
    fallback_key: Optional[str] = None,
) -> list[float]:
    x_offset, y_offset = _legend_offset(options, key, fallback_key)
    return [bounds[0] + x_offset, bounds[1] + y_offset, bounds[2], bounds[3]]


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


def _metadata_colorbar_is_horizontal(options: OncoplotOptions) -> bool:
    return (
        options.metadata_legend_position == "bottom"
        or options.metadata_legend_orientation_heatmap == "horizontal"
    )


def _numeric_colorbar_bounds(min_value: float, max_value: float) -> tuple[float, float, list[float], list[str]]:
    if np.isclose(min_value, max_value):
        padding = max(0.5, abs(float(min_value)) * 0.05)
        return (
            float(min_value) - padding,
            float(max_value) + padding,
            [float(min_value)],
            [f"{min_value:g}"],
        )
    return (
        float(min_value),
        float(max_value),
        [float(min_value), float(max_value)],
        [f"{min_value:g}", f"{max_value:g}"],
    )


def _numeric_metadata_legend_spec(
    title: str,
    column: object,
    colormap_spec: object,
    min_value: float,
    max_value: float,
) -> _MetadataLegendSpec:
    return _MetadataLegendSpec(
        title=title,
        key=f"metadata:{column}",
        colormap=coerce_continuous_colormap(colormap_spec, column),
        min_value=min_value,
        max_value=max_value,
    )


def _metadata_has_numeric_legends(metadata_legends: Sequence[_MetadataLegendSpec]) -> bool:
    return any(legend.is_numeric for legend in metadata_legends)


def _metadata_has_bottom_legends(
    metadata_legends: Sequence[_MetadataLegendSpec],
    options: OncoplotOptions,
) -> bool:
    if not options.show_metadata_legends or not metadata_legends:
        return False
    if options.metadata_legend_position == "bottom":
        return True
    return _metadata_has_numeric_legends(metadata_legends) and _metadata_colorbar_is_horizontal(options)


def _metadata_has_right_legends(
    metadata_legends: Sequence[_MetadataLegendSpec],
    options: OncoplotOptions,
) -> bool:
    if not options.show_metadata_legends or not metadata_legends or options.metadata_legend_position != "right":
        return False
    if not _metadata_colorbar_is_horizontal(options):
        return True
    return any(not legend.is_numeric for legend in metadata_legends)


def _centered_right_legend_x(width: float) -> float:
    return RIGHT_LEGEND_LEFT + max(0.0, (RIGHT_LEGEND_RIGHT - RIGHT_LEGEND_LEFT - width) / 2)


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


def _has_expanded_main_grid(prepared: PreparedOncoplotData) -> bool:
    return prepared.main_grid_mode == "expanded" and prepared.main_grid_rows is not None


def _main_grid_has_mutation_rows(prepared: PreparedOncoplotData) -> bool:
    if _has_expanded_main_grid(prepared):
        rows = prepared.main_grid_rows
        return bool(rows is not None and not rows.empty and (rows["Kind"] == "mutation_type").any())
    return prepared.variant_value_col is None


def _main_grid_has_continuous_rows(prepared: PreparedOncoplotData) -> bool:
    if _has_expanded_main_grid(prepared):
        rows = prepared.main_grid_rows
        return bool(rows is not None and not rows.empty and (rows["Kind"] == "variant_value").any())
    return prepared.variant_value_col is not None


def _main_grid_gene_centers(prepared: PreparedOncoplotData) -> dict[str, float]:
    rows = prepared.main_grid_rows
    if rows is None or rows.empty:
        return {gene: index + 0.5 for index, gene in enumerate(prepared.genes)}
    centers = {}
    for gene, group in rows.groupby("Gene", sort=False, observed=False):
        centers[str(gene)] = (float(group["RowIndex"].min()) + float(group["RowIndex"].max()) + 1) / 2
    return centers


def _continuous_track_groups(prepared: PreparedOncoplotData) -> list[tuple[str, pd.DataFrame]]:
    rows = prepared.main_grid_rows
    if rows is None or rows.empty:
        return []
    track_rows = (
        rows[rows["Kind"] == "variant_value"]
        .sort_values("TrackIndex", kind="mergesort")
        .drop_duplicates("TrackId")
    )
    if track_rows.empty:
        return []
    if prepared.variant_value_scale == "shared":
        return [("variant_value_shared", track_rows)]
    return [(str(track_id), group) for track_id, group in track_rows.groupby("TrackId", sort=False)]


def _add_variant_value_colorbar(
    ax,
    colormap,
    min_value: float,
    max_value: float,
    title: str,
    legend_key: str,
    options: OncoplotOptions,
) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cmin, cmax, tick_values, tick_labels = _numeric_colorbar_bounds(min_value, max_value)
    mappable = ScalarMappable(norm=Normalize(vmin=cmin, vmax=cmax), cmap=colormap)
    mappable.set_array([])
    colorbar = ax.figure.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        fraction=0.06,
        pad=0.08,
        aspect=30,
        ticks=tick_values,
    )
    if legend_key in options.legend_offsets or "variant" in options.legend_offsets:
        colorbar.ax.set_position(_offset_bounds(colorbar.ax.get_position().bounds, options, legend_key, "variant"))
    colorbar.ax.set_xticklabels(tick_labels)
    fontsize = options.font_size_legend_text or options.font_size_metadata
    title_fontsize = options.font_size_legend_title or fontsize
    colorbar.ax.set_title(
        _truncate_text(title, options.legend_title_max_chars),
        fontsize=title_fontsize,
        pad=max(2, title_fontsize * 0.35),
    )
    colorbar.ax.tick_params(labelsize=fontsize)


def _is_missing_palette_spec(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _draw_expanded_main(
    ax,
    prepared: PreparedOncoplotData,
    palette: Mapping[str, str],
    variant_value_palette: object,
    options: OncoplotOptions,
) -> None:
    _plt, _ListedColormap, _GridSpec, _Patch, Rectangle = _require_matplotlib()
    rows = prepared.main_grid_rows
    tiles = prepared.main_grid_tiles
    if rows is None:
        return
    n_rows = len(rows)
    n_samples = len(prepared.samples)
    track_count = int(rows["TrackId"].nunique()) if not rows.empty else 1
    pathway_width = 0.95 if prepared.pathway_groups else 0
    subrow_label_width = 0.95 if track_count > 1 else 0
    ax.set_xlim(-(pathway_width + subrow_label_width), n_samples)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_facecolor(options.background_color)

    for row_index in range(n_rows):
        for sample_index, _sample in enumerate(prepared.samples):
            ax.add_patch(
                Rectangle(
                    (sample_index, row_index),
                    options.tile_width,
                    options.tile_height,
                    facecolor=options.background_color,
                    edgecolor="white",
                    linewidth=options.tile_linewidth,
                )
            )

    continuous_groups = _continuous_track_groups(prepared)
    continuous_maps: dict[str, tuple[object, object]] = {}
    if continuous_groups and tiles is not None and not tiles.empty:
        from matplotlib.colors import Normalize

        continuous_tiles = tiles[tiles["Kind"] == "variant_value"]
        for _group_id, track_rows in continuous_groups:
            track_ids = set(track_rows["TrackId"].astype(str))
            group_tiles = continuous_tiles[continuous_tiles["TrackId"].astype(str).isin(track_ids)]
            non_missing_tiles = group_tiles[group_tiles["VariantValue"].notna()]
            if non_missing_tiles.empty:
                continue
            min_value = float(non_missing_tiles["VariantValueMin"].dropna().iloc[0])
            max_value = float(non_missing_tiles["VariantValueMax"].dropna().iloc[0])
            cmin, cmax, _tick_values, _tick_labels = _numeric_colorbar_bounds(min_value, max_value)
            first_track = track_rows.iloc[0]
            if prepared.variant_value_scale == "shared":
                palette_spec = variant_value_palette
                title = (
                    "Variant value"
                    if track_rows["TrackId"].nunique() > 1
                    else _legend_title(first_track["Label"], options)
                )
                legend_key = (
                    "variant:shared"
                    if track_rows["TrackId"].nunique() > 1
                    else f"variant:{first_track['VariantValueColumn']}"
                )
            else:
                palette_spec = first_track["VariantValuePalette"]
                if _is_missing_palette_spec(palette_spec):
                    palette_spec = variant_value_palette
                title = _legend_title(first_track["Label"], options)
                legend_key = f"variant:{first_track['VariantValueColumn']}"
            cmap = coerce_continuous_colormap(palette_spec, first_track["VariantValueColumn"])
            norm = Normalize(vmin=cmin, vmax=cmax)
            for track_id in track_ids:
                continuous_maps[track_id] = (cmap, norm)
            if options.show_legend and options.mutation_legend_position != "none":
                _add_variant_value_colorbar(ax, cmap, min_value, max_value, title, legend_key, options)

    sample_pos = {sample: index for index, sample in enumerate(prepared.samples)}
    if tiles is not None and not tiles.empty:
        for _index, tile in tiles.iterrows():
            sample = str(tile["Sample"])
            if sample not in sample_pos:
                continue
            row_index = int(tile["RowIndex"])
            if tile["Kind"] == "variant_value":
                if pd.isna(tile["VariantValue"]) or str(tile["TrackId"]) not in continuous_maps:
                    continue
                cmap, norm = continuous_maps[str(tile["TrackId"])]
                color = cmap(norm(float(tile["VariantValue"])))
            else:
                mutation_type = tile["MutationType"]
                color = (
                    options.unspecified_mutation_color
                    if pd.isna(mutation_type)
                    else palette.get(str(mutation_type), options.unspecified_mutation_color)
                )
            ax.add_patch(
                Rectangle(
                    (sample_pos[sample], row_index),
                    options.tile_width,
                    options.tile_height,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=options.tile_linewidth,
                )
            )

    for _gene, group in rows.groupby("Gene", sort=False, observed=False):
        ax.hlines(
            [float(group["RowIndex"].min()), float(group["RowIndex"].max()) + 1],
            xmin=0,
            xmax=n_samples,
            colors="#666666",
            linewidth=options.row_separator_linewidth,
            alpha=0.75,
        )

    if track_count > 1:
        label_x = -0.08
        for _index, row_spec in rows.sort_values("RowIndex", kind="mergesort").iterrows():
            ax.text(
                label_x,
                float(row_spec["RowIndex"]) + 0.5,
                _legend_title(row_spec["Label"], options),
                ha="right",
                va="center",
                fontsize=max(7, options.font_size_genes * 0.72),
                color="#555555",
                clip_on=False,
            )

    if prepared.pathway_groups:
        for group in prepared.pathway_groups:
            group_rows = rows[rows["Gene"].astype(str).isin(group.genes)]
            if group_rows.empty:
                continue
            start = float(group_rows["RowIndex"].min())
            end = float(group_rows["RowIndex"].max()) + 1
            ax.add_patch(
                Rectangle(
                    (-(pathway_width + subrow_label_width), start),
                    pathway_width,
                    end - start,
                    facecolor=options.pathway_background_color,
                    edgecolor=options.pathway_outline_color,
                    linewidth=max(0.4, options.row_separator_linewidth),
                    clip_on=False,
                )
            )
            ax.text(
                -(pathway_width + subrow_label_width) + pathway_width / 2,
                start + (end - start) / 2,
                group.name,
                ha="center",
                va="center",
                rotation=options.pathway_text_angle,
                color=options.pathway_text_color,
                fontsize=options.font_size_pathway or max(7, options.font_size_genes * 0.75),
                clip_on=False,
            )
            ax.hlines(
                [start, end],
                xmin=-(pathway_width + subrow_label_width),
                xmax=n_samples,
                colors=options.pathway_outline_color,
                linewidth=max(0.4, options.row_separator_linewidth),
                alpha=0.85,
            )

    gene_centers = _main_grid_gene_centers(prepared)
    ax.set_yticks([])
    gene_label_shift = -(54 + options.gene_name_x_offset)
    for gene in prepared.genes:
        if gene not in gene_centers:
            continue
        ax.annotate(
            gene,
            xy=(0, gene_centers[gene]),
            xytext=(gene_label_shift, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=options.font_size_genes,
            clip_on=False,
            **_font_kwargs(options.gene_font_style),
        )
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


def _draw_main(
    ax,
    prepared: PreparedOncoplotData,
    palette: Mapping[str, str],
    variant_value_palette: object,
    options: OncoplotOptions,
):
    _plt, _ListedColormap, _GridSpec, _Patch, Rectangle = _require_matplotlib()
    if _has_expanded_main_grid(prepared):
        _draw_expanded_main(ax, prepared, palette, variant_value_palette, options)
        return

    continuous_cmap = None
    continuous_norm = None
    if prepared.variant_value_col is not None and prepared.variant_value_min is not None and prepared.variant_value_max is not None:
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
            _add_variant_value_colorbar(
                ax,
                continuous_cmap,
                float(min_value),
                float(max_value),
                _legend_title(prepared.variant_value_col, options),
                f"variant:{prepared.variant_value_col}",
                options,
            )
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
        if prepared.variant_value_col is not None:
            if continuous_cmap is None or continuous_norm is None or pd.isna(row["VariantValue"]):
                continue
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
                fontsize=options.font_size_pathway or max(7, options.font_size_genes * 0.75),
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
    expanded = _has_expanded_main_grid(prepared)
    if expanded:
        gene_centers = _main_grid_gene_centers(prepared)
        y_positions = np.array([gene_centers[gene] for gene in prepared.genes], dtype=float)
        row_counts = (
            prepared.main_grid_rows.groupby("Gene", sort=False, observed=False)["RowIndex"].nunique()
            if prepared.main_grid_rows is not None and not prepared.main_grid_rows.empty
            else pd.Series(1, index=prepared.genes)
        )
        bar_heights = np.array([max(0.75, float(row_counts.get(gene, 1)) * 0.84) for gene in prepared.genes])
        y_limit = len(prepared.main_grid_rows) if prepared.main_grid_rows is not None else len(prepared.genes)
    else:
        y_positions = np.arange(len(prepared.genes)) + 0.5
        bar_heights = 0.85
        y_limit = len(prepared.genes)
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
        ax.barh(y_positions, values, left=left, height=bar_heights, color=color)
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
    ax.set_ylim(0, y_limit)
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
            if expanded and prepared.main_grid_rows is not None:
                group_rows = prepared.main_grid_rows[
                    prepared.main_grid_rows["Gene"].astype(str).isin(group.genes)
                ]
                boundaries = [float(group_rows["RowIndex"].min()), float(group_rows["RowIndex"].max()) + 1]
            else:
                boundaries = [group.start, group.end + 1]
            ax.hlines(
                boundaries,
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
                handles.append(Patch(color=color, label=_legend_entry_label(mutation_type, options)))
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
) -> List[_MetadataLegendSpec]:
    if prepared.metadata is None or not prepared.metadata_cols:
        return []
    _plt, _ListedColormap, _GridSpec, Patch, Rectangle = _require_matplotlib()
    metadata = prepared.metadata.set_index("Sample")
    legends: List[_MetadataLegendSpec] = []
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
        numeric_color_spec = numeric_colormap if numeric_colormap is not None else options.metadata_default_colors
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
                color = sample_numeric_metadata_colormap(numeric_color_spec, value, min_value, max_value, column)
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
            legends.append(
                _numeric_metadata_legend_spec(
                    display_column,
                    column,
                    numeric_color_spec,
                    min_value,
                    max_value,
                )
            )
            continue

        if is_numeric:
            numeric_values = values.astype(float)
            min_value = float(numeric_values.min(skipna=True)) if numeric_values.notna().any() else 0.0
            max_value = float(numeric_values.max(skipna=True)) if numeric_values.notna().any() else 1.0
            for col_index, sample in enumerate(prepared.samples):
                value = numeric_values.get(sample, np.nan)
                if pd.isna(value):
                    color = "#D9D9D9"
                else:
                    color = sample_numeric_metadata_colormap(numeric_color_spec, value, min_value, max_value, column)
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
            legends.append(
                _numeric_metadata_legend_spec(
                    display_column,
                    column,
                    numeric_color_spec,
                    min_value,
                    max_value,
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
            Patch(color=color, label=_legend_entry_label(level, options))
            for level, color in column_palette.items()
        ]
        if values.isna().any():
            handles.append(Patch(color="#D9D9D9", label=_truncate_text(options.metadata_na_marker, options.legend_label_max_chars)))
        legends.append(_MetadataLegendSpec(title=display_column, key=f"metadata:{column}", handles=handles))

    ax.set_yticks(np.arange(len(prepared.metadata_cols)))
    ax.set_yticklabels(
        [_legend_title(column, options) for column in prepared.metadata_cols],
        fontsize=options.font_size_metadata,
    )
    for label in ax.get_yticklabels():
        label.update(_font_kwargs(options.font_style_metadata))
    ax.set_xticks([])
    return legends


def _add_numeric_metadata_colorbar(
    figure,
    legend: _MetadataLegendSpec,
    bounds: Sequence[float],
    orientation: str,
    options: OncoplotOptions,
    fontsize: float,
    title_fontsize: float,
) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cmin, cmax, tick_values, tick_labels = _numeric_colorbar_bounds(legend.min_value, legend.max_value)
    cax = figure.add_axes(_offset_bounds(bounds, options, legend.key))
    mappable = ScalarMappable(norm=Normalize(vmin=cmin, vmax=cmax), cmap=legend.colormap)
    mappable.set_array([])
    colorbar = figure.colorbar(mappable, cax=cax, orientation=orientation, ticks=tick_values)
    if orientation == "horizontal":
        colorbar.ax.set_xticklabels(tick_labels)
    else:
        colorbar.ax.set_yticklabels(tick_labels)
    colorbar.ax.tick_params(labelsize=fontsize)
    if options.show_legend_titles:
        if orientation == "horizontal":
            colorbar.set_label(_truncate_text(legend.title, options.legend_title_max_chars), fontsize=title_fontsize)
        else:
            colorbar.ax.set_title(
                _truncate_text(legend.title, options.legend_title_max_chars),
                fontsize=title_fontsize,
                pad=max(2, title_fontsize * 0.35),
            )


def _add_bottom_numeric_metadata_colorbars(
    figure,
    numeric_legends: Sequence[_MetadataLegendSpec],
    options: OncoplotOptions,
    fontsize: float,
    title_fontsize: float,
    *,
    has_lower_bottom_legend: bool,
) -> None:
    if not numeric_legends:
        return
    total = len(numeric_legends)
    slot_width = min(0.28, 0.86 / max(total, 1))
    colorbar_width = slot_width * 0.68
    thickness = max(0.012, min(0.035, 0.015 * max(options.metadata_legend_key_size, 0.5)))
    y = 0.105 if has_lower_bottom_legend else 0.055
    start_x = 0.08 if total > 1 else 0.36
    for index, legend in enumerate(numeric_legends):
        x = start_x + index * slot_width
        _add_numeric_metadata_colorbar(
            figure,
            legend,
            [x, y, colorbar_width, thickness],
            "horizontal",
            options,
            fontsize,
            title_fontsize,
        )


def _add_static_legends(
    figure,
    mutation_handles,
    tmb_handles,
    metadata_legends,
    options: OncoplotOptions,
    *,
    mutation_legend_key: str,
    tmb_legend_key: str,
) -> None:
    large_figure = options.width >= 1600 or options.height >= 1200
    mutation_base = 10 if large_figure else 8
    metadata_base = 9 if large_figure else 7
    mutation_scale = 1.2 if large_figure else 0.75
    metadata_scale = 1.2 if large_figure else 0.85
    title_padding = 4 if large_figure else 1
    figure_height = figure.get_figheight()
    mutation_fontsize = options.font_size_legend_text or _legend_font_size(
        mutation_base,
        options.font_size_genes,
        scale=mutation_scale,
    )
    mutation_title_fontsize = options.font_size_legend_title or mutation_fontsize + title_padding
    metadata_fontsize = options.font_size_metadata_legend_text or options.font_size_legend_text or _legend_font_size(
        metadata_base,
        options.font_size_metadata,
        scale=metadata_scale,
    )
    metadata_title_fontsize = (
        options.font_size_metadata_legend_title
        or options.font_size_legend_title
        or metadata_fontsize + title_padding
    )
    if options.show_legend and mutation_handles and options.mutation_legend_position == "bottom":
        figure.legend(
            handles=mutation_handles,
            loc="lower center",
            bbox_to_anchor=_offset_anchor((0.42, 0.01), options, mutation_legend_key, "mutation"),
            ncol=min(5, len(mutation_handles)),
            frameon=True,
            fontsize=mutation_fontsize,
            title=_legend_entry_title("Mutation Type", options) if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )
    if options.show_legend and tmb_handles and options.mutation_legend_position == "bottom":
        figure.legend(
            handles=tmb_handles,
            loc="lower right",
            bbox_to_anchor=_offset_anchor((0.99, 0.01), options, tmb_legend_key, "tmb"),
            ncol=min(4, len(tmb_handles)),
            frameon=True,
            fontsize=mutation_fontsize,
            title=_legend_entry_title("TMB Type", options) if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )

    right_y = 0.96
    if options.show_legend and mutation_handles and options.mutation_legend_position == "right":
        figure.legend(
            handles=mutation_handles,
            loc="upper left",
            bbox_to_anchor=_offset_anchor((0.74, right_y), options, mutation_legend_key, "mutation"),
            ncol=1,
            frameon=False,
            fontsize=mutation_fontsize,
            title=_legend_entry_title("Mutation Type", options) if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )
        right_y -= _legend_stack_step(len(mutation_handles), mutation_fontsize, figure_height=figure_height)
    if options.show_legend and tmb_handles and options.mutation_legend_position == "right":
        figure.legend(
            handles=tmb_handles,
            loc="upper left",
            bbox_to_anchor=_offset_anchor((0.74, right_y), options, tmb_legend_key, "tmb"),
            ncol=1,
            frameon=False,
            fontsize=mutation_fontsize,
            title=_legend_entry_title("TMB Type", options) if options.show_legend_titles else None,
            title_fontsize=mutation_title_fontsize,
            handlelength=options.legend_key_size,
            handleheight=options.legend_key_size,
        )
        right_y -= _legend_stack_step(len(tmb_handles), mutation_fontsize, figure_height=figure_height)

    if not options.show_metadata_legends or not metadata_legends:
        return

    metadata_legends = list(metadata_legends)
    categorical_bottom_legends = [
        legend
        for legend in metadata_legends
        if not legend.is_numeric and options.metadata_legend_position == "bottom"
    ]
    numeric_bottom_legends = [
        legend
        for legend in metadata_legends
        if legend.is_numeric and _metadata_colorbar_is_horizontal(options)
    ]
    right_legends = [
        legend
        for legend in metadata_legends
        if options.metadata_legend_position == "right"
        and (not legend.is_numeric or not _metadata_colorbar_is_horizontal(options))
    ]
    has_bottom_mutation_legend = bool(
        options.show_legend
        and (mutation_handles or tmb_handles)
        and options.mutation_legend_position == "bottom"
    )

    if categorical_bottom_legends:
        offset_legends = [legend for legend in categorical_bottom_legends if legend.key in options.legend_offsets]
        default_legends = [legend for legend in categorical_bottom_legends if legend.key not in options.legend_offsets]
        if offset_legends:
            handles = []
            for legend in default_legends:
                handles.extend((legend.handles or [])[: max(1, min(3, len(legend.handles or [])))])
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
            for index, legend in enumerate(offset_legends):
                handles = legend.handles or []
                if not handles:
                    continue
                ncol = options.metadata_legend_ncol or 3
                anchor = _offset_anchor((0.99, 0.01 + index * 0.055), options, legend.key)
                figure.legend(
                    handles=handles,
                    loc="lower right",
                    bbox_to_anchor=anchor,
                    ncol=ncol,
                    frameon=True,
                    fontsize=metadata_fontsize,
                    title=_truncate_text(legend.title, options.legend_title_max_chars) if options.show_legend_titles else None,
                    title_fontsize=metadata_title_fontsize,
                    handlelength=options.metadata_legend_key_size,
                    handleheight=options.metadata_legend_key_size,
                )
        else:
            handles = []
            for legend in categorical_bottom_legends:
                handles.extend((legend.handles or [])[: max(1, min(3, len(legend.handles or [])))])
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
    _add_bottom_numeric_metadata_colorbars(
        figure,
        numeric_bottom_legends,
        options,
        metadata_fontsize,
        metadata_title_fontsize,
        has_lower_bottom_legend=has_bottom_mutation_legend or bool(categorical_bottom_legends),
    )

    if not right_legends:
        return

    y = right_y
    for legend in right_legends:
        if legend.is_numeric:
            height = max(0.10, min(0.18, 0.13 * max(options.metadata_legend_key_size, 0.5)))
            width = max(0.012, min(0.035, 0.016 * max(options.metadata_legend_key_size, 0.5)))
            _add_numeric_metadata_colorbar(
                figure,
                legend,
                [_centered_right_legend_x(width), max(0.05, y - height), width, height],
                "vertical",
                options,
                metadata_fontsize,
                metadata_title_fontsize,
            )
            y -= height + 0.065
            if y < 0.05:
                break
            continue
        handles = legend.handles or []
        if not handles:
            continue
        ncol = options.metadata_legend_ncol or 1
        if options.metadata_legend_nrow:
            ncol = max(ncol, int(np.ceil(len(handles) / options.metadata_legend_nrow)))
        figure.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=_offset_anchor((0.74, y), options, legend.key),
            ncol=ncol,
            frameon=False,
            fontsize=metadata_fontsize,
            title=_truncate_text(legend.title, options.legend_title_max_chars) if options.show_legend_titles else None,
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
    main_row_count = (
        len(prepared.main_grid_rows)
        if _has_expanded_main_grid(prepared) and prepared.main_grid_rows is not None
        else len(prepared.genes)
    )
    rows.append(("main", max(3, main_row_count * 0.55)))
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
    mutation_legend_visible = _main_grid_has_mutation_rows(prepared) or draw_gene_bar
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
    if options.main_subplot_title is not None:
        axes["main"].set_title(options.main_subplot_title, fontsize=options.font_size_subplot_title)
    if options.tmb_subplot_title is not None and "tmb" in axes:
        axes["tmb"].set_title(options.tmb_subplot_title, fontsize=options.font_size_subplot_title)
    if options.metadata_subplot_title is not None and "metadata" in axes:
        axes["metadata"].set_title(options.metadata_subplot_title, fontsize=options.font_size_subplot_title)
    if options.gene_bar_subplot_title is not None and "gene_bar" in axes:
        axes["gene_bar"].set_title(options.gene_bar_subplot_title, fontsize=options.font_size_subplot_title)
    if options.title_text is not None:
        figure.suptitle(options.title_text, fontsize=options.font_size_title)

    mutation_handles = []
    if (
        options.show_legend
        and options.mutation_legend_position != "none"
        and (_main_grid_has_mutation_rows(prepared) or draw_gene_bar)
    ):
        mutation_handles = [
            Patch(color=palette.get(name, options.unspecified_mutation_color), label=_legend_entry_label(name, options))
            for name in _mutation_levels(prepared, palette)
        ]
    has_bottom_mutation_legend = bool((mutation_handles or tmb_handles) and options.mutation_legend_position == "bottom")
    has_bottom_metadata_legend = _metadata_has_bottom_legends(metadata_legends, options)
    if has_bottom_mutation_legend and has_bottom_metadata_legend:
        bottom_margin = 0.26
    elif has_bottom_metadata_legend:
        bottom_margin = 0.20
    elif has_bottom_mutation_legend:
        bottom_margin = 0.18
    else:
        bottom_margin = 0.08
    has_right_legend = bool(
        ((mutation_handles or tmb_handles) and options.mutation_legend_position == "right")
        or _metadata_has_right_legends(metadata_legends, options)
    )
    variant_colorbar_active = (
        _main_grid_has_continuous_rows(prepared)
        and options.show_legend
        and options.mutation_legend_position != "none"
    )
    if variant_colorbar_active:
        bottom_margin = max(bottom_margin, 0.14)
    mutation_legend_key = f"mutation:{prepared.mutation_type_col}" if prepared.mutation_type_col is not None else "mutation"
    tmb_legend_key = f"tmb:{prepared.tmb_type_col}" if prepared.tmb_type_col is not None else "tmb"
    _add_static_legends(
        figure,
        mutation_handles,
        tmb_handles,
        metadata_legends,
        options,
        mutation_legend_key=mutation_legend_key,
        tmb_legend_key=tmb_legend_key,
    )
    figure.subplots_adjust(
        left=_left_margin_for_metadata(prepared, options, draw_metadata=draw_metadata, fig_width=fig_width),
        right=0.70 if has_right_legend else 0.98,
        top=0.86 if options.title_text is not None else 0.90 if any(
            title is not None
            for title in (
                options.main_subplot_title,
                options.tmb_subplot_title,
                options.gene_bar_subplot_title,
                options.metadata_subplot_title,
            )
        ) else 0.92,
        bottom=bottom_margin,
        hspace=max(0.02, min(0.6, max(options.buffer_tmb, options.buffer_metadata))),
        wspace=max(0.01, min(0.4, options.buffer_gene_bar)),
    )
    if draw_gene_bar and "gene_bar" in axes:
        main_position = axes["main"].get_position()
        gene_bar_position = axes["gene_bar"].get_position()
        axes["gene_bar"].set_position(
            [gene_bar_position.x0, main_position.y0, gene_bar_position.width, main_position.height]
        )
    return figure
