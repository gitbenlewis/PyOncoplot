"""Plotly renderer for interactive oncoplots."""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._data import PreparedOncoplotData
from ._metadata_colors import (
    categorical_metadata_palette,
    continuous_colorscale,
    metadata_palette_spec,
    numeric_metadata_colorscale,
    numeric_metadata_colormap_spec,
    sample_numeric_metadata_colormap,
)
from ._options import OncoplotOptions, coerce_options
from ._params import merge_params
from ._utils import as_percent, reorder_by_priority


PLOTLY_RENDER_PARAM_KEYS = {
    "prepared",
    "palette",
    "tmb_palette",
    "metadata_palette",
    "variant_value_palette",
    "options",
    "draw_gene_bar",
    "draw_tmb_bar",
    "copy_on_click",
}

PLOTLY_RENDER_DEFAULTS: dict[str, Any] = {
    "tmb_palette": None,
    "metadata_palette": None,
    "variant_value_palette": "viridis",
    "draw_gene_bar": False,
    "draw_tmb_bar": False,
    "copy_on_click": "sample",
}


def _require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("Plotly rendering requires the 'plotly' package.") from exc
    return go, make_subplots


def _legend_label(value: object, options: OncoplotOptions) -> str:
    text = "Unspecified" if pd.isna(value) else str(value)
    return options.prettify_function(text) if options.prettify_legend_values else text


def _legend_title(value: object, options: OncoplotOptions) -> str:
    text = "" if value is None else str(value)
    return options.prettify_function(text) if options.prettify_legend_titles else text


def _metadata_value_label(value: object, options: OncoplotOptions) -> str:
    if pd.isna(value):
        return options.metadata_na_marker
    text = str(value)
    return options.prettify_function(text) if options.prettify_legend_values else text


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


def _copy_value(row: pd.Series, copy_on_click: str) -> str:
    if copy_on_click == "sample":
        return str(row["Sample"])
    if copy_on_click == "gene":
        return str(row["Gene"])
    if copy_on_click == "tooltip":
        return str(row["Tooltip"])
    if copy_on_click == "mutation_type":
        return "" if pd.isna(row["MutationType"]) else str(row["MutationType"])
    return ""


def _custom_payload(**values: object) -> Dict[str, object]:
    return dict(values)


def _font_options(size: float, options: OncoplotOptions) -> Dict[str, object]:
    return {"size": size, "family": options.font_family}


def _truncate_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


def _legend_entry_label(value: object, options: OncoplotOptions) -> str:
    return _truncate_text(_legend_label(value, options), options.legend_label_max_chars)


def _legend_entry_title(value: object, options: OncoplotOptions) -> str:
    return _truncate_text(_legend_title(value, options), options.legend_title_max_chars)


def _legend_offset_key(options: OncoplotOptions, key: str, fallback_key: Optional[str] = None) -> Optional[str]:
    if key in options.legend_offsets:
        return key
    if fallback_key is not None and fallback_key in options.legend_offsets:
        return fallback_key
    return None


def _legend_offset(options: OncoplotOptions, key: str, fallback_key: Optional[str] = None) -> tuple[float, float]:
    offset_key = _legend_offset_key(options, key, fallback_key)
    if offset_key is None:
        return 0.0, 0.0
    offset = options.legend_offsets[offset_key]
    return float(offset.get("x", 0.0)), float(offset.get("y", 0.0))


def _offset_layout_value(
    value: float,
    options: OncoplotOptions,
    key: str,
    axis: str,
    fallback_key: Optional[str] = None,
) -> float:
    x_offset, y_offset = _legend_offset(options, key, fallback_key)
    return value + (x_offset if axis == "x" else y_offset)


def _category_range(length: int) -> list[float]:
    return [-0.5, max(length - 0.5, 0.5)]


def _sample_axis_options(prepared: PreparedOncoplotData) -> Dict[str, object]:
    return {
        "categoryorder": "array",
        "categoryarray": list(prepared.samples),
        "range": _category_range(len(prepared.samples)),
    }


def _gene_axis_options(prepared: PreparedOncoplotData) -> Dict[str, object]:
    return {
        "categoryorder": "array",
        "categoryarray": list(reversed(prepared.genes)),
        "range": _category_range(len(prepared.genes)),
        "tickmode": "array",
        "tickvals": list(prepared.genes),
        "ticktext": list(prepared.genes),
        "automargin": True,
    }


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
        return {gene: float(index) for index, gene in enumerate(prepared.genes)}
    centers = {}
    for gene, group in rows.groupby("Gene", sort=False, observed=False):
        centers[str(gene)] = (float(group["RowIndex"].min()) + float(group["RowIndex"].max())) / 2
    return centers


def _expanded_main_axis_options(prepared: PreparedOncoplotData) -> Dict[str, object]:
    rows = prepared.main_grid_rows
    n_rows = 0 if rows is None else len(rows)
    centers = _main_grid_gene_centers(prepared)
    return {
        "range": [max(n_rows - 0.5, 0.5), -0.5],
        "tickmode": "array",
        "tickvals": [centers[gene] for gene in prepared.genes if gene in centers],
        "ticktext": ["" for gene in prepared.genes if gene in centers],
        "automargin": True,
    }


def _metadata_axis_options(prepared: PreparedOncoplotData, options: OncoplotOptions) -> Dict[str, object]:
    labels = [_legend_title(column, options) for column in prepared.metadata_cols or []]
    return {
        "categoryorder": "array",
        "categoryarray": labels,
        "tickmode": "array",
        "tickvals": labels,
        "ticktext": labels,
        "automargin": True,
    }


def _validate_metadata_levels(prepared: PreparedOncoplotData, options: OncoplotOptions) -> None:
    for track in prepared.metadata_tracks or []:
        if track.kind == "categorical" and len(track.levels) > options.metadata_max_levels:
            raise ValueError(
                f"Metadata column {track.column!r} has {len(track.levels)} levels, "
                f"which exceeds metadata_max_levels={options.metadata_max_levels}."
            )


def _grid_positions(prepared: PreparedOncoplotData, draw_tmb_bar: bool, draw_metadata: bool, options: OncoplotOptions):
    rows = []
    if options.metadata_position == "top" and draw_metadata:
        rows.append(("metadata", options.metadata_height_ratio))
    if draw_tmb_bar:
        rows.append(("tmb", options.tmb_height_ratio))
    rows.append(("main", 1.0))
    if options.metadata_position == "bottom" and draw_metadata:
        rows.append(("metadata", options.metadata_height_ratio))

    fixed = sum(weight for name, weight in rows if name != "main")
    main_weight = max(0.2, 1.0 - fixed)
    row_heights = [main_weight if name == "main" else weight for name, weight in rows]
    total = sum(row_heights)
    row_heights = [height / total for height in row_heights]
    row_by_name = {name: index + 1 for index, (name, _weight) in enumerate(rows)}
    return rows, row_heights, row_by_name


def _trace_customdata_role(trace) -> Optional[str]:
    customdata = getattr(trace, "customdata", None)
    if customdata is None or len(customdata) == 0:
        return None
    first = customdata[0]
    if isinstance(first, dict):
        return first.get("role")
    if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], dict):
        return first[0].get("role")
    return None


def _plotly_legend_trace_key(trace, mutation_key: str, tmb_key: str) -> tuple[Optional[str], Optional[str]]:
    if getattr(trace, "showlegend", None) is not True:
        return None, None
    legendgroup = str(getattr(trace, "legendgroup", "") or "")
    if legendgroup == "tmb":
        return tmb_key, "tmb"
    if legendgroup.startswith("gene_bar:"):
        return "gene_bar", None
    if legendgroup.startswith("metadata:"):
        return legendgroup, None
    if _trace_customdata_role(trace) == "main_tile":
        return mutation_key, "mutation"
    return None, None


def _plotly_offset_legend_layout(
    base_legend: Mapping[str, object],
    options: OncoplotOptions,
    key: str,
    fallback_key: Optional[str],
) -> dict[str, object]:
    layout = dict(base_legend)
    x_offset, y_offset = _legend_offset(options, key, fallback_key)
    layout["x"] = float(layout.get("x", 1.02)) + x_offset
    layout["y"] = float(layout.get("y", 1.0)) + y_offset
    return layout


def _assign_plotly_offset_legends(
    figure,
    options: OncoplotOptions,
    base_legend: Mapping[str, object],
    *,
    mutation_key: str,
    tmb_key: str,
) -> dict[str, object]:
    offset_layouts: dict[str, object] = {}
    offset_key_to_legend_name: dict[str, str] = {}
    for trace in figure.data:
        key, fallback_key = _plotly_legend_trace_key(trace, mutation_key, tmb_key)
        if key is None:
            continue
        offset_key = _legend_offset_key(options, key, fallback_key)
        if offset_key is None:
            continue
        legend_name = offset_key_to_legend_name.get(offset_key)
        if legend_name is None:
            legend_name = f"legend{len(offset_key_to_legend_name) + 2}"
            offset_key_to_legend_name[offset_key] = legend_name
            offset_layouts[legend_name] = _plotly_offset_legend_layout(
                base_legend,
                options,
                key,
                fallback_key,
            )
        trace.legend = legend_name
    return offset_layouts


def _add_tmb_bar(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    palette: Mapping[str, str],
    tmb_palette: Optional[Mapping[str, str]],
    options: OncoplotOptions,
    mutation_legend_visible: bool = True,
):
    go, _make_subplots = _require_plotly()
    tmb = prepared.tmb
    if (
        tmb is None
        or prepared.tmb_sample_col is None
        or prepared.tmb_value_col is None
        or prepared.tmb_type_col is None
    ):
        return

    sample_col = prepared.tmb_sample_col
    value_col = prepared.tmb_value_col
    type_col = prepared.tmb_type_col
    render_stacked = prepared.tmb_render_stacked and not options.log10_transform_tmb
    show_tmb_legend = _should_show_tmb_legend(
        prepared,
        options,
        render_stacked,
        palette,
        tmb_palette,
        mutation_legend_visible=mutation_legend_visible,
    )
    if prepared.tmb_render_stacked and options.log10_transform_tmb:
        warnings.warn(
            "log10_transform_tmb=True disables stacked TMB rendering; totals are rendered instead.",
            stacklevel=2,
        )

    if render_stacked:
        tmb_color_palette = tmb_palette or palette
        for tmb_type in _tmb_levels(prepared, tmb_color_palette):
            group = tmb[tmb[type_col].astype(str) == tmb_type]
            color = tmb_color_palette.get(str(tmb_type), options.unspecified_mutation_color)
            sample_values = list(prepared.samples)
            values_by_sample = (
                group.groupby(sample_col, observed=False)[value_col]
                .sum()
                .reindex(sample_values, fill_value=0)
            )
            values = values_by_sample.astype(float).tolist()
            tmb_label = _legend_label(tmb_type, options)
            tmb_legend_label = _legend_entry_label(tmb_type, options)
            fig.add_trace(
                go.Bar(
                    x=sample_values,
                    y=values,
                    width=1,
                    name=f"TMB: {tmb_legend_label}",
                    legendgroup="tmb",
                    marker_color=color,
                    customdata=[
                        _custom_payload(
                            role="tmb",
                            sample=sample,
                            tmb=value,
                            tmb_type=None if pd.isna(tmb_type) else str(tmb_type),
                        )
                        for sample, value in zip(sample_values, values)
                    ],
                    hovertemplate="Sample: %{x}<br>TMB: %{y}<br>Type: "
                    + tmb_label
                    + "<extra></extra>",
                    showlegend=show_tmb_legend,
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.18)),
                ),
                row=row,
                col=col,
            )
        fig.update_layout(barmode="stack")
    else:
        totals = tmb.groupby(sample_col, observed=False)[value_col].sum().reindex(prepared.samples, fill_value=0)
        y_values = totals.astype(float)
        axis_title = value_col
        if options.log10_transform_tmb:
            y_values = np.log10(np.maximum(y_values, 1))
            axis_title = f"log10 {value_col}"
        customdata = [
            _custom_payload(role="tmb", sample=sample, tmb=float(total), rendered_tmb=float(rendered))
            for sample, total, rendered in zip(prepared.samples, totals.astype(float), y_values.astype(float))
        ]
        fig.add_trace(
            go.Bar(
                x=prepared.samples,
                y=y_values,
                width=1,
                marker_color="#4D4D4D",
                customdata=customdata,
                hovertemplate="Sample: %{x}<br>TMB: %{customdata.tmb}<extra></extra>",
                showlegend=False,
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.18)),
            ),
            row=row,
            col=col,
        )
        if options.show_tmb_y_label:
            fig.update_yaxes(
                title_text=axis_title,
                title_font=_font_options(options.font_size_tmb_axis, options),
                row=row,
                col=col,
            )

    fig.update_yaxes(tickfont=_font_options(options.font_size_tmb_axis, options), row=row, col=col)
    if options.scientific_tmb:
        fig.update_yaxes(tickformat=".1e", row=row, col=col)
    if not options.show_tmb_axis:
        fig.update_yaxes(showticklabels=False, showline=False, ticks="", row=row, col=col)
    fig.update_xaxes(**_sample_axis_options(prepared), showticklabels=False, ticks="", row=row, col=col)


def _metadata_color_map(
    levels: Sequence[object],
    options: OncoplotOptions,
    supplied: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    level_names = [str(value) for value in levels]
    mapping = {
        level: options.metadata_default_colors[index % len(options.metadata_default_colors)]
        for index, level in enumerate(level_names)
    }
    if supplied:
        for level, color in supplied.items():
            if str(level) in mapping:
                mapping[str(level)] = color
    return mapping


def _numeric_metadata_color(value: object, min_value: float, max_value: float, options: OncoplotOptions) -> str:
    if pd.isna(value):
        return "#D9D9D9"
    span = max(max_value - min_value, 1e-9)
    bucket = min(
        len(options.metadata_default_colors) - 1,
        int((float(value) - min_value) / span * len(options.metadata_default_colors)),
    )
    return options.metadata_default_colors[bucket]


def _metadata_numeric_color(
    value: object,
    min_value: float,
    max_value: float,
    column: object,
    options: OncoplotOptions,
    colormap_spec: object = None,
) -> str:
    if pd.isna(value):
        return "#D9D9D9"
    if colormap_spec is not None:
        return sample_numeric_metadata_colormap(colormap_spec, value, min_value, max_value, column)
    return _numeric_metadata_color(value, min_value, max_value, options)


def _metadata_colorbar_is_horizontal(options: OncoplotOptions) -> bool:
    return (
        options.metadata_legend_position == "bottom"
        or options.metadata_legend_orientation_heatmap == "horizontal"
    )


def _metadata_colorbar_layout(
    index: int,
    total: int,
    key: str,
    options: OncoplotOptions,
) -> Dict[str, object]:
    thickness = max(8, int(options.metadata_legend_key_size * 12))
    if _metadata_colorbar_is_horizontal(options):
        slot_width = min(0.92, 0.92 / max(total, 1))
        return {
            "orientation": "h",
            "x": _offset_layout_value(0.04 + (index + 0.5) * slot_width, options, key, "x"),
            "xanchor": "center",
            "y": _offset_layout_value(-0.32, options, key, "y"),
            "yanchor": "top",
            "len": max(0.12, min(0.28, slot_width * 0.72)),
            "thickness": thickness,
            "outlinewidth": 0,
        }

    slot_height = min(0.22, 0.86 / max(total, 1))
    return {
        "x": _offset_layout_value(1.08, options, key, "x"),
        "xanchor": "left",
        "y": _offset_layout_value(1.0 - index * slot_height, options, key, "y"),
        "yanchor": "top",
        "len": max(0.10, min(0.20, slot_height * 0.72)),
        "thickness": thickness,
        "outlinewidth": 0,
    }


def _numeric_colorbar_bounds(min_value: float, max_value: float) -> tuple[float, float, list[float], list[str]]:
    if math.isclose(min_value, max_value):
        padding = max(0.5, abs(min_value) * 0.05)
        return (
            min_value - padding,
            max_value + padding,
            [min_value],
            [f"{min_value:g}"],
        )
    return min_value, max_value, [min_value, max_value], [f"{min_value:g}", f"{max_value:g}"]


def _add_numeric_metadata_colorbars(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    colorbars: Sequence[tuple[str, str, float, float, list[list[object]]]],
    options: OncoplotOptions,
) -> None:
    if not colorbars:
        return
    go, _make_subplots = _require_plotly()
    x_values = [prepared.samples[0], prepared.samples[-1]] if prepared.samples else [None, None]
    metadata_labels = [_legend_title(column, options) for column in prepared.metadata_cols or []]
    y_value = metadata_labels[0] if metadata_labels else None

    fontsize = options.font_size_metadata_legend_text or options.font_size_legend_text or options.font_size_metadata
    title_fontsize = options.font_size_metadata_legend_title or options.font_size_legend_title or fontsize
    for index, (key, title, min_value, max_value, colorscale) in enumerate(colorbars):
        cmin, cmax, tickvals, ticktext = _numeric_colorbar_bounds(min_value, max_value)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[y_value, y_value],
                mode="markers",
                marker=dict(
                    color=[cmin, cmax],
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                    showscale=True,
                    size=0.1,
                    opacity=0,
                    colorbar=dict(
                        title=dict(text=_truncate_text(title, options.legend_title_max_chars), font=_font_options(title_fontsize, options)),
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickfont=_font_options(fontsize, options),
                        **_metadata_colorbar_layout(index, len(colorbars), key, options),
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _variant_value_hover_text(row: pd.Series, title: str) -> str:
    value = f"{float(row['VariantValue']):g}"
    tooltip = str(row["Tooltip"])
    lines = set(tooltip.split("<br>"))
    title_line = f"{title}: {value}"
    raw_label = row.get("Label", title)
    raw_line = "" if pd.isna(raw_label) else f"{raw_label}: {value}"
    if title_line in lines or raw_line in lines:
        return tooltip
    return f"{tooltip}<br>{title_line}"


def _variant_colorbar_layout(index: int, total: int, key: str, options: OncoplotOptions) -> Dict[str, object]:
    if total <= 1:
        x_offset, y_offset = _legend_offset(options, key, "variant")
        if not x_offset and not y_offset:
            return {}
        return {"x": 1.02 + x_offset, "xanchor": "left", "y": 0.5 + y_offset}
    slot_height = min(0.24, 0.86 / max(total, 1))
    return {
        "x": _offset_layout_value(1.02 + min(index, 2) * 0.07, options, key, "x", "variant"),
        "xanchor": "left",
        "y": _offset_layout_value(1.0 - index * slot_height, options, key, "y", "variant"),
        "yanchor": "top",
        "len": max(0.14, min(0.24, slot_height * 0.72)),
        "thickness": 12,
        "outlinewidth": 0,
    }


def _is_missing_palette_spec(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _add_continuous_main_tiles(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    variant_value_palette: object,
    options: OncoplotOptions,
    copy_on_click: str,
) -> None:
    go, _make_subplots = _require_plotly()
    sample_index = {sample: index for index, sample in enumerate(prepared.samples)}
    gene_index = {gene: index for index, gene in enumerate(prepared.genes)}
    z = np.full((len(prepared.genes), len(prepared.samples)), np.nan)
    text = [["" for _sample in prepared.samples] for _gene in prepared.genes]
    customdata = [
        [_custom_payload(role="main_tile", sample=str(sample), gene=str(gene)) for sample in prepared.samples]
        for gene in prepared.genes
    ]
    title = _legend_title(prepared.variant_value_col, options)

    for _index, row_value in prepared.tiles.iterrows():
        sample = str(row_value["Sample"])
        gene = str(row_value["Gene"])
        if sample not in sample_index or gene not in gene_index:
            continue
        if pd.isna(row_value["VariantValue"]):
            continue
        y = gene_index[gene]
        x = sample_index[sample]
        value = float(row_value["VariantValue"])
        z[y, x] = value
        text[y][x] = _variant_value_hover_text(row_value, title)
        customdata[y][x] = _custom_payload(
            role="main_tile",
            sample=sample,
            gene=gene,
            mutation_type="" if pd.isna(row_value["MutationType"]) else str(row_value["MutationType"]),
            variant_value=value,
            copy_value=_copy_value(row_value, copy_on_click),
        )

    has_values = prepared.variant_value_min is not None and prepared.variant_value_max is not None
    min_value = prepared.variant_value_min if has_values else 0.0
    max_value = prepared.variant_value_max if has_values else 1.0
    cmin, cmax, tickvals, ticktext = _numeric_colorbar_bounds(float(min_value), float(max_value))
    show_colorbar = has_values and options.show_legend and options.mutation_legend_position != "none"
    legend_key = f"variant:{prepared.variant_value_col}"
    legend_fontsize = options.font_size_legend_text or options.font_size_metadata
    legend_title_fontsize = options.font_size_legend_title or legend_fontsize
    fig.add_trace(
        go.Heatmap(
            x=prepared.samples,
            y=prepared.genes,
            z=z,
            text=text,
            customdata=customdata,
            hovertemplate="%{text}<extra></extra>",
            hoverongaps=False,
            colorscale=continuous_colorscale(variant_value_palette, prepared.variant_value_col),
            zmin=cmin,
            zmax=cmax,
            showscale=show_colorbar,
            colorbar=dict(
                title=dict(text=_truncate_text(title, options.legend_title_max_chars), font=_font_options(legend_title_fontsize, options)),
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=_font_options(legend_fontsize, options),
                **_variant_colorbar_layout(0, 1, legend_key, options),
            ),
        ),
        row=row,
        col=col,
    )


def _add_expanded_row_labels(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    options: OncoplotOptions,
) -> None:
    rows = prepared.main_grid_rows
    if rows is None or rows.empty or rows["TrackId"].nunique() <= 1:
        return
    xref = f"x{'' if row == 1 and col == 1 else row}"
    yref = f"y{'' if row == 1 and col == 1 else row}"
    first_sample = prepared.samples[0] if prepared.samples else 0
    track_rows = rows.drop_duplicates("RowId").sort_values("RowIndex", kind="mergesort")
    for _index, row_spec in track_rows.iterrows():
        fig.add_annotation(
            xref=xref,
            yref=yref,
            x=first_sample,
            y=float(row_spec["RowIndex"]),
            text=_legend_title(row_spec["Label"], options),
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            xshift=-(8 + options.main_grid_rows_label_x_offset),
            font=dict(size=max(7, options.font_size_genes * 0.72), color="#555555"),
        )


def _add_expanded_gene_labels(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    options: OncoplotOptions,
) -> None:
    rows = prepared.main_grid_rows
    if rows is None or rows.empty:
        return
    xref = f"x{'' if row == 1 and col == 1 else row}"
    yref = f"y{'' if row == 1 and col == 1 else row}"
    first_sample = prepared.samples[0] if prepared.samples else 0
    centers = _main_grid_gene_centers(prepared)
    for gene in prepared.genes:
        if gene not in centers:
            continue
        fig.add_annotation(
            xref=xref,
            yref=yref,
            x=first_sample,
            y=centers[gene],
            text=gene,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            xshift=-(54 + options.gene_name_x_offset),
            font=_font_options(options.font_size_genes, options),
        )


def _add_expanded_gene_separators(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    options: OncoplotOptions,
) -> None:
    rows = prepared.main_grid_rows
    if rows is None or rows.empty:
        return
    xref = f"x{'' if row == 1 and col == 1 else row}"
    yref = f"y{'' if row == 1 and col == 1 else row}"
    x0 = -0.5
    x1 = max(len(prepared.samples) - 0.5, 0.5)
    for _gene, group in rows.groupby("Gene", sort=False, observed=False):
        for boundary in (float(group["RowIndex"].min()) - 0.5, float(group["RowIndex"].max()) + 0.5):
            fig.add_shape(
                type="line",
                xref=xref,
                yref=yref,
                x0=x0,
                x1=x1,
                y0=boundary,
                y1=boundary,
                line=dict(color="#666666", width=options.row_separator_linewidth),
                opacity=0.65,
            )


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


def _add_expanded_continuous_tiles(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    variant_value_palette: object,
    options: OncoplotOptions,
    copy_on_click: str,
) -> None:
    go, _make_subplots = _require_plotly()
    rows = prepared.main_grid_rows
    tiles = prepared.main_grid_tiles
    if rows is None or rows.empty or tiles is None or tiles.empty:
        return

    sample_index = {sample: index for index, sample in enumerate(prepared.samples)}
    n_rows = len(rows)
    show_colorbar = options.show_legend and options.mutation_legend_position != "none"
    continuous_groups = _continuous_track_groups(prepared)
    continuous_tiles = tiles[tiles["Kind"] == "variant_value"]

    for colorbar_index, (_group_id, track_rows) in enumerate(continuous_groups):
        track_ids = set(track_rows["TrackId"].astype(str))
        group_tiles = continuous_tiles[continuous_tiles["TrackId"].astype(str).isin(track_ids)]
        non_missing_tiles = group_tiles[group_tiles["VariantValue"].notna()]
        if non_missing_tiles.empty:
            continue
        z = np.full((n_rows, len(prepared.samples)), np.nan)
        text = [["" for _sample in prepared.samples] for _row in range(n_rows)]
        customdata = [
            [
                _custom_payload(role="main_tile", sample=str(sample), row_index=row_index)
                for sample in prepared.samples
            ]
            for row_index in range(n_rows)
        ]
        min_value = float(non_missing_tiles["VariantValueMin"].dropna().iloc[0])
        max_value = float(non_missing_tiles["VariantValueMax"].dropna().iloc[0])
        cmin, cmax, tickvals, ticktext = _numeric_colorbar_bounds(min_value, max_value)
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

        for _index, row_value in group_tiles.iterrows():
            sample = str(row_value["Sample"])
            if sample not in sample_index:
                continue
            if pd.isna(row_value["VariantValue"]):
                continue
            y = int(row_value["RowIndex"])
            x = sample_index[sample]
            value = float(row_value["VariantValue"])
            label = _legend_title(row_value["Label"], options)
            z[y, x] = value
            text[y][x] = _variant_value_hover_text(row_value, label)
            customdata[y][x] = _custom_payload(
                role="main_tile",
                sample=sample,
                gene=str(row_value["Gene"]),
                row_id=str(row_value["RowId"]),
                row_label=str(row_value["Label"]),
                mutation_type="" if pd.isna(row_value["MutationType"]) else str(row_value["MutationType"]),
                variant_value=value,
                variant_value_column=str(row_value["VariantValueColumn"]),
                copy_value=_copy_value(row_value, copy_on_click),
            )

        legend_fontsize = options.font_size_legend_text or options.font_size_metadata
        legend_title_fontsize = options.font_size_legend_title or legend_fontsize
        fig.add_trace(
            go.Heatmap(
                x=prepared.samples,
                y=list(range(n_rows)),
                z=z,
                text=text,
                customdata=customdata,
                hovertemplate="%{text}<extra></extra>",
                hoverongaps=False,
                colorscale=continuous_colorscale(palette_spec, first_track["VariantValueColumn"]),
                zmin=cmin,
                zmax=cmax,
                showscale=show_colorbar,
                colorbar=dict(
                    title=dict(text=_truncate_text(title, options.legend_title_max_chars), font=_font_options(legend_title_fontsize, options)),
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickfont=_font_options(legend_fontsize, options),
                    **_variant_colorbar_layout(colorbar_index, len(continuous_groups), legend_key, options),
                ),
            ),
            row=row,
            col=col,
        )


def _add_expanded_main_tiles(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    palette: Mapping[str, str],
    variant_value_palette: object,
    options: OncoplotOptions,
    copy_on_click: str,
) -> None:
    go, _make_subplots = _require_plotly()
    rows = prepared.main_grid_rows
    if rows is None:
        return
    n_rows = len(rows)
    show_mutation_legend = options.show_legend and options.mutation_legend_position != "none"
    fig.add_trace(
        go.Heatmap(
            x=prepared.samples,
            y=list(range(n_rows)),
            z=np.zeros((n_rows, len(prepared.samples))),
            colorscale=[[0, options.background_color], [1, options.background_color]],
            showscale=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    _add_expanded_continuous_tiles(
        fig,
        prepared,
        row,
        col,
        variant_value_palette,
        options,
        copy_on_click,
    )

    tiles = prepared.main_grid_tiles
    mutation_tiles = (
        tiles[tiles["Kind"] == "mutation_type"]
        if tiles is not None and not tiles.empty
        else pd.DataFrame()
    )
    if not mutation_tiles.empty:
        mutation_values = _mutation_levels(prepared, palette)
        for mutation_type in mutation_values:
            group = mutation_tiles[mutation_tiles["MutationType"].astype(str) == mutation_type]
            color = palette.get(str(mutation_type), options.unspecified_mutation_color)
            customdata = [
                _custom_payload(
                    role="main_tile",
                    sample=str(row_value["Sample"]),
                    gene=str(row_value["Gene"]),
                    row_id=str(row_value["RowId"]),
                    row_label=str(row_value["Label"]),
                    mutation_type=str(row_value["MutationType"]),
                    copy_value=_copy_value(row_value, copy_on_click),
                )
                for _index, row_value in group.iterrows()
            ]
            fig.add_trace(
                go.Scatter(
                    x=group["Sample"].astype(str),
                    y=group["RowIndex"].astype(float),
                    mode="markers",
                    marker=dict(
                        symbol="square",
                        size=13,
                        color=color,
                        line=dict(color="white", width=options.tile_linewidth),
                    ),
                    name=_legend_entry_label(mutation_type, options),
                    text=group["Tooltip"].astype(str),
                    customdata=customdata,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=show_mutation_legend,
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.18)),
                ),
                row=row,
                col=col,
            )
        unspecified = mutation_tiles[mutation_tiles["MutationType"].isna()]
        if not unspecified.empty:
            fig.add_trace(
                go.Scatter(
                    x=unspecified["Sample"].astype(str),
                    y=unspecified["RowIndex"].astype(float),
                    mode="markers",
                    marker=dict(
                        symbol="square",
                        size=13,
                        color=options.unspecified_mutation_color,
                        line=dict(color="white", width=options.tile_linewidth),
                    ),
                    name=_truncate_text("Mutation", options.legend_label_max_chars),
                    text=unspecified["Tooltip"].astype(str),
                    customdata=[
                        _custom_payload(
                            role="main_tile",
                            sample=str(row_value["Sample"]),
                            gene=str(row_value["Gene"]),
                            row_id=str(row_value["RowId"]),
                            row_label=str(row_value["Label"]),
                            mutation_type="",
                            copy_value=_copy_value(row_value, copy_on_click),
                        )
                        for _index, row_value in unspecified.iterrows()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=show_mutation_legend and prepared.mutation_type_col is None,
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.18)),
                ),
                row=row,
                col=col,
            )

    fig.update_yaxes(**_expanded_main_axis_options(prepared), row=row, col=col)
    fig.update_yaxes(tickfont=_font_options(options.font_size_genes, options), row=row, col=col)
    fig.update_xaxes(
        **_sample_axis_options(prepared),
        tickfont=_font_options(options.font_size_samples, options),
        row=row,
        col=col,
    )
    if not options.show_sample_ids:
        fig.update_xaxes(showticklabels=False, ticks="", row=row, col=col)
    else:
        fig.update_xaxes(side=options.sample_id_position, tickangle=options.sample_id_angle, row=row, col=col)
    if options.show_x_label:
        fig.update_xaxes(
            title_text=options.x_label,
            title_font=_font_options(options.font_size_x_label, options),
            row=row,
            col=col,
        )
    if options.show_y_label:
        fig.update_yaxes(
            title_text=options.y_label,
            title_font=_font_options(options.font_size_y_label, options),
            row=row,
            col=col,
        )

    _add_expanded_gene_labels(fig, prepared, row, col, options)
    _add_expanded_row_labels(fig, prepared, row, col, options)
    _add_expanded_gene_separators(fig, prepared, row, col, options)
    if prepared.pathway_groups:
        xref = f"x{'' if row == 1 and col == 1 else row}"
        yref = f"y{'' if row == 1 and col == 1 else row}"
        for group in prepared.pathway_groups:
            group_rows = rows[rows["Gene"].astype(str).isin(group.genes)]
            if group_rows.empty:
                continue
            center = (float(group_rows["RowIndex"].min()) + float(group_rows["RowIndex"].max())) / 2
            fig.add_annotation(
                xref=xref,
                yref=yref,
                x=prepared.samples[0] if prepared.samples else 0,
                y=center,
                text=group.name,
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                textangle=options.pathway_text_angle,
                xshift=-86 if rows["TrackId"].nunique() > 1 else -16,
                font=dict(color=options.pathway_text_color, size=options.font_size_pathway or 10),
                bgcolor=options.pathway_background_color,
                bordercolor=options.pathway_outline_color,
                borderwidth=1,
            )


def _add_metadata_strip(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    options: OncoplotOptions,
    metadata_palette: Optional[Mapping[str, Any]] = None,
):
    go, _make_subplots = _require_plotly()
    metadata = prepared.metadata
    if metadata is None or not prepared.metadata_cols:
        return

    matrix = []
    text = []
    customdata = []
    colorscale = []
    color_index = 0
    color_lookup: Dict[object, int] = {}
    numeric_colorbars: list[tuple[str, str, float, float, list[list[object]]]] = []
    metadata_palette = metadata_palette or {}
    metadata_by_sample = metadata.set_index("Sample")

    for col_name in prepared.metadata_cols:
        display_col = _legend_title(col_name, options)
        row_values = []
        row_text = []
        row_custom = []
        values_by_sample = metadata_by_sample[col_name]
        is_numeric = pd.api.types.is_numeric_dtype(values_by_sample)
        if is_numeric and values_by_sample.notna().any():
            min_value = float(values_by_sample.min(skipna=True))
            max_value = float(values_by_sample.max(skipna=True))
        else:
            min_value = 0.0
            max_value = 1.0
        palette_spec = metadata_palette_spec(metadata_palette, col_name)
        numeric_colormap = numeric_metadata_colormap_spec(palette_spec) if is_numeric else None
        if is_numeric and options.show_metadata_legends and values_by_sample.notna().any():
            colorbar_spec = numeric_colormap if numeric_colormap is not None else options.metadata_default_colors
            numeric_colorbars.append(
                (
                    f"metadata:{col_name}",
                    display_col,
                    min_value,
                    max_value,
                    numeric_metadata_colorscale(colorbar_spec, col_name),
                )
            )
        if is_numeric:
            level_map = {}
        else:
            categorical_levels = _metadata_levels(prepared, col_name, values_by_sample, palette_spec)
            level_map = _metadata_color_map(
                categorical_levels,
                options,
                supplied=categorical_metadata_palette(palette_spec, col_name, categorical_levels),
            )
        for sample in prepared.samples:
            value = values_by_sample.get(sample, np.nan)
            row_text.append(f"{display_col}: {_metadata_value_label(value, options)}")
            if pd.isna(value):
                key = ("__NA__", col_name)
                color = "#D9D9D9"
            elif is_numeric:
                color = _metadata_numeric_color(value, min_value, max_value, col_name, options, numeric_colormap)
                key = ("__NUM__", col_name, color)
            else:
                key = (col_name, str(value))
                color = level_map[str(value)]
            if key not in color_lookup:
                color_lookup[key] = color_index
                colorscale.append([color_index, color])
                color_index += 1
            row_values.append(color_lookup[key])
            row_custom.append(
                _custom_payload(
                    role="metadata",
                    sample=str(sample),
                    column=str(col_name),
                    value=None if pd.isna(value) else value,
                )
            )
        matrix.append(row_values)
        text.append(row_text)
        customdata.append(row_custom)

        if not is_numeric and options.show_metadata_legends:
            for level, color in level_map.items():
                display_level = _legend_entry_label(level, options)
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(symbol="square", size=max(8, options.metadata_legend_key_size * 9), color=color),
                        name=f"{_truncate_text(display_col, options.legend_title_max_chars)}: {display_level}",
                        legendgroup=f"metadata:{col_name}",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

    max_value = max(color_index - 1, 1)
    normalized_scale = [[index / max_value, color] for index, color in colorscale]
    normalized_scale = sorted(normalized_scale, key=lambda item: item[0])

    fig.add_trace(
        go.Heatmap(
            x=prepared.samples,
            y=[_legend_title(column, options) for column in prepared.metadata_cols],
            z=matrix,
            text=text,
            customdata=customdata,
            hovertemplate="%{text}<extra></extra>",
            colorscale=normalized_scale,
            showscale=False,
            zmin=0,
            zmax=max_value,
        ),
        row=row,
        col=col,
    )
    _add_numeric_metadata_colorbars(
        fig,
        prepared,
        row,
        col,
        numeric_colorbars,
        options,
    )
    fig.update_xaxes(
        **_sample_axis_options(prepared),
        showticklabels=options.show_sample_ids,
        tickfont=_font_options(options.font_size_samples, options),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        **_metadata_axis_options(prepared, options),
        autorange="reversed",
        tickfont=_font_options(options.font_size_metadata, options),
        row=row,
        col=col,
    )


def _add_main_tiles(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    palette: Mapping[str, str],
    variant_value_palette: object,
    options: OncoplotOptions,
    copy_on_click: str,
):
    go, _make_subplots = _require_plotly()
    if _has_expanded_main_grid(prepared):
        _add_expanded_main_tiles(
            fig,
            prepared,
            row,
            col,
            palette,
            variant_value_palette,
            options,
            copy_on_click,
        )
        return

    show_mutation_legend = options.show_legend and options.mutation_legend_position != "none"
    z = np.zeros((len(prepared.genes), len(prepared.samples)))
    fig.add_trace(
        go.Heatmap(
            x=prepared.samples,
            y=prepared.genes,
            z=z,
            colorscale=[[0, options.background_color], [1, options.background_color]],
            showscale=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    tiles = prepared.tiles
    if prepared.variant_value_col is not None:
        if not tiles.empty:
            _add_continuous_main_tiles(
                fig,
                prepared,
                row,
                col,
                variant_value_palette,
                options,
                copy_on_click,
            )
        fig.update_yaxes(**_gene_axis_options(prepared), row=row, col=col)
        fig.update_yaxes(tickfont=_font_options(options.font_size_genes, options), row=row, col=col)
        fig.update_xaxes(
            **_sample_axis_options(prepared),
            tickfont=_font_options(options.font_size_samples, options),
            row=row,
            col=col,
        )
        if not options.show_sample_ids:
            fig.update_xaxes(showticklabels=False, ticks="", row=row, col=col)
        else:
            fig.update_xaxes(side=options.sample_id_position, tickangle=options.sample_id_angle, row=row, col=col)
        if options.show_x_label:
            fig.update_xaxes(
                title_text=options.x_label,
                title_font=_font_options(options.font_size_x_label, options),
                row=row,
                col=col,
            )
        if options.show_y_label:
            fig.update_yaxes(
                title_text=options.y_label,
                title_font=_font_options(options.font_size_y_label, options),
                row=row,
                col=col,
            )
        if prepared.pathway_groups:
            xref = f"x{'' if row == 1 and col == 1 else row}"
            yref = f"y{'' if row == 1 and col == 1 else row}"
            for group in prepared.pathway_groups:
                center_gene = group.genes[len(group.genes) // 2]
                fig.add_annotation(
                    xref=xref,
                    yref=yref,
                    x=prepared.samples[0] if prepared.samples else 0,
                    y=center_gene,
                    text=group.name,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    textangle=options.pathway_text_angle,
                    xshift=-16,
                    font=dict(color=options.pathway_text_color, size=options.font_size_pathway or 10),
                    bgcolor=options.pathway_background_color,
                    bordercolor=options.pathway_outline_color,
                    borderwidth=1,
                )
        return

    if not tiles.empty:
        mutation_values = _mutation_levels(prepared, palette)
        for mutation_type in mutation_values:
            group = tiles[tiles["MutationType"].astype(str) == mutation_type]
            color = palette.get(str(mutation_type), options.unspecified_mutation_color)
            customdata = [
                _custom_payload(
                    role="main_tile",
                    sample=str(row_value["Sample"]),
                    gene=str(row_value["Gene"]),
                    mutation_type=str(row_value["MutationType"]),
                    copy_value=_copy_value(row_value, copy_on_click),
                )
                for _index, row_value in group.iterrows()
            ]
            fig.add_trace(
                go.Scatter(
                    x=group["Sample"].astype(str),
                    y=group["Gene"].astype(str),
                    mode="markers",
                    marker=dict(
                        symbol="square",
                        size=13,
                        color=color,
                        line=dict(color="white", width=options.tile_linewidth),
                    ),
                    name=_legend_entry_label(mutation_type, options),
                    text=group["Tooltip"].astype(str),
                    customdata=customdata,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=show_mutation_legend,
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.18)),
                ),
                row=row,
                col=col,
            )
        unspecified = tiles[tiles["MutationType"].isna()]
        if not unspecified.empty:
            fig.add_trace(
                go.Scatter(
                    x=unspecified["Sample"].astype(str),
                    y=unspecified["Gene"].astype(str),
                    mode="markers",
                    marker=dict(
                        symbol="square",
                        size=13,
                        color=options.unspecified_mutation_color,
                        line=dict(color="white", width=options.tile_linewidth),
                    ),
                    name=_truncate_text("Mutation", options.legend_label_max_chars),
                    text=unspecified["Tooltip"].astype(str),
                    customdata=[
                        _custom_payload(
                            role="main_tile",
                            sample=str(row_value["Sample"]),
                            gene=str(row_value["Gene"]),
                            mutation_type="",
                            copy_value=_copy_value(row_value, copy_on_click),
                        )
                        for _index, row_value in unspecified.iterrows()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=show_mutation_legend and prepared.mutation_type_col is None,
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.18)),
                ),
                row=row,
                col=col,
            )

    fig.update_yaxes(**_gene_axis_options(prepared), row=row, col=col)
    fig.update_yaxes(tickfont=_font_options(options.font_size_genes, options), row=row, col=col)
    fig.update_xaxes(
        **_sample_axis_options(prepared),
        tickfont=_font_options(options.font_size_samples, options),
        row=row,
        col=col,
    )
    if not options.show_sample_ids:
        fig.update_xaxes(showticklabels=False, ticks="", row=row, col=col)
    else:
        fig.update_xaxes(side=options.sample_id_position, tickangle=options.sample_id_angle, row=row, col=col)
    if options.show_x_label:
        fig.update_xaxes(
            title_text=options.x_label,
            title_font=_font_options(options.font_size_x_label, options),
            row=row,
            col=col,
        )
    if options.show_y_label:
        fig.update_yaxes(
            title_text=options.y_label,
            title_font=_font_options(options.font_size_y_label, options),
            row=row,
            col=col,
        )

    if prepared.pathway_groups:
        xref = f"x{'' if row == 1 and col == 1 else row}"
        yref = f"y{'' if row == 1 and col == 1 else row}"
        for group in prepared.pathway_groups:
            center_gene = group.genes[len(group.genes) // 2]
            fig.add_annotation(
                xref=xref,
                yref=yref,
                x=prepared.samples[0] if prepared.samples else 0,
                y=center_gene,
                text=group.name,
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                textangle=options.pathway_text_angle,
                xshift=-16,
                font=dict(color=options.pathway_text_color, size=options.font_size_pathway or 10),
                bgcolor=options.pathway_background_color,
                bordercolor=options.pathway_outline_color,
                borderwidth=1,
            )


def _add_gene_bar(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    palette: Mapping[str, str],
    options: OncoplotOptions,
    show_gene_bar_legend: bool = False,
):
    go, _make_subplots = _require_plotly()
    tiles = prepared.tiles
    if tiles.empty:
        return

    expanded = _has_expanded_main_grid(prepared)
    if expanded:
        gene_centers = _main_grid_gene_centers(prepared)
        y_values = [gene_centers[gene] for gene in prepared.genes]
        row_counts = (
            prepared.main_grid_rows.groupby("Gene", sort=False, observed=False)["RowIndex"].nunique()
            if prepared.main_grid_rows is not None and not prepared.main_grid_rows.empty
            else pd.Series(1, index=prepared.genes)
        )
        bar_widths = [max(0.75, float(row_counts.get(gene, 1)) * 0.84) for gene in prepared.genes]
    else:
        y_values = prepared.genes
        bar_widths = 1

    total_counts = tiles.groupby("Gene", observed=False).size().reindex(prepared.genes, fill_value=0)
    denominator = max(prepared.total_samples, 1)
    mutation_groups = [
        (mutation_type, tiles[tiles["MutationType"].astype(str) == mutation_type])
        for mutation_type in _mutation_levels(prepared, palette)
    ]
    unspecified = tiles[tiles["MutationType"].isna()]
    if not unspecified.empty:
        mutation_groups.append((np.nan, unspecified))

    for mutation_type, group in mutation_groups:
        counts = group.groupby("Gene", observed=False).size().reindex(prepared.genes, fill_value=0)
        if options.gene_bar_mode == "percent":
            values = (counts / total_counts.replace(0, np.nan) * 100).fillna(0.0)
        else:
            values = counts
        mutation_label = _legend_label(mutation_type, options)
        mutation_legend_label = _legend_entry_label(mutation_type, options)
        hover = [
            "Total Samples Mutated: "
            f"{int(total_counts[gene])} ({as_percent(total_counts[gene] / max(prepared.total_samples, 1), options.gene_bar_label_round)} of all samples)"
            "<br>"
            f"{mutation_label}: {int(counts[gene])} "
            f"({as_percent(counts[gene] / max(total_counts[gene], 1), options.gene_bar_label_round)} of all mutations in this gene)"
            for gene in prepared.genes
        ]
        color = options.unspecified_mutation_color if pd.isna(mutation_type) else palette.get(str(mutation_type), options.unspecified_mutation_color)
        fig.add_trace(
            go.Bar(
                x=values.values,
                y=y_values,
                orientation="h",
                width=bar_widths,
                marker_color=color,
                hovertext=hover,
                customdata=[
                    _custom_payload(
                        role="gene_bar",
                        gene=str(gene),
                        count=int(total_counts[gene]),
                        mutation_type_count=int(counts[gene]),
                    )
                    for gene in prepared.genes
                ],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=show_gene_bar_legend,
                legendgroup=f"gene_bar:{mutation_label}",
                name=mutation_legend_label,
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.18)),
            ),
            row=row,
            col=col,
        )
    fig.update_layout(barmode="stack")
    if options.show_gene_bar_labels:
        max_total = 100.0 if options.gene_bar_mode == "percent" else max(float(total_counts.max()), 1.0)
        if options.gene_bar_mode == "percent":
            label_x = pd.Series(max_total, index=total_counts.index)
        else:
            label_x = total_counts.astype(float)
        label_x = label_x + max_total * options.gene_bar_label_padding + options.gene_bar_label_nudge
        fig.add_trace(
            go.Scatter(
                x=label_x.values,
                y=y_values,
                mode="text",
                text=[
                    as_percent(total_counts[gene] / denominator, options.gene_bar_label_round)
                    for gene in prepared.genes
                ],
                textposition="middle right",
                hoverinfo="skip",
                showlegend=False,
                cliponaxis=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(range=[0, max(label_x.max(), max_total) * 1.08], row=row, col=col)
    elif options.gene_bar_mode == "percent":
        fig.update_xaxes(range=[0, 100], row=row, col=col)
    if expanded:
        fig.update_yaxes(**_expanded_main_axis_options(prepared), showticklabels=False, row=row, col=col)
    else:
        fig.update_yaxes(
            **_gene_axis_options(prepared),
            showticklabels=False,
            row=row,
            col=col,
        )
    if not options.show_gene_bar_axis:
        fig.update_xaxes(showticklabels=False, ticks="", row=row, col=col)
    else:
        fig.update_xaxes(
            title_text="Mutation Type (%)" if options.gene_bar_mode == "percent" else "Samples",
            tickfont=_font_options(options.font_size_gene_bar_axis, options),
            title_font=_font_options(options.font_size_gene_bar_axis, options),
            row=row,
            col=col,
        )
    if options.gene_bar_scale_breaks is not None:
        fig.update_xaxes(tickmode="array", tickvals=list(options.gene_bar_scale_breaks), row=row, col=col)
    elif options.gene_bar_scale_n_breaks is not None:
        fig.update_xaxes(nticks=options.gene_bar_scale_n_breaks, row=row, col=col)


def render_plotly_oncoplot(
    prepared: Optional[PreparedOncoplotData] = None,
    *,
    params: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> object:
    """Render an interactive Plotly oncoplot."""

    supplied = merge_params(params, allowed_keys=PLOTLY_RENDER_PARAM_KEYS, context="plotly renderer", **kwargs)
    merged = {**PLOTLY_RENDER_DEFAULTS, **supplied}
    if prepared is None:
        prepared = merged.pop("prepared", None)
    else:
        merged.pop("prepared", None)
    if prepared is None:
        raise TypeError("render_plotly_oncoplot requires prepared data as the first argument or params['prepared'].")
    if "palette" not in merged:
        raise TypeError("render_plotly_oncoplot requires 'palette'.")
    if "options" not in merged:
        raise TypeError("render_plotly_oncoplot requires 'options'.")

    palette = merged["palette"]
    tmb_palette = merged["tmb_palette"]
    metadata_palette = merged["metadata_palette"]
    variant_value_palette = merged["variant_value_palette"]
    options = coerce_options(merged["options"])
    draw_gene_bar = merged["draw_gene_bar"]
    draw_tmb_bar = merged["draw_tmb_bar"]
    copy_on_click = merged["copy_on_click"]

    go, make_subplots = _require_plotly()
    draw_metadata = prepared.metadata is not None and bool(prepared.metadata_cols)
    if draw_metadata:
        _validate_metadata_levels(prepared, options)
    rows, row_heights, row_by_name = _grid_positions(prepared, draw_tmb_bar, draw_metadata, options)
    has_gene_bar = draw_gene_bar

    specs = []
    subplot_titles = []
    for name, _weight in rows:
        if not has_gene_bar:
            specs.append([{}])
            subplot_titles.append(
                options.metadata_subplot_title
                if name == "metadata"
                else options.tmb_subplot_title
                if name == "tmb"
                else options.main_subplot_title
                if name == "main"
                else None
            )
        elif name == "main":
            specs.append([{}, {} if has_gene_bar else None])
            subplot_titles.append(options.main_subplot_title)
            subplot_titles.append(options.gene_bar_subplot_title)
        else:
            specs.append([{}, None])
            subplot_titles.append(options.metadata_subplot_title if name == "metadata" else options.tmb_subplot_title)
    subplot_titles = [title or "" for title in subplot_titles]
    has_subplot_titles = any(subplot_titles)

    figure = make_subplots(
        rows=len(rows),
        cols=2 if has_gene_bar else 1,
        row_heights=row_heights,
        column_widths=[1 - options.gene_bar_width_ratio, options.gene_bar_width_ratio] if has_gene_bar else [1],
        horizontal_spacing=max(0.0, min(0.25, options.buffer_gene_bar)) if has_gene_bar else 0.01,
        vertical_spacing=max(0.0, min(0.25, max(options.buffer_tmb, options.buffer_metadata))),
        specs=specs,
        shared_xaxes=False,
        subplot_titles=subplot_titles if has_subplot_titles else None,
    )
    if has_subplot_titles:
        for annotation in figure.layout.annotations:
            annotation.font = _font_options(options.font_size_subplot_title, options)

    mutation_legend_visible = _main_grid_has_mutation_rows(prepared) or has_gene_bar
    if draw_tmb_bar and "tmb" in row_by_name:
        _add_tmb_bar(
            figure,
            prepared,
            row_by_name["tmb"],
            1,
            palette,
            tmb_palette,
            options,
            mutation_legend_visible=mutation_legend_visible,
        )
    if draw_metadata and "metadata" in row_by_name:
        _add_metadata_strip(figure, prepared, row_by_name["metadata"], 1, options, metadata_palette=metadata_palette)

    main_row = row_by_name["main"]
    _add_main_tiles(figure, prepared, main_row, 1, palette, variant_value_palette, options, copy_on_click)
    if has_gene_bar:
        _add_gene_bar(
            figure,
            prepared,
            main_row,
            2,
            palette,
            options,
            show_gene_bar_legend=_main_grid_has_continuous_rows(prepared)
            and not _main_grid_has_mutation_rows(prepared)
            and options.show_legend
            and options.mutation_legend_position != "none",
        )

    dragmode = {"none": "zoom", "multiple": "lasso", "single": "select"}[options.selection_type]
    mutation_legend_active = bool(
        options.show_legend
        and options.mutation_legend_position != "none"
        and (_main_grid_has_mutation_rows(prepared) or has_gene_bar)
    )
    tmb_render_stacked = (
        draw_tmb_bar
        and prepared.tmb is not None
        and prepared.tmb_render_stacked
        and not options.log10_transform_tmb
    )
    tmb_legend_active = _should_show_tmb_legend(
        prepared,
        options,
        tmb_render_stacked,
        palette,
        tmb_palette,
        mutation_legend_visible=mutation_legend_visible,
    )
    metadata_legend_active = options.show_metadata_legends and draw_metadata
    numeric_metadata_legend_active = bool(
        metadata_legend_active
        and prepared.metadata is not None
        and any(
            pd.api.types.is_numeric_dtype(prepared.metadata[column])
            and prepared.metadata[column].notna().any()
            for column in prepared.metadata_cols or []
        )
    )
    legend_active = mutation_legend_active or tmb_legend_active or metadata_legend_active
    bottom_legend = (
        (mutation_legend_active and options.mutation_legend_position == "bottom")
        or (tmb_legend_active and options.mutation_legend_position == "bottom")
        or (metadata_legend_active and options.metadata_legend_position == "bottom")
    )
    margin = dict(l=70, r=30, t=25, b=50)
    legend: Dict[str, object] = {
        "itemsizing": "constant",
        "itemwidth": max(30, int(options.legend_key_size * 30)),
    }
    if options.font_size_legend_text is not None:
        legend["font"] = _font_options(options.font_size_legend_text, options)
    if legend_active and bottom_legend:
        legend.update(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.18,
            yanchor="top",
        )
        margin["b"] = 90
    elif legend_active:
        legend.update(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        )
        margin["r"] = 150
    if numeric_metadata_legend_active and _metadata_colorbar_is_horizontal(options):
        margin["b"] = max(margin["b"], 135)
    elif numeric_metadata_legend_active:
        margin["r"] = max(margin["r"], 220)
    if _main_grid_has_continuous_rows(prepared) and options.show_legend and options.mutation_legend_position != "none":
        margin["r"] = max(margin["r"], 130)
    if options.title_text is not None:
        margin["t"] = max(margin["t"], 70)
    elif has_subplot_titles:
        margin["t"] = max(margin["t"], 50)
    mutation_key = f"mutation:{prepared.mutation_type_col}" if prepared.mutation_type_col is not None else "mutation"
    tmb_key = f"tmb:{prepared.tmb_type_col}" if prepared.tmb_type_col is not None else "tmb"
    extra_legends = _assign_plotly_offset_legends(
        figure,
        options,
        legend,
        mutation_key=mutation_key,
        tmb_key=tmb_key,
    )
    layout: Dict[str, object] = dict(
        width=options.width,
        height=options.height,
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=options.font_family),
        dragmode=dragmode,
        showlegend=legend_active,
        margin=margin,
        legend=legend,
    )
    if options.title_text is not None:
        layout["title"] = dict(
            text=options.title_text,
            font=_font_options(options.font_size_title, options),
            x=0.5,
            xanchor="center",
        )
    layout.update(extra_legends)
    figure.update_layout(**layout)
    return figure
