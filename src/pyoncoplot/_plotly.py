"""Plotly renderer for interactive oncoplots."""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._data import PreparedOncoplotData
from ._options import OncoplotOptions, coerce_options
from ._params import merge_params
from ._utils import as_percent


PLOTLY_RENDER_PARAM_KEYS = {
    "prepared",
    "palette",
    "tmb_palette",
    "metadata_palette",
    "options",
    "draw_gene_bar",
    "draw_tmb_bar",
    "copy_on_click",
}

PLOTLY_RENDER_DEFAULTS: dict[str, Any] = {
    "tmb_palette": None,
    "metadata_palette": None,
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


def _should_show_tmb_legend(
    prepared: PreparedOncoplotData,
    options: OncoplotOptions,
    render_stacked: bool,
    mutation_palette: Optional[Mapping[str, str]] = None,
    tmb_palette: Optional[Mapping[str, str]] = None,
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

    tmb_categories = [
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


def _add_tmb_bar(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    palette: Mapping[str, str],
    tmb_palette: Optional[Mapping[str, str]],
    options: OncoplotOptions,
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
    show_tmb_legend = _should_show_tmb_legend(prepared, options, render_stacked, palette, tmb_palette)
    if prepared.tmb_render_stacked and options.log10_transform_tmb:
        warnings.warn(
            "log10_transform_tmb=True disables stacked TMB rendering; totals are rendered instead.",
            stacklevel=2,
        )

    if render_stacked:
        tmb_color_palette = tmb_palette or palette
        for tmb_type, group in tmb.groupby(type_col, dropna=False, sort=False):
            color = tmb_color_palette.get(str(tmb_type), options.unspecified_mutation_color)
            sample_values = group[sample_col].astype(str).tolist()
            values = group[value_col].astype(float).tolist()
            tmb_label = _legend_label(tmb_type, options)
            fig.add_trace(
                go.Bar(
                    x=sample_values,
                    y=values,
                    name=f"TMB: {tmb_label}",
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
    fig.update_xaxes(showticklabels=False, ticks="", row=row, col=col)


def _metadata_color_map(
    values: pd.Series,
    options: OncoplotOptions,
    supplied: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    levels = [str(value) for value in pd.unique(values.dropna())]
    mapping = {
        level: options.metadata_default_colors[index % len(options.metadata_default_colors)]
        for index, level in enumerate(levels)
    }
    if supplied:
        mapping.update({str(level): color for level, color in supplied.items()})
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


def _add_metadata_strip(
    fig,
    prepared: PreparedOncoplotData,
    row: int,
    col: int,
    options: OncoplotOptions,
    metadata_palette: Optional[Mapping[str, Mapping[str, str]]] = None,
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
        level_map = (
            {}
            if is_numeric
            else _metadata_color_map(
                values_by_sample.astype("object"),
                options,
                supplied=metadata_palette.get(str(col_name), {}),
            )
        )
        for sample in prepared.samples:
            value = values_by_sample.get(sample, np.nan)
            row_text.append(f"{display_col}: {_metadata_value_label(value, options)}")
            if pd.isna(value):
                key = ("__NA__", col_name)
                color = "#D9D9D9"
            elif is_numeric:
                color = _numeric_metadata_color(value, min_value, max_value, options)
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
                display_level = _metadata_value_label(level, options)
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(symbol="square", size=max(8, options.metadata_legend_key_size * 9), color=color),
                        name=f"{display_col}: {display_level}",
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
    fig.update_xaxes(
        showticklabels=options.show_sample_ids,
        tickfont=_font_options(options.font_size_samples, options),
        row=row,
        col=col,
    )
    fig.update_yaxes(
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
    options: OncoplotOptions,
    copy_on_click: str,
):
    go, _make_subplots = _require_plotly()
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
    if not tiles.empty:
        mutation_values = [value for value in pd.unique(tiles["MutationType"]) if not pd.isna(value)]
        for mutation_type in mutation_values:
            group = tiles[tiles["MutationType"] == mutation_type]
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
                    name=_legend_label(mutation_type, options),
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
                    name="Mutation",
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

    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(prepared.genes)), row=row, col=col)
    fig.update_yaxes(tickfont=_font_options(options.font_size_genes, options), row=row, col=col)
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=prepared.samples,
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
                font=dict(color=options.pathway_text_color, size=10),
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
):
    go, _make_subplots = _require_plotly()
    tiles = prepared.tiles
    if tiles.empty:
        return

    total_counts = tiles.groupby("Gene", observed=False).size().reindex(prepared.genes, fill_value=0)
    denominator = max(prepared.total_samples, 1)
    for mutation_type, group in tiles.groupby("MutationType", dropna=False, sort=False):
        counts = group.groupby("Gene", observed=False).size().reindex(prepared.genes, fill_value=0)
        mutation_label = _legend_label(mutation_type, options)
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
                x=counts.values,
                y=prepared.genes,
                orientation="h",
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
                showlegend=False,
                name=_legend_label(mutation_type, options),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.18)),
            ),
            row=row,
            col=col,
        )
    fig.update_layout(barmode="stack")
    if options.show_gene_bar_labels:
        max_total = max(float(total_counts.max()), 1.0)
        label_x = (
            total_counts.astype(float)
            + max_total * options.gene_bar_label_padding
            + options.gene_bar_label_nudge
        )
        fig.add_trace(
            go.Scatter(
                x=label_x.values,
                y=prepared.genes,
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
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(prepared.genes)),
        showticklabels=False,
        row=row,
        col=col,
    )
    if not options.show_gene_bar_axis:
        fig.update_xaxes(showticklabels=False, ticks="", row=row, col=col)
    else:
        fig.update_xaxes(
            title_text="Samples",
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
    for name, _weight in rows:
        if not has_gene_bar:
            specs.append([{}])
        elif name == "main":
            specs.append([{}, {} if has_gene_bar else None])
        else:
            specs.append([{}, None])

    figure = make_subplots(
        rows=len(rows),
        cols=2 if has_gene_bar else 1,
        row_heights=row_heights,
        column_widths=[1 - options.gene_bar_width_ratio, options.gene_bar_width_ratio] if has_gene_bar else [1],
        horizontal_spacing=max(0.0, min(0.25, options.buffer_gene_bar)) if has_gene_bar else 0.01,
        vertical_spacing=max(0.0, min(0.25, max(options.buffer_tmb, options.buffer_metadata))),
        specs=specs,
        shared_xaxes=False,
    )

    if draw_tmb_bar and "tmb" in row_by_name:
        _add_tmb_bar(figure, prepared, row_by_name["tmb"], 1, palette, tmb_palette, options)
    if draw_metadata and "metadata" in row_by_name:
        _add_metadata_strip(figure, prepared, row_by_name["metadata"], 1, options, metadata_palette=metadata_palette)

    main_row = row_by_name["main"]
    _add_main_tiles(figure, prepared, main_row, 1, palette, options, copy_on_click)
    if has_gene_bar:
        _add_gene_bar(figure, prepared, main_row, 2, palette, options)

    dragmode = {"none": "zoom", "multiple": "lasso", "single": "select"}[options.selection_type]
    mutation_legend_active = options.show_legend and options.mutation_legend_position != "none"
    tmb_render_stacked = (
        draw_tmb_bar
        and prepared.tmb is not None
        and prepared.tmb_render_stacked
        and not options.log10_transform_tmb
    )
    tmb_legend_active = _should_show_tmb_legend(prepared, options, tmb_render_stacked, palette, tmb_palette)
    metadata_legend_active = options.show_metadata_legends and draw_metadata
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
    figure.update_layout(
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
    return figure
