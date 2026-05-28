"""Recreate goal gallery images from deterministic local inputs.

Generated PNGs are written under ``python_refactor_goal_sources/generated_plots``
so numbered source goal plots remain untouched.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import yaml

from pyoncoplot import load_oncoplot_params, oncoplot


GOAL_SOURCE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GOAL_SOURCE_ROOT.parent
CONFIG_PATH = GOAL_SOURCE_ROOT / "config.yaml"
GOAL_PLOTS = GOAL_SOURCE_ROOT / "goal_plots"
INPUTS = GOAL_SOURCE_ROOT / "syntheitic_goal_data"

def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config() -> Mapping[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "gallery_params" not in config:
        raise ValueError(f"{CONFIG_PATH} must contain a top-level gallery_params block.")
    return config["gallery_params"]


def _resolve_repo_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


GALLERY_CONFIG = _load_config()
OUTPUT_DIRS = GALLERY_CONFIG.get("output_dirs", {})
GENERATED_ROOT = GOAL_SOURCE_ROOT / "generated_plots"
CLEAN_OUT = _resolve_repo_path(OUTPUT_DIRS.get("clean", GENERATED_ROOT / "clean"))
COMPARISON_OUT = _resolve_repo_path(OUTPUT_DIRS.get("comparison", GENERATED_ROOT / "comparison"))
OUT = CLEAN_OUT


@dataclass(frozen=True)
class GalleryPreset:
    name: str
    output_name: str
    renderer: Callable[..., None]
    expected_size: tuple[int, int]
    goal_plot: str
    params: Mapping[str, Any]
    run: bool = True
    clean_only: bool = False
    feature_axes_min: int = 1


def _read_tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(INPUTS / name, sep="\t")


def _read_palette(name: str) -> Dict[str, str]:
    return json.loads((INPUTS / name).read_text(encoding="utf-8"))


def _input_file(family: str, key: str) -> str:
    return str(GALLERY_CONFIG["input_files"][family][key])


def _load_run_oncoplot_params(run_name: Optional[str]) -> dict[str, Any]:
    if not run_name:
        raise ValueError("Config-backed gallery oncoplot renderers require a run_name.")
    return load_oncoplot_params(CONFIG_PATH, key=f"gallery_params.plot_runs.{run_name}.params.oncoplot")


def _save_exact(result, output_path: Path, dpi: int = 120) -> None:
    result.figure.savefig(output_path, dpi=dpi)


def _save_dpi(params: Mapping[str, Any], default: int = 120) -> int:
    save_params = params.get("save", {})
    if save_params is None:
        return default
    if not isinstance(save_params, Mapping):
        raise TypeError("gallery save params must be a mapping.")
    return int(save_params.get("dpi", default))


def _output_width(params: Mapping[str, Any], dpi: int) -> float:
    figure_size = params.get("figure_size", [1, 1])
    return float(figure_size[0]) * dpi


def _multimodal_marker_style(params: Mapping[str, Any], dpi: int) -> dict[str, float]:
    marker_params = params.get("marker_style", {})
    if not isinstance(marker_params, Mapping):
        marker_params = {}
    output_width = _output_width(params, dpi)
    reference_width = float(marker_params.get("reference_output_width", 2281))
    max_scale = float(marker_params.get("max_scale", 3.0))
    scale = min(max_scale, max(1.0, output_width / reference_width))
    return {
        "point_size": float(marker_params.get("point_size", 170)) * scale**2,
        "selected_point_size": float(marker_params.get("selected_point_size", 340)) * scale**2,
        "edge_linewidth": float(marker_params.get("edge_linewidth", 0.35)) * scale,
        "selected_linewidth": float(marker_params.get("selected_linewidth", 1.9)) * scale,
    }


def _multimodal_title_font_size(params: Mapping[str, Any], dpi: int) -> float:
    title_params = params.get("title_style", {})
    if not isinstance(title_params, Mapping):
        title_params = {}
    minimum = float(title_params.get("min_font_size", 14))
    maximum = float(title_params.get("max_font_size", 52))
    width_divisor = float(title_params.get("width_divisor", 160))
    return max(minimum, min(maximum, _output_width(params, dpi) / width_divisor))


def _multimodal_panel_title_font_size(params: Mapping[str, Any], dpi: int) -> float:
    title_params = params.get("panel_title_style", {})
    if not isinstance(title_params, Mapping):
        title_params = {}
    minimum = float(title_params.get("min_font_size", 9))
    maximum = float(title_params.get("max_font_size", 30))
    width_divisor = float(title_params.get("width_divisor", 280))
    return max(minimum, min(maximum, _output_width(params, dpi) / width_divisor))


def _axes_from_params(params: Mapping[str, Any], required: Sequence[str]) -> dict[str, Sequence[float]]:
    axes = params.get("axes", {})
    if not isinstance(axes, Mapping):
        raise TypeError("gallery axes params must be a mapping.")
    missing = [name for name in required if name not in axes]
    if missing:
        raise ValueError(f"Missing configured axes: {', '.join(missing)}.")
    return dict(axes)


def _filter_aml_inputs(
    mutations: pd.DataFrame,
    metadata: pd.DataFrame,
    tmb: pd.DataFrame,
    filter_params: Optional[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not filter_params:
        return mutations, metadata, tmb
    column = str(filter_params["column"])
    value = str(filter_params["value"])
    selected = metadata.loc[metadata[column].astype(str) == value, "sample"].astype(str)
    selected_samples = set(selected)
    return (
        mutations[mutations["sample"].astype(str).isin(selected_samples)].copy(),
        metadata[metadata["sample"].astype(str).isin(selected_samples)].copy(),
        tmb[tmb["sample"].astype(str).isin(selected_samples)].copy(),
    )


def _filter_aml_oncoplot_params(
    oncoplot_params: Mapping[str, Any],
    filter_params: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if not filter_params:
        return dict(oncoplot_params)
    filtered = dict(oncoplot_params)
    mutations, metadata, tmb = _filter_aml_inputs(
        filtered["data"],
        filtered["metadata"],
        filtered["tmb_data"],
        filter_params,
    )
    selected_samples = set(metadata["sample"].astype(str))
    sample_order = filtered.get("sample_order")
    if sample_order is not None:
        filtered["sample_order"] = [sample for sample in sample_order if str(sample) in selected_samples]
    filtered["data"] = mutations
    filtered["metadata"] = metadata
    filtered["tmb_data"] = tmb
    return filtered


def _load_readme():
    mutations = _read_tsv(_input_file("readme", "mutations"))
    metadata = _read_tsv(_input_file("readme", "metadata"))
    tmb = _read_tsv(_input_file("readme", "tmb"))
    palette = _read_palette(_input_file("readme", "palette"))
    return mutations, metadata, tmb, palette


def _load_multimodal():
    samples = _read_tsv(_input_file("multimodal", "samples"))
    points = _read_tsv(_input_file("multimodal", "points"))
    events = _read_tsv(_input_file("multimodal", "events"))
    clinical = _read_tsv(_input_file("multimodal", "clinical"))
    selection = _read_tsv(_input_file("multimodal", "selection"))
    palette = _read_palette(_input_file("multimodal", "palette"))
    return samples, points, events, clinical, selection, palette


def _load_comparison_table():
    return _read_tsv(_input_file("comparison_table", "table"))


def _group_events(events: pd.DataFrame, sample_col: str = "sample", gene_col: str = "gene", type_col: str = "alteration") -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in events.itertuples(index=False):
        sample = str(getattr(row, sample_col))
        gene = str(getattr(row, gene_col))
        alteration = str(getattr(row, type_col))
        grouped.setdefault((sample, gene), [])
        if alteration not in grouped[(sample, gene)]:
            grouped[(sample, gene)].append(alteration)
    return grouped


def _draw_split_tile(ax, x: float, y: float, alterations: Sequence[str], palette: Mapping[str, str], default_color: str = "#1A1A1A") -> None:
    from matplotlib.patches import Rectangle

    if not alterations:
        return
    width = 1 / len(alterations)
    for index, alteration in enumerate(alterations):
        ax.add_patch(
            Rectangle(
                (x + index * width, y),
                width,
                1,
                facecolor=palette.get(alteration, default_color),
                edgecolor="white",
                linewidth=0.02,
            )
        )


# Generates all config-backed oncoplot gallery presets.
def render_oncoplot(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, run_name: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    merged = dict(params or {})
    oncoplot_params = _filter_aml_oncoplot_params(
        _load_run_oncoplot_params(run_name),
        merged.get("metadata_filter"),
    )
    result = oncoplot(params=oncoplot_params)
    dpi = _save_dpi(merged, default=120)
    options = oncoplot_params.get("options", {})
    if result.backend == "matplotlib" and isinstance(options, Mapping):
        result.figure.set_size_inches(float(options["width"]) / dpi, float(options["height"]) / dpi, forward=True)
    _save_exact(result, output_path, dpi=dpi)
    if result.backend == "matplotlib":
        plt.close(result.figure)


# Generates gen.goal_plot_21.png from the structural variation panel preset.
def render_sv_panel(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = {**dict(params or {}), **kwargs}
    depth = _read_tsv(_input_file("sv", "depth"))
    allele = _read_tsv(_input_file("sv", "allele_fraction"))
    gene_models = _read_tsv(_input_file("sv", "gene_models"))
    samples = list(merged["samples"])
    fig, axes = plt.subplots(2, len(samples), figsize=tuple(merged["figure_size"]), sharex=True)
    depth_ylim = tuple(merged.get("depth_ylim", [-15, 120]))
    allele_ylim = tuple(merged.get("allele_ylim", [-0.12, 1.1]))
    for col, sample in enumerate(samples):
        sample_depth = depth[depth["sample"] == sample]
        sample_allele = allele[allele["sample"] == sample]
        axes[0, col].scatter(sample_depth["position"], sample_depth["depth"], s=16, color="#2CA02C")
        axes[0, col].set_title(sample_depth["title"].iloc[0], fontsize=18)
        axes[0, col].set_ylim(*depth_ylim)
        axes[0, col].grid(True, color="white")
        for gene, models in gene_models.groupby("gene", sort=False):
            for _, model in models.iterrows():
                axes[0, col].add_patch(Rectangle((model["start"], -7), model["end"] - model["start"], 4, color="black"))
            axes[0, col].text((models["start"].min() + models["end"].max()) / 2, -10, gene, ha="center", va="top", fontsize=9, style="italic")

        for allele_name, color in [("REF", "#4C72B0"), ("ALT", "#DD8452")]:
            group = sample_allele[sample_allele["allele"] == allele_name]
            axes[1, col].scatter(group["position"], group["allele_fraction"], s=14, label=allele_name, color=color)
        axes[1, col].set_ylim(*allele_ylim)
        axes[1, col].grid(True, color="white")
        axes[1, col].set_xlabel("Position", fontsize=14)
        for gene, models in gene_models.groupby("gene", sort=False):
            for _, model in models.iterrows():
                axes[1, col].add_patch(Rectangle((model["start"], -0.08), model["end"] - model["start"], 0.04, color="black"))
            axes[1, col].text((models["start"].min() + models["end"].max()) / 2, -0.10, gene, ha="center", va="top", fontsize=9, style="italic")
    axes[0, 0].set_ylabel("Depth", fontsize=14)
    axes[1, 0].set_ylabel("Allele fraction", fontsize=14)
    axes[1, 0].legend(loc="upper left", fontsize=16, frameon=True)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#EAEAF2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=_save_dpi(merged, default=100))
    plt.close(fig)


# Generates gen.goal_plot_6.png from the package mark preset.
def render_package_mark(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    merged = {**dict(params or {}), **kwargs}
    dpi = _save_dpi(merged, default=120)
    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.10, 0.12), 0.80, 0.70, facecolor="#F7F7F7", edgecolor="#1F1F1F", linewidth=1.1))
    palette = ["#4DAF4A", "#377EB8", "#E41A1C", "#984EA3", "#FF7F00"]
    for row in range(5):
        for col in range(6):
            face = "#FFFFFF" if (row + col) % 3 else palette[(row + col) % len(palette)]
            ax.add_patch(Rectangle((0.17 + col * 0.105, 0.22 + row * 0.105), 0.08, 0.075, facecolor=face, edgecolor="#D0D0D0", linewidth=0.4))
    ax.add_patch(Circle((0.24, 0.86), 0.055, facecolor="#4DAF4A", edgecolor="#1F1F1F", linewidth=0.8))
    ax.add_patch(Circle((0.40, 0.86), 0.055, facecolor="#377EB8", edgecolor="#1F1F1F", linewidth=0.8))
    ax.add_patch(Circle((0.56, 0.86), 0.055, facecolor="#E41A1C", edgecolor="#1F1F1F", linewidth=0.8))
    ax.text(0.50, 0.075, str(merged.get("title", "Pyoncoplot")), ha="center", va="center", fontsize=9, weight="bold")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# Generates gen.goal_plot_7.png from the interactive snapshot preset.
def render_interactive_snapshot(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    merged = {**dict(params or {}), **kwargs}
    dpi = _save_dpi(merged, default=100)
    mutations, metadata, _tmb, palette = _load_readme()
    samples = metadata.sort_values(["Subtype", "sample"])["sample"].astype(str).tolist()[:48]
    genes = list(merged["genes"])
    grouped = _group_events(
        mutations[mutations["gene"].isin(genes)].rename(columns={"mutation_type": "alteration"}),
        type_col="alteration",
    )
    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax = fig.add_axes([0.06, 0.12, 0.70, 0.74])
    ax_bar = fig.add_axes([0.77, 0.12, 0.10, 0.74])
    ax_legend = fig.add_axes([0.88, 0.18, 0.10, 0.58])
    ax_legend.axis("off")
    for y, gene in enumerate(genes):
        for x, sample in enumerate(samples):
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#E6E6E6", edgecolor="white", linewidth=0.25))
            _draw_split_tile(ax, x, y, grouped.get((sample, gene), []), palette)
    ax.add_patch(Rectangle((5, 1), 11, 3, fill=False, edgecolor="#E43D30", linewidth=2.2, linestyle="--"))
    ax.set_xlim(0, len(samples))
    ax.set_ylim(0, len(genes))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(genes)) + 0.5)
    ax.set_yticklabels(genes, fontsize=10, fontstyle="italic")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    counts = mutations[mutations["gene"].isin(genes)].groupby("gene").size().reindex(genes, fill_value=0)
    ax_bar.barh(np.arange(len(genes)) + 0.5, counts.values, color="#4DAF4A", height=0.78)
    ax_bar.invert_yaxis()
    ax_bar.set_yticks([])
    ax_bar.xaxis.set_ticks_position("top")
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.spines[["left", "right", "bottom"]].set_visible(False)
    bubble = FancyBboxPatch((0.33, 0.70), 0.28, 0.18, boxstyle="round,pad=0.02", transform=fig.transFigure, facecolor="white", edgecolor="#222222", linewidth=1.0)
    fig.patches.append(bubble)
    fig.text(0.35, 0.82, "Clicked sample", fontsize=12, weight="bold")
    fig.text(0.35, 0.78, "README-012", fontsize=11)
    fig.text(0.35, 0.74, "Copied to clipboard", fontsize=10, color="#E43D30")
    fig.suptitle(str(merged.get("title", "Interactive oncoplot snapshot")), fontsize=18, weight="bold")
    y_cursor = 0.96
    for label, color in list(palette.items())[:6]:
        ax_legend.add_patch(Rectangle((0.0, y_cursor - 0.035), 0.08, 0.04, facecolor=color, edgecolor="#333333", linewidth=0.3))
        ax_legend.text(0.11, y_cursor - 0.015, label, fontsize=11, va="center")
        y_cursor -= 0.09
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# Generates gen.goal_plot_8.png and gen.goal_plot_9.png from comparison table presets.
def render_comparison_table(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = {**dict(params or {}), **kwargs}
    table = _load_comparison_table()
    dpi = _save_dpi(merged, default=100)
    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.84])
    ax.axis("off")
    columns = list(table.columns)
    cell_text = table.astype(str).values.tolist()
    tbl = ax.table(cellText=cell_text, colLabels=columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7 if merged.get("compact") else 8)
    tbl.scale(1, 1.35 if merged.get("compact") else 1.55)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.65)
        if row == 0:
            cell.set_facecolor("#244A7F")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col == 0:
            cell.set_facecolor("#E6EEF8")
            cell.get_text().set_weight("bold")
        elif table.iloc[row - 1, col] == "Yes":
            cell.set_facecolor("#DFF0D8")
        elif table.iloc[row - 1, col] in {"No", "Limited"}:
            cell.set_facecolor("#F2DEDE")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
    fig.add_artist(Rectangle((0.02, 0.03), 0.96, 0.91, transform=fig.transFigure, fill=False, edgecolor="#111111", linewidth=1.0))
    fig.text(0.5, 0.965, str(merged.get("title", "Oncoplot package comparison")), ha="center", va="top", fontsize=13, weight="bold")
    if merged.get("subtitle"):
        fig.text(0.5, 0.925, str(merged["subtitle"]), ha="center", va="top", fontsize=8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# Generates gen.goal_plot_10.png from the lasso scatter preset.
def render_lasso_scatter(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    merged = {**dict(params or {}), **kwargs}
    _samples, points, _events, _clinical, selection, palette = _load_multimodal()
    dpi = _save_dpi(merged, default=100)
    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax = fig.add_axes([0.09, 0.11, 0.82, 0.78])
    for classification, group in points.groupby("classification", sort=False):
        ax.scatter(group["tsne_x"], group["tsne_y"], s=34, color=palette.get(str(classification), "#999999"), label=str(classification), alpha=0.78, edgecolor="white", linewidth=0.35)
    selected = points[points["selected"]]
    ax.scatter(selected["tsne_x"], selected["tsne_y"], s=64, facecolor="none", edgecolor=palette["Selected"], linewidth=1.5)
    ax.add_patch(Polygon(selection[["x", "y"]].to_numpy(), closed=True, fill=False, edgecolor=palette["Selected"], linewidth=2.5, linestyle="--"))
    ax.set_title(str(merged.get("title", "Lasso-selected t-SNE cluster")), fontsize=18, weight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#E6E6E6")
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _draw_multimodal_oncoplot(ax, events: pd.DataFrame, samples: Sequence[str], genes: Sequence[str], palette: Mapping[str, str], selected_samples: set[str]) -> None:
    from matplotlib.patches import Rectangle

    grouped = _group_events(events)
    for y, gene in enumerate(genes):
        for x, sample in enumerate(samples):
            background = "#FFE9E9" if sample in selected_samples else "#F1F1F1"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=background, edgecolor="white", linewidth=0.12))
            _draw_split_tile(ax, x, y, grouped.get((sample, gene), []), palette)
    ax.set_xlim(0, len(samples))
    ax.set_ylim(0, len(genes))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(genes)) + 0.5)
    ax.set_yticklabels(genes, fontsize=7, fontstyle="italic")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# Generates gen.goal_plot_11.png through gen.goal_plot_13.png from multimodal presets.
def render_multimodal_panel(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    merged = {**dict(params or {}), **kwargs}
    _samples, points, events, clinical, selection, palette = _load_multimodal()
    dpi = _save_dpi(merged, default=100)
    marker_style = _multimodal_marker_style(merged, dpi)
    title_font_size = _multimodal_title_font_size(merged, dpi)
    panel_title_font_size = _multimodal_panel_title_font_size(merged, dpi)
    genes = list(merged["genes"])
    tracks = list(merged.get("tracks", ["PR_status", "ER_status", "HER2_status", "Classification"]))
    selected_samples = set(points.loc[points["selected"], "sample"].astype(str))
    ordered_samples = points.sort_values(["selected", "classification", "sample"], ascending=[False, True, True])["sample"].astype(str).tolist()
    axes_config = _axes_from_params(merged, ["tsne", "umap", "main", "clinical", "legend"])
    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax_tsne = fig.add_axes(axes_config["tsne"])
    ax_umap = fig.add_axes(axes_config["umap"])
    ax_main = fig.add_axes(axes_config["main"])
    ax_clinical = fig.add_axes(axes_config["clinical"])
    ax_legend = fig.add_axes(axes_config["legend"])
    ax_legend.axis("off")

    for ax, prefix, title in [(ax_tsne, "tsne", "Expression t-SNE"), (ax_umap, "umap", "Methylation UMAP")]:
        for classification, group in points.groupby("classification", sort=False):
            ax.scatter(
                group[f"{prefix}_x"],
                group[f"{prefix}_y"],
                s=marker_style["point_size"],
                color=palette.get(str(classification), "#999999"),
                alpha=0.78,
                edgecolor="white",
                linewidth=marker_style["edge_linewidth"],
            )
        selected = points[points["selected"]]
        ax.scatter(
            selected[f"{prefix}_x"],
            selected[f"{prefix}_y"],
            s=marker_style["selected_point_size"],
            facecolor="none",
            edgecolor=palette["Selected"],
            linewidth=marker_style["selected_linewidth"],
        )
        if prefix == "tsne" and merged.get("show_lasso", False):
            ax.add_patch(Polygon(selection[["x", "y"]].to_numpy(), closed=True, fill=False, edgecolor=palette["Selected"], linewidth=1.8, linestyle="--"))
        ax.set_title(title, fontsize=panel_title_font_size, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, color="#EFEFEF", linewidth=0.4)

    _draw_multimodal_oncoplot(ax_main, events, ordered_samples, genes, palette, selected_samples)
    ax_main.set_title("Linked oncoplot", fontsize=panel_title_font_size, weight="bold")

    clinical_indexed = clinical.set_index("sample")
    ax_clinical.set_xlim(0, len(ordered_samples))
    ax_clinical.set_ylim(0, len(tracks))
    ax_clinical.invert_yaxis()
    for y, track in enumerate(tracks):
        for x, sample in enumerate(ordered_samples):
            value = str(clinical_indexed.loc[sample, track])
            color = palette.get(value, "#FFFFFF")
            ax_clinical.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=0.08))
    ax_clinical.set_yticks(np.arange(len(tracks)) + 0.5)
    ax_clinical.set_yticklabels(tracks, fontsize=7)
    ax_clinical.set_xticks([])
    ax_clinical.tick_params(length=0)
    for spine in ax_clinical.spines.values():
        spine.set_visible(False)

    y_cursor = 0.95
    for label in ["Triple Negative", "Not Triple Negative", "Ambiguous", "Mutation", "Amplification", "Deletion"]:
        ax_legend.add_patch(Rectangle((0.0, y_cursor - 0.035), 0.07, 0.04, facecolor=palette.get(label, "#999999"), edgecolor="#333333", linewidth=0.3))
        ax_legend.text(0.09, y_cursor - 0.015, label, fontsize=11, va="center")
        y_cursor -= 0.10
    fig.suptitle(
        str(merged.get("title", "Multimodal sample selection")),
        fontsize=title_font_size,
        weight="bold",
        y=0.985,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


RENDERERS: Dict[str, Callable[..., None]] = {
    "oncoplot": render_oncoplot,
    "sv_panel": render_sv_panel,
    "package_mark": render_package_mark,
    "interactive_snapshot": render_interactive_snapshot,
    "comparison_table": render_comparison_table,
    "lasso_scatter": render_lasso_scatter,
    "multimodal_panel": render_multimodal_panel,
}

def _configured_runs() -> dict[str, dict[str, Any]]:
    defaults = GALLERY_CONFIG.get("default_params", {})
    runs = GALLERY_CONFIG.get("plot_runs", {})
    configured: dict[str, dict[str, Any]] = {}
    for name, run_config in runs.items():
        merged = _deep_merge(defaults, run_config or {})
        nested_params = _deep_merge(defaults.get("params", {}), merged.get("params", {}))
        if "save" in merged:
            nested_params = _deep_merge({"save": merged["save"]}, nested_params)
        if "backend" in merged and isinstance(nested_params.get("oncoplot"), Mapping):
            nested_params["oncoplot"] = _deep_merge({"backend": merged["backend"]}, nested_params["oncoplot"])
        merged["params"] = nested_params
        configured[name] = merged
    return configured


CONFIGURED_RUNS = _configured_runs()


def _renderer_from_config(run_name: str, run_config: Mapping[str, Any]) -> Callable[..., None]:
    renderer_name = run_config["renderer"]
    try:
        return RENDERERS[renderer_name]
    except KeyError as exc:
        available = ", ".join(sorted(RENDERERS))
        raise ValueError(f"Unknown renderer {renderer_name!r} for {run_name!r}. Available renderers: {available}.") from exc


def _build_gallery_presets() -> Dict[str, GalleryPreset]:
    presets: Dict[str, GalleryPreset] = {}
    for name, run_config in CONFIGURED_RUNS.items():
        if run_config.get("style") != "clean":
            continue
        presets[name] = GalleryPreset(
            name=name,
            output_name=run_config["output_name"],
            renderer=_renderer_from_config(name, run_config),
            expected_size=tuple(run_config["expected_size"]),
            goal_plot=run_config["goal_plot"],
            params=run_config.get("params", {}),
            run=bool(run_config.get("run", False)),
            clean_only=bool(run_config.get("clean_only", False)),
            feature_axes_min=int(run_config.get("feature_axes_min", 1)),
        )
    return presets


GALLERY_PRESETS = _build_gallery_presets()


# Clean gallery images are dispatched through GALLERY_PRESETS built from config.yaml.
def render_preset(name: str, out_dir: Optional[Path] = None, style: str = "clean") -> Path:
    if style == "comparison":
        return render_brca_comparison_sheet(name, out_dir or COMPARISON_OUT)
    if style != "clean":
        raise ValueError("style must be one of: clean, comparison.")

    if name not in GALLERY_PRESETS:
        available = ", ".join(sorted(GALLERY_PRESETS))
        raise ValueError(f"Unknown gallery preset {name!r}. Available presets: {available}")
    out_dir = out_dir or CLEAN_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = GALLERY_PRESETS[name]
    output_path = out_dir / preset.output_name
    if preset.renderer is render_oncoplot:
        preset.renderer(output_path, params=preset.params, run_name=name)
    else:
        preset.renderer(output_path, params=preset.params)
    return output_path


# Comparison sheets write compare.goal_plot_1.png and compare.goal_plot_15.png.
def render_brca_comparison_sheet(name: str, out_dir: Optional[Path] = None) -> Path:
    comparison_config = GALLERY_CONFIG.get("comparison_runs", {})
    base_name = name
    if base_name not in comparison_config:
        if base_name == "brca_compact":
            base_name = "brca_compact_complex"
    if base_name not in comparison_config:
        raise ValueError("Comparison sheets are only available for configured BRCA gallery presets.")
    run_config = _deep_merge(GALLERY_CONFIG.get("default_params", {}), comparison_config[base_name])
    clean_name = run_config["clean_preset"]
    clean_preset = GALLERY_PRESETS[clean_name]
    out_dir = out_dir or COMPARISON_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / run_config["output_name"]

    with tempfile.TemporaryDirectory(prefix="pyoncoplot_compare_") as temporary_dir:
        work_dir = Path(temporary_dir)
        clean_path = render_preset(clean_name, work_dir / "clean", style="clean")
        original_path = GOAL_PLOTS / clean_preset.goal_plot
        labels = ["Original reference", "Clean generated"]
        paths = [original_path, clean_path]
        thumb_width = 620
        thumb_height = 360
        label_height = 38
        canvas = Image.new("RGB", (thumb_width * len(paths), thumb_height + label_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (label, path) in enumerate(zip(labels, paths)):
            if path.exists():
                with Image.open(path) as source_image:
                    image = source_image.convert("RGB")
                image.thumbnail((thumb_width - 20, thumb_height - 20), Image.Resampling.LANCZOS)
                x = index * thumb_width + (thumb_width - image.width) // 2
                y = label_height + (thumb_height - image.height) // 2
                canvas.paste(image, (x, y))
            draw.text((index * thumb_width + 12, 10), label, fill="black")
        canvas.save(output_path)
    return output_path


def _selected_clean_presets(names: Optional[Sequence[str]]) -> list[str]:
    if not names:
        return [name for name, preset in GALLERY_PRESETS.items() if preset.run]
    selected = []
    for name in names:
        if name in GALLERY_PRESETS:
            selected.append(name)
        else:
            raise ValueError(f"Unknown gallery preset {name!r}.")
    return selected


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=["clean", "comparison", "both"], default="clean")
    parser.add_argument("--preset", action="append", help="Preset name to render. May be supplied more than once.")
    parser.add_argument("--out-dir", type=Path, help="Override the output directory for the selected style.")
    args = parser.parse_args(argv)

    if args.style in {"clean", "both"}:
        clean_out = args.out_dir if args.style == "clean" and args.out_dir else CLEAN_OUT
        for name in _selected_clean_presets(args.preset):
            render_preset(name, clean_out, style="clean")
        print(f"Wrote clean gallery images to {clean_out}")

    if args.style in {"comparison", "both"}:
        comparison_out = args.out_dir if args.style == "comparison" and args.out_dir else COMPARISON_OUT
        comparison_names = args.preset or ["brca_large"]
        for name in comparison_names:
            render_preset(name, comparison_out, style="comparison")
        print(f"Wrote comparison sheets to {comparison_out}")


if __name__ == "__main__":
    main()
