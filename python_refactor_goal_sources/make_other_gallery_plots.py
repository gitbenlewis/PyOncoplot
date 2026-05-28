"""Render custom gallery presets that are not covered by public ``oncoplot()``."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pyoncoplot import oncoplot

try:
    from python_refactor_goal_sources.make_oncoplots import (
        CLEAN_OUT,
        COMPARISON_OUT,
        CONFIG_PATH,
        GALLERY_CONFIG,
        GOAL_PLOTS,
        INPUTS,
    )
except ModuleNotFoundError:  # pragma: no cover - supports ``python path/to/script.py``.
    from make_oncoplots import (  # type: ignore
        CLEAN_OUT,
        COMPARISON_OUT,
        CONFIG_PATH,
        GALLERY_CONFIG,
        GOAL_PLOTS,
        INPUTS,
    )


def _load_readme():
    files = GALLERY_CONFIG["input_files"]["readme"]
    mutations = pd.read_csv(INPUTS / files["mutations"], sep="\t")
    metadata = pd.read_csv(INPUTS / files["metadata"], sep="\t")
    tmb = pd.read_csv(INPUTS / files["tmb"], sep="\t")
    palette = json.loads((INPUTS / files["palette"]).read_text(encoding="utf-8"))
    return mutations, metadata, tmb, palette


def _load_multimodal():
    files = GALLERY_CONFIG["input_files"]["multimodal"]
    samples = pd.read_csv(INPUTS / files["samples"], sep="\t")
    points = pd.read_csv(INPUTS / files["points"], sep="\t")
    events = pd.read_csv(INPUTS / files["events"], sep="\t")
    clinical = pd.read_csv(INPUTS / files["clinical"], sep="\t")
    selection = pd.read_csv(INPUTS / files["selection"], sep="\t")
    palette = json.loads((INPUTS / files["palette"]).read_text(encoding="utf-8"))
    return samples, points, events, clinical, selection, palette


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


# Generates gen.goal_plot_20.png from the structural variation panel preset.
def render_sv_panel(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = {**dict(params or {}), **kwargs}
    files = GALLERY_CONFIG["input_files"]["sv"]
    depth = pd.read_csv(INPUTS / files["depth"], sep="\t")
    allele = pd.read_csv(INPUTS / files["allele_fraction"], sep="\t")
    gene_models = pd.read_csv(INPUTS / files["gene_models"], sep="\t")
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
    fig.savefig(output_path, dpi=int(merged["save"]["dpi"]))
    plt.close(fig)


# Generates gen.goal_plot_06.png from the package mark preset.
def render_package_mark(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    merged = {**dict(params or {}), **kwargs}
    dpi = int(merged["save"]["dpi"])
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


# Generates gen.goal_plot_07.png from the interactive snapshot preset.
def render_interactive_snapshot(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    merged = {**dict(params or {}), **kwargs}
    dpi = int(merged["save"]["dpi"])
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


# Generates gen.goal_plot_08.png from the compact comparison table preset.
def render_comparison_table(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = {**dict(params or {}), **kwargs}
    files = GALLERY_CONFIG["input_files"]["comparison_table"]
    table = pd.read_csv(INPUTS / files["table"], sep="\t")
    dpi = int(merged["save"]["dpi"])
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


# Generates gen.goal_plot_09.png from the lasso scatter preset.
def render_lasso_scatter(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    merged = {**dict(params or {}), **kwargs}
    _samples, points, _events, _clinical, selection, palette = _load_multimodal()
    dpi = int(merged["save"]["dpi"])
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


# Generates gen.goal_plot_10.png through gen.goal_plot_12.png from multimodal presets.
def render_multimodal_panel(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    merged = {**dict(params or {}), **kwargs}
    _samples, points, events, clinical, selection, palette = _load_multimodal()
    dpi = int(merged["save"]["dpi"])
    output_width = float(merged["figure_size"][0]) * dpi
    marker_params = merged["marker_style"]
    scale = min(float(marker_params["max_scale"]), max(1.0, output_width / float(marker_params["reference_output_width"])))
    marker_style = {
        "point_size": float(marker_params["point_size"]) * scale**2,
        "selected_point_size": float(marker_params["selected_point_size"]) * scale**2,
        "edge_linewidth": float(marker_params["edge_linewidth"]) * scale,
        "selected_linewidth": float(marker_params["selected_linewidth"]) * scale,
    }
    title_params = merged["title_style"]
    title_font_size = max(
        float(title_params["min_font_size"]),
        min(float(title_params["max_font_size"]), output_width / float(title_params["width_divisor"])),
    )
    panel_title_params = merged["panel_title_style"]
    panel_title_font_size = max(
        float(panel_title_params["min_font_size"]),
        min(float(panel_title_params["max_font_size"]), output_width / float(panel_title_params["width_divisor"])),
    )
    genes = list(merged["genes"])
    tracks = list(merged.get("tracks", ["PR_status", "ER_status", "HER2_status", "Classification"]))
    selected_samples = set(points.loc[points["selected"], "sample"].astype(str))
    ordered_samples = points.sort_values(["selected", "classification", "sample"], ascending=[False, True, True])["sample"].astype(str).tolist()
    axes_config = merged["axes"]
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


RENDERERS: dict[str, Callable[..., None]] = {
    "sv_panel": render_sv_panel,
    "package_mark": render_package_mark,
    "interactive_snapshot": render_interactive_snapshot,
    "comparison_table": render_comparison_table,
    "lasso_scatter": render_lasso_scatter,
    "multimodal_panel": render_multimodal_panel,
}


def render_preset(name: str, out_dir: Optional[Path] = None, style: str = "clean") -> Path:
    if style == "comparison":
        return render_brca_comparison_sheet(name, out_dir or COMPARISON_OUT)
    if style != "clean":
        raise ValueError("style must be one of: clean, comparison.")

    run_config = GALLERY_CONFIG["plot_runs"].get(name)
    if run_config is None:
        raise ValueError(f"Unknown gallery preset {name!r}.")
    if run_config.get("style") != "clean" or run_config["renderer"] not in RENDERERS:
        raise ValueError(f"Preset {name!r} is not a custom gallery preset.")

    out_dir = out_dir or CLEAN_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / run_config["output_name"]
    RENDERERS[run_config["renderer"]](output_path, params=run_config["params"])
    return output_path


# Comparison sheets write compare.goal_plot_01.png and compare.goal_plot_14.png.
def render_brca_comparison_sheet(name: str, out_dir: Optional[Path] = None) -> Path:
    comparison_config = GALLERY_CONFIG.get("comparison_runs", {})
    base_name = name
    if base_name not in comparison_config:
        if base_name == "brca_compact":
            base_name = "brca_compact_complex"
    if base_name not in comparison_config:
        raise ValueError("Comparison sheets are only available for configured BRCA gallery presets.")
    run_config = comparison_config[base_name]
    clean_name = run_config["clean_preset"]
    clean_run = GALLERY_CONFIG["plot_runs"][clean_name]
    out_dir = out_dir or COMPARISON_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / run_config["output_name"]

    with tempfile.TemporaryDirectory(prefix="pyoncoplot_compare_") as temporary_dir:
        work_dir = Path(temporary_dir)
        clean_dir = work_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_path = clean_dir / clean_run["output_name"]
        save = dict(clean_run["params"]["save"])
        save["path"] = clean_path
        result = oncoplot(
            params=CONFIG_PATH,
            params_key=f"gallery_params.plot_runs.{clean_name}.params.oncoplot",
            save=save,
        )
        if result.backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.close(result.figure)
        original_path = GOAL_PLOTS / clean_run["goal_plot"]
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


def _selected_custom_presets(names: Optional[Sequence[str]]) -> list[str]:
    runs = GALLERY_CONFIG["plot_runs"]
    selected = names or [
        name
        for name, run in runs.items()
        if run.get("style") == "clean" and run.get("run", True) and run["renderer"] in RENDERERS
    ]
    custom = []
    for name in selected:
        run = runs.get(name)
        if run is None or run.get("style") != "clean":
            raise ValueError(f"Unknown gallery preset {name!r}.")
        if run["renderer"] not in RENDERERS:
            raise ValueError(f"Preset {name!r} is not a custom gallery preset.")
        custom.append(name)
    return custom


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=["clean", "comparison", "both"], default="clean")
    parser.add_argument("--preset", action="append", help="Preset name to render. May be supplied more than once.")
    parser.add_argument("--out-dir", type=Path, help="Override the output directory for the selected style.")
    args = parser.parse_args(argv)

    if args.style in {"clean", "both"}:
        clean_out = args.out_dir if args.style == "clean" and args.out_dir else CLEAN_OUT
        for name in _selected_custom_presets(args.preset):
            render_preset(name, clean_out, style="clean")
        print(f"Wrote custom gallery images to {clean_out}")

    if args.style in {"comparison", "both"}:
        comparison_out = args.out_dir if args.style == "comparison" and args.out_dir else COMPARISON_OUT
        comparison_names = args.preset or ["brca_large"]
        for name in comparison_names:
            render_preset(name, comparison_out, style="comparison")
        print(f"Wrote comparison sheets to {comparison_out}")


if __name__ == "__main__":
    main()
