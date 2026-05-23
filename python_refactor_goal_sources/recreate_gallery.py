"""Recreate goal gallery images from deterministic synthetic inputs.

Generated PNGs are written under ``python_refactor_goal_sources/generated_plots``
so numbered source goal plots remain untouched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import yaml

from pyoncoplot import OncoplotOptions, oncoplot
from pyoncoplot._params import merge_params


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
REFERENCE_LIKE_OUT = _resolve_repo_path(OUTPUT_DIRS.get("reference_like", GENERATED_ROOT / "reference_like"))
COMPARISON_OUT = _resolve_repo_path(OUTPUT_DIRS.get("comparison", GENERATED_ROOT / "comparison"))
OUT = CLEAN_OUT

ONCOPLOT_GALLERY_KEYS = {
    "include_genes",
    "metadata_cols",
    "metadata_sort_cols",
    "metadata_sort_by",
    "metadata_sort_desc",
    "oncoplot",
    "options",
    "save",
    "title",
}

BRCA_LARGE_KEYS = {
    "genes",
    "metadata_cols",
    "metadata_legend_cols",
    "sort_columns",
    "figure_size",
    "axes",
    "save",
    "tmb_scale",
    "tmb_ymax_min",
    "mutation_legend_limit",
}

CSSC_KEYS = {
    "samples",
    "genes",
    "figure_size",
    "axes",
    "save",
    "tmb_ymax_min",
    "bar_xlim",
    "bar_xticks",
}

GBM_KEYS = {
    "tracks",
    "genes",
    "figure_size",
    "axes",
    "save",
    "footer_text",
    "legend_entries",
}

SV_KEYS = {
    "samples",
    "figure_size",
    "save",
    "depth_ylim",
    "allele_ylim",
}

README_ONCOPLOT_KEYS = {
    "include_genes",
    "metadata_cols",
    "metadata_sort_cols",
    "metadata_sort_by",
    "metadata_sort_desc",
    "oncoplot",
    "options",
    "save",
    "title",
}

PACKAGE_MARK_KEYS = {"figure_size", "save", "title"}

COMPARISON_TABLE_KEYS = {"figure_size", "save", "title", "subtitle", "compact"}

MULTIMODAL_KEYS = {
    "figure_size",
    "save",
    "variant",
    "genes",
    "tracks",
    "axes",
    "title",
    "show_lasso",
    "selected_only_label",
}


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


@dataclass(frozen=True)
class GalleryVariant:
    name: str
    output_name: str
    renderer: Callable[..., None]
    expected_size: tuple[int, int]
    goal_plot: str
    params: Mapping[str, Any]
    run: bool = True


def _read_tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(INPUTS / name, sep="\t")


def _read_palette(name: str) -> Dict[str, str]:
    return json.loads((INPUTS / name).read_text(encoding="utf-8"))


def _input_file(family: str, key: str) -> str:
    return str(GALLERY_CONFIG["input_files"][family][key])


def _save_exact(result, output_path: Path, dpi: int = 120) -> None:
    result.figure.savefig(output_path, dpi=dpi)


def _save_dpi(params: Mapping[str, Any], default: int = 120) -> int:
    save_params = params.get("save", {})
    if save_params is None:
        return default
    if not isinstance(save_params, Mapping):
        raise TypeError("gallery save params must be a mapping.")
    return int(save_params.get("dpi", default))


def _options_from_params(params: Mapping[str, Any]) -> OncoplotOptions:
    options = params.get("options", {})
    if isinstance(options, OncoplotOptions):
        return options
    if not isinstance(options, Mapping):
        raise TypeError("gallery options must be a mapping or OncoplotOptions instance.")
    return OncoplotOptions(**dict(options))


def _axes_from_params(params: Mapping[str, Any], required: Sequence[str]) -> dict[str, Sequence[float]]:
    axes = params.get("axes", {})
    if not isinstance(axes, Mapping):
        raise TypeError("gallery axes params must be a mapping.")
    missing = [name for name in required if name not in axes]
    if missing:
        raise ValueError(f"Missing configured axes: {', '.join(missing)}.")
    return dict(axes)


def _aml_metadata_palette() -> Mapping[str, Mapping[str, str]]:
    return {
        "FAB_classification": {
            "M0": "#1B9E77",
            "M1": "#D95F02",
            "M2": "#7570B3",
            "M3": "#E7298A",
            "M4": "#66A61E",
            "M5": "#E6AB02",
            "M6": "#A6761D",
            "M7": "#666666",
        },
        "Overall_Survival_Status": {"0": "#FDB7B4", "1": "#BBD7EA"},
        "cluster": {
            "FLT3/NPM1": "#1B9E77",
            "DNMT3A/TET2": "#D95F02",
            "IDH": "#7570B3",
            "TP53/CEBPA": "#E7298A",
        },
    }


def _brca_metadata_palette() -> Mapping[str, Mapping[str, str]]:
    return {
        "Subtype": {
            "HR+/HER2-": "#E41A1C",
            "HR-/HER2+": "#377EB8",
            "HR+/HER2+": "#F781BF",
            "HR-/HER2-": "#A65628",
        },
        "ER_status": {"Positive": "#4DAF4A", "Negative": "#FFFFFF"},
        "PR_status": {"Positive": "#4DAF4A", "Negative": "#FFFFFF"},
        "HER2_status": {"Positive": "#377EB8", "Negative": "#FFFFFF"},
        "Age": {"<=50": "#3366CC", ">50": "#99CCFF"},
        "Menopause": {"Premenopausal": "#D95F02", "Postmenopausal": "#F2C16B"},
        "LN_stage": {"Positive": "#6A00D4", "Negative": "#FFFFFF"},
        "Grade": {"I": "#B7F7C0", "II": "#66C2A5", "III": "#4D8F77", "Unknown": "#CFCFCF"},
        "TNM_stage": {"I": "#B7F7C0", "II": "#7DDB82", "III": "#4DAF4A", "IV": "#1B7837"},
        "Histological_type": {
            "Infiltrating Ductal Carcinoma": "#7B61B4",
            "Infiltrating Lobular Carcinoma": "#D6B3E6",
            "Others": "#B8DE6F",
        },
        "Classification": {
            "Ambiguous": "#D9D9D9",
            "Not Triple Negative": "#FF1A1A",
            "Triple Negative": "#000000",
        },
    }


def _load_aml():
    mutations = _read_tsv(_input_file("aml", "mutations"))
    metadata = _read_tsv(_input_file("aml", "metadata"))
    metadata["FAB_classification"] = metadata["FAB_classification"].astype(str)
    metadata["Overall_Survival_Status"] = metadata["Overall_Survival_Status"].astype(str)
    tmb = _read_tsv(_input_file("aml", "tmb"))
    palette = _read_palette(_input_file("aml", "palette"))
    return mutations, metadata, tmb, palette


def _load_brca():
    mutations = _read_tsv(_input_file("brca", "mutations"))
    metadata = _read_tsv(_input_file("brca", "metadata"))
    for column in metadata.columns:
        if column not in {"sample", "Age_years"}:
            metadata[column] = metadata[column].astype(str)
    tmb = _read_tsv(_input_file("brca", "tmb"))
    palette = _read_palette(_input_file("brca", "palette"))
    return mutations, metadata, tmb, palette


def _load_cssc():
    return (
        _read_tsv(_input_file("cssc", "mutations")),
        _read_tsv(_input_file("cssc", "tmb")),
        _read_palette(_input_file("cssc", "palette")),
    )


def _load_gbm():
    palette = _read_palette(_input_file("gbm", "palette"))
    return _read_tsv(_input_file("gbm", "tracks")), _read_tsv(_input_file("gbm", "events")), palette


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


def _readme_metadata_palette() -> Mapping[str, Mapping[str, str]]:
    return {
        "Subtype": {"Luminal": "#4DAF4A", "Basal": "#E41A1C", "HER2": "#377EB8", "Normal-like": "#984EA3"},
        "ER_status": {"Positive": "#4DAF4A", "Negative": "#FFFFFF"},
        "PR_status": {"Positive": "#4DAF4A", "Negative": "#FFFFFF"},
        "HER2_status": {"Positive": "#377EB8", "Negative": "#FFFFFF"},
    }


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


def _draw_alteration_glyph(ax, x: float, y: float, alteration: str, palette: Mapping[str, str], linewidth: float = 1.8) -> None:
    from matplotlib.patches import Rectangle

    color = palette.get(alteration, "#111111")
    if alteration == "missense_variant":
        ax.add_patch(Rectangle((x + 0.32, y + 0.08), 0.32, 0.84, facecolor=color, edgecolor="none"))
    elif alteration == "synonymous_variant":
        ax.add_patch(Rectangle((x + 0.08, y + 0.47), 0.84, 0.11, facecolor=color, edgecolor="none"))
    elif alteration == "stop_gained":
        ax.add_patch(Rectangle((x + 0.12, y + 0.22), 0.76, 0.22, facecolor=color, edgecolor="none"))
    elif alteration == "complex_substitution":
        ax.plot([x + 0.08, x + 0.92], [y + 0.86, y + 0.08], color=color, linewidth=linewidth, solid_capstyle="butt")
    elif alteration == "splice_site_variant":
        ax.add_patch(Rectangle((x + 0.36, y + 0.12), 0.28, 0.28, facecolor=color, edgecolor="none"))
    elif alteration == "frameshift_truncation":
        ax.add_patch(Rectangle((x + 0.22, y + 0.18), 0.56, 0.60, facecolor=color, edgecolor="none"))
    elif alteration == "inframe_deletion":
        ax.plot([x + 0.08, x + 0.88], [y + 0.88, y + 0.12], color=color, linewidth=linewidth, solid_capstyle="butt")
    else:
        ax.add_patch(Rectangle((x + 0.18, y + 0.18), 0.64, 0.64, facecolor=color, edgecolor="none"))


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


def _manual_legend(ax, title: str, entries: Sequence[tuple[str, str]], x: float, y: float, step: float = 0.04, fontsize: float = 8) -> float:
    from matplotlib.patches import Rectangle

    ax.text(x, y, title, fontsize=fontsize + 1, weight="bold", ha="left", va="top")
    y -= step * 0.9
    for label, color in entries:
        ax.add_patch(Rectangle((x, y - step * 0.42), 0.035, step * 0.6, facecolor=color, edgecolor="#333333", linewidth=0.4))
        ax.text(x + 0.045, y - step * 0.1, label, fontsize=fontsize, ha="left", va="center")
        y -= step
    return y - step * 0.45


def render_aml_basic(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    merged = merge_params(params, allowed_keys=ONCOPLOT_GALLERY_KEYS, context="aml_basic gallery", **kwargs)
    mutations, _metadata, tmb, palette = _load_aml()
    oncoplot_params = {
        **dict(merged.get("oncoplot", {})),
        "data": mutations,
        "include_genes": merged["include_genes"],
        "palette": palette,
        "tmb_data": tmb,
        "tmb_palette": palette,
        "options": _options_from_params(merged),
    }
    result = oncoplot(params=oncoplot_params)
    _save_exact(result, output_path, dpi=_save_dpi(merged, default=120))


def render_aml_metadata(
    output_path: Path,
    sort_metadata: bool = False,
    *,
    params: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> None:
    extra_kwargs = dict(kwargs)
    if sort_metadata:
        extra_kwargs.setdefault("metadata_sort_cols", ["FAB_classification"])
    merged = merge_params(params, allowed_keys=ONCOPLOT_GALLERY_KEYS, context="aml metadata gallery", **extra_kwargs)
    mutations, metadata, tmb, palette = _load_aml()
    oncoplot_params = {
        **dict(merged.get("oncoplot", {})),
        "data": mutations,
        "include_genes": merged["include_genes"],
        "palette": palette,
        "tmb_data": tmb,
        "tmb_palette": palette,
        "metadata": metadata,
        "metadata_cols": merged["metadata_cols"],
        "metadata_palette": _aml_metadata_palette(),
        "metadata_sort_cols": merged.get("metadata_sort_cols"),
        "metadata_sort_by": merged.get("metadata_sort_by", "alphabetical"),
        "metadata_sort_desc": merged.get("metadata_sort_desc", False),
        "options": _options_from_params(merged),
    }
    result = oncoplot(params=oncoplot_params)
    result.figure.suptitle(str(merged.get("title", output_path.stem + ".py")), fontweight="bold", y=0.98)
    _save_exact(result, output_path, dpi=_save_dpi(merged, default=120))


def render_brca_large(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    render_brca_large_reference_like(output_path, params=params, **kwargs)


def render_brca_large_reference_like(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = merge_params(params, allowed_keys=BRCA_LARGE_KEYS, context="brca_large gallery", **kwargs)
    mutations, metadata, tmb, palette = _load_brca()
    metadata_palette = _brca_metadata_palette()
    metadata = metadata.sort_values(list(merged["sort_columns"]), kind="stable")
    samples = metadata["sample"].astype(str).tolist()
    genes = list(merged["genes"])
    grouped = _group_events(
        mutations[mutations["gene"].isin(genes)].rename(columns={"mutation_type": "alteration"}),
        type_col="alteration",
    )
    axes_config = _axes_from_params(merged, ["tmb", "main", "gene", "metadata", "legend"])
    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax_tmb = fig.add_axes(axes_config["tmb"])
    ax_main = fig.add_axes(axes_config["main"])
    ax_gene = fig.add_axes(axes_config["gene"])
    ax_meta = fig.add_axes(axes_config["metadata"])
    ax_legend = fig.add_axes(axes_config["legend"])
    ax_legend.axis("off")

    x_positions = np.arange(len(samples))
    bottoms = np.zeros(len(samples))
    type_col = "mutation_type"
    tmb_scale = float(merged.get("tmb_scale", 1.0))
    for mutation_type, group in tmb.groupby(type_col, dropna=False, sort=False):
        values = (
            group.groupby("sample", observed=False)["mutations"]
            .sum()
            .reindex(samples, fill_value=0)
            .to_numpy(dtype=float)
            * tmb_scale
        )
        color = "#4D4D4D" if pd.isna(mutation_type) else palette.get(str(mutation_type), "#4D4D4D")
        ax_tmb.bar(x_positions, values, bottom=bottoms, width=1.0, color=color, linewidth=0)
        bottoms += values
    ymax = max(float(bottoms.max()) * 1.05, float(merged.get("tmb_ymax_min", 100)))
    ax_tmb.set_xlim(-0.5, len(samples) - 0.5)
    ax_tmb.set_ylim(0, ymax)
    ax_tmb.set_ylabel("No. of\nMutations", fontsize=24)
    ax_tmb.set_xticks([])
    ax_tmb.tick_params(axis="y", labelsize=16)
    ax_tmb.spines[["top", "right"]].set_visible(False)

    for y, gene in enumerate(genes):
        for x, sample in enumerate(samples):
            ax_main.add_patch(Rectangle((x, y), 1, 1, facecolor="#D7D7D7", edgecolor="white", linewidth=0.10))
            _draw_split_tile(ax_main, x, y, grouped.get((sample, gene), []), palette)
    ax_main.set_xlim(0, len(samples))
    ax_main.set_ylim(0, len(genes))
    ax_main.invert_yaxis()
    ax_main.set_xticks([])
    ax_main.set_yticks(np.arange(len(genes)) + 0.5)
    ax_main.set_yticklabels(genes, fontsize=30, fontstyle="italic")
    ax_main.tick_params(axis="y", length=0)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    count_by_gene_type = (
        mutations[mutations["gene"].isin(genes)]
        .drop_duplicates(["sample", "gene", "mutation_type"])
        .groupby(["gene", "mutation_type"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    left = np.zeros(len(genes))
    max_total = 1
    for mutation_type, group in count_by_gene_type.groupby("mutation_type", sort=False):
        counts = group.set_index("gene")["count"].reindex(genes, fill_value=0).to_numpy(dtype=float)
        ax_gene.barh(np.arange(len(genes)) + 0.5, counts, left=left, height=0.70, color=palette.get(str(mutation_type), "#1A1A1A"))
        left += counts
        max_total = max(max_total, int(left.max()))
    for y, total in enumerate(left):
        ax_gene.text(total + max_total * 0.04, y + 0.5, f"{total / max(len(samples), 1) * 100:.0f}%", va="center", fontsize=19)
    ax_gene.set_xlim(0, max_total * 1.35)
    ax_gene.set_ylim(0, len(genes))
    ax_gene.invert_yaxis()
    ax_gene.set_yticks([])
    ax_gene.xaxis.set_ticks_position("top")
    ax_gene.tick_params(axis="x", labelsize=14)
    ax_gene.spines[["left", "right", "bottom"]].set_visible(False)

    meta_cols = list(merged["metadata_cols"])
    metadata_indexed = metadata.set_index("sample")
    ax_meta.set_xlim(0, len(samples))
    ax_meta.set_ylim(0, len(meta_cols))
    ax_meta.invert_yaxis()
    for y, column in enumerate(meta_cols):
        values = metadata_indexed[column].reindex(samples)
        if pd.api.types.is_numeric_dtype(values):
            min_value = float(values.min())
            max_value = float(values.max())
            span = max(max_value - min_value, 1e-9)
            ax_meta.add_patch(Rectangle((0, y), len(samples), 1, facecolor="#F4F4F4", edgecolor="white", linewidth=0))
            for x, value in enumerate(values):
                frac = (float(value) - min_value) / span
                ax_meta.add_patch(Rectangle((x, y + 1 - frac), 1, frac, facecolor="#7F7F7F", edgecolor="white", linewidth=0.04))
            continue
        column_palette = metadata_palette.get(column, {})
        for x, value in enumerate(values.astype(str)):
            ax_meta.add_patch(Rectangle((x, y), 1, 1, facecolor=column_palette.get(value, "#D9D9D9"), edgecolor="white", linewidth=0.04))
    ax_meta.set_yticks(np.arange(len(meta_cols)) + 0.5)
    ax_meta.set_yticklabels(meta_cols, fontsize=20)
    ax_meta.set_xticks([])
    ax_meta.tick_params(axis="y", length=0)
    for spine in ax_meta.spines.values():
        spine.set_visible(False)

    used_mutation_types = [name for name in palette if name in set(mutations["mutation_type"].astype(str))]
    mutation_limit = int(merged.get("mutation_legend_limit", 9))
    y_cursor = _manual_legend(ax_legend, "Mutation", [(name, palette[name]) for name in used_mutation_types[:mutation_limit]], 0.0, 0.98, step=0.032, fontsize=9)
    for column in merged["metadata_legend_cols"]:
        entries = list(metadata_palette[column].items())
        y_cursor = _manual_legend(ax_legend, column, entries, 0.0, y_cursor, step=0.033, fontsize=8)
    _manual_legend(ax_legend, "Age_years", [("28-90", "#7F7F7F")], 0.0, y_cursor, step=0.033, fontsize=8)
    fig.savefig(output_path, dpi=_save_dpi(merged, default=120))
    plt.close(fig)


def render_brca_compact_complex(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    merged = merge_params(params, allowed_keys=ONCOPLOT_GALLERY_KEYS, context="brca compact gallery", **kwargs)
    mutations, metadata, tmb, palette = _load_brca()
    oncoplot_params = {
        **dict(merged.get("oncoplot", {})),
        "data": mutations,
        "include_genes": merged["include_genes"],
        "palette": palette,
        "tmb_data": tmb,
        "tmb_palette": palette,
        "metadata": metadata,
        "metadata_cols": merged["metadata_cols"],
        "metadata_palette": _brca_metadata_palette(),
        "metadata_sort_cols": merged.get("metadata_sort_cols"),
        "metadata_sort_by": merged.get("metadata_sort_by"),
        "metadata_sort_desc": merged.get("metadata_sort_desc"),
        "options": _options_from_params(merged),
    }
    result = oncoplot(params=oncoplot_params)
    _save_exact(result, output_path, dpi=_save_dpi(merged, default=100))


def render_brca_compact_reference_like(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    render_brca_compact_complex(output_path, params=params, **kwargs)


def render_cssc_compact(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = merge_params(params, allowed_keys=CSSC_KEYS, context="cssc gallery", **kwargs)
    mutations, tmb, palette = _load_cssc()
    samples = list(merged["samples"])
    genes = list(merged["genes"])
    grouped = _group_events(mutations)
    axes_config = _axes_from_params(merged, ["tmb", "main", "bar", "legend"])
    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax_tmb = fig.add_axes(axes_config["tmb"])
    ax_main = fig.add_axes(axes_config["main"])
    ax_bar = fig.add_axes(axes_config["bar"])
    ax_legend = fig.add_axes(axes_config["legend"])
    ax_legend.axis("off")

    bottoms = np.zeros(len(samples))
    for alteration, group in tmb.groupby("alteration", sort=False):
        values = (
            group.groupby("sample", observed=False)["mutations"]
            .sum()
            .reindex(samples, fill_value=0)
            .to_numpy(dtype=float)
        )
        ax_tmb.bar(np.arange(len(samples)), values, bottom=bottoms, color=palette.get(str(alteration), "#111111"), width=0.62, linewidth=0)
        bottoms += values
    ax_tmb.set_xlim(-0.5, len(samples) - 0.5)
    ax_tmb.set_ylim(0, max(float(merged.get("tmb_ymax_min", 15)), float(bottoms.max()) * 1.15))
    ax_tmb.set_xticks([])
    ax_tmb.tick_params(axis="y", labelsize=8)
    ax_tmb.spines[["top", "right"]].set_visible(False)

    for y, gene in enumerate(genes):
        gene_samples = mutations.loc[mutations["gene"] == gene, "sample"].nunique()
        ax_main.text(-0.14, y + 0.5, f"{gene_samples / len(samples) * 100:.0f}%", ha="right", va="center", fontsize=22)
        for x, sample in enumerate(samples):
            ax_main.add_patch(Rectangle((x, y), 1, 1, facecolor="#C9C9C9", edgecolor="white", linewidth=0.55))
            for alteration in grouped.get((sample, gene), []):
                _draw_alteration_glyph(ax_main, x, y, alteration, palette, linewidth=1.6)
    ax_main.set_xlim(0, len(samples))
    ax_main.set_ylim(0, len(genes))
    ax_main.invert_yaxis()
    ax_main.set_xticks(np.arange(len(samples)) + 0.5)
    ax_main.set_xticklabels(samples, rotation=90, fontsize=20, ha="right", va="center", rotation_mode="anchor")
    ax_main.set_yticks(np.arange(len(genes)) + 0.5)
    ax_main.set_yticklabels(genes, fontsize=22, fontstyle="italic")
    ax_main.yaxis.tick_right()
    ax_main.tick_params(axis="x", length=0)
    ax_main.tick_params(axis="y", length=0, pad=8)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    y_positions = np.arange(len(genes)) + 0.5
    left = np.zeros(len(genes))
    for alteration, group in mutations.drop_duplicates(["sample", "gene", "alteration"]).groupby("alteration", sort=False):
        counts = group.groupby("gene", observed=False).size().reindex(genes, fill_value=0).to_numpy(dtype=float)
        ax_bar.barh(y_positions, counts, left=left, color=palette.get(str(alteration), "#111111"), height=0.65)
        left += counts
    ax_bar.set_xlim(*merged.get("bar_xlim", [0, 30]))
    ax_bar.set_ylim(0, len(genes))
    ax_bar.invert_yaxis()
    ax_bar.xaxis.set_ticks_position("top")
    ax_bar.set_xticks(list(merged.get("bar_xticks", [0, 10, 20, 30])))
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.set_yticks([])
    ax_bar.spines[["left", "right", "bottom"]].set_visible(False)

    ax_legend.text(0.0, 0.98, "Alterations", fontsize=10, weight="bold", va="top")
    y_cursor = 0.84
    for label, color in palette.items():
        ax_legend.add_patch(Rectangle((0.0, y_cursor - 0.027), 0.055, 0.054, facecolor=color, edgecolor="#333333", linewidth=0.4))
        ax_legend.text(0.075, y_cursor, label, fontsize=8, va="center")
        y_cursor -= 0.095
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    fig.savefig(output_path, dpi=_save_dpi(merged, default=100))
    plt.close(fig)


def render_gbm_clinical_molecular(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = merge_params(params, allowed_keys=GBM_KEYS, context="gbm gallery", **kwargs)
    tracks, events, palette = _load_gbm()
    track_palette = palette["tracks"]
    alteration_palette = palette["alterations"]
    samples = tracks["sample"].astype(str).tolist()
    grouped = _group_events(events)
    tracks_indexed = tracks.set_index("sample")
    track_rows = list(merged["tracks"])
    genes = list(merged["genes"])
    axes_config = _axes_from_params(merged, ["tracks", "main", "legend"])

    fig = plt.figure(figsize=tuple(merged["figure_size"]))
    ax_tracks = fig.add_axes(axes_config["tracks"])
    ax_main = fig.add_axes(axes_config["main"])
    ax_legend = fig.add_axes(axes_config["legend"])
    ax_legend.axis("off")

    for y, track in enumerate(track_rows):
        for x, sample in enumerate(samples):
            value = str(tracks_indexed.loc[sample, track])
            ax_tracks.add_patch(Rectangle((x, y), 1, 1, facecolor=track_palette.get(value, "#FFFFFF"), edgecolor="#111111", linewidth=0.18))
    ax_tracks.set_xlim(0, len(samples))
    ax_tracks.set_ylim(0, len(track_rows))
    ax_tracks.invert_yaxis()
    ax_tracks.set_xticks([])
    ax_tracks.set_yticks(np.arange(len(track_rows)) + 0.5)
    ax_tracks.set_yticklabels(track_rows, fontsize=7)
    ax_tracks.tick_params(axis="y", length=0)

    for y, gene in enumerate(genes):
        for x, sample in enumerate(samples):
            ax_main.add_patch(Rectangle((x, y), 1, 1, facecolor="#FFFFFF", edgecolor="#1A1A1A", linewidth=0.18))
            _draw_split_tile(ax_main, x, y, grouped.get((sample, gene), []), alteration_palette, default_color="#1E355C")
    ax_main.set_xlim(0, len(samples))
    ax_main.set_ylim(0, len(genes))
    ax_main.invert_yaxis()
    ax_main.set_xticks([])
    ax_main.set_yticks(np.arange(len(genes)) + 0.5)
    ax_main.set_yticklabels(genes, fontsize=7)
    ax_main.tick_params(axis="y", length=0)
    ax_main.text(len(samples) - 1, len(genes) + 1.0, str(merged["footer_text"]), fontsize=7, ha="right")

    legend_entries = [(label, track_palette[label]) for label in merged["legend_entries"]]
    x_cursor = 0.00
    y_rows = [0.68, 0.20]
    for index, (label, color) in enumerate(legend_entries):
        row = 0 if index < 4 else 1
        col_index = index if index < 4 else index - 4
        x_cursor = 0.01 + col_index * 0.24
        ax_legend.add_patch(Rectangle((x_cursor, y_rows[row] - 0.12), 0.018, 0.22, facecolor=color, edgecolor="#333333", linewidth=0.35))
        ax_legend.text(x_cursor + 0.024, y_rows[row], label, fontsize=7, va="center")
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    fig.savefig(output_path, dpi=_save_dpi(merged, default=100))
    plt.close(fig)


def render_sv_panel(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = merge_params(params, allowed_keys=SV_KEYS, context="sv gallery", **kwargs)
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
        for _, model in gene_models.iterrows():
            axes[0, col].add_patch(Rectangle((model["start"], -7), model["end"] - model["start"], 4, color="black"))
            axes[0, col].text((model["start"] + model["end"]) / 2, -10, model["gene"], ha="center", va="top", fontsize=9, style="italic")

        for allele_name, color in [("REF", "#4C72B0"), ("ALT", "#DD8452")]:
            group = sample_allele[sample_allele["allele"] == allele_name]
            axes[1, col].scatter(group["position"], group["allele_fraction"], s=14, label=allele_name, color=color)
        axes[1, col].set_ylim(*allele_ylim)
        axes[1, col].grid(True, color="white")
        axes[1, col].set_xlabel("Position", fontsize=14)
        for _, model in gene_models.iterrows():
            axes[1, col].add_patch(Rectangle((model["start"], -0.08), model["end"] - model["start"], 0.04, color="black"))
            axes[1, col].text((model["start"] + model["end"]) / 2, -0.10, model["gene"], ha="center", va="top", fontsize=9, style="italic")
    axes[0, 0].set_ylabel("Depth", fontsize=14)
    axes[1, 0].set_ylabel("Allele fraction", fontsize=14)
    axes[1, 0].legend(loc="upper left", fontsize=16, frameon=True)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#EAEAF2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=_save_dpi(merged, default=100))
    plt.close(fig)


def render_readme_oncoplot(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    merged = merge_params(params, allowed_keys=README_ONCOPLOT_KEYS, context="readme oncoplot gallery", **kwargs)
    mutations, metadata, tmb, palette = _load_readme()
    render_options = _options_from_params(merged)
    oncoplot_params = {
        **dict(merged.get("oncoplot", {})),
        "data": mutations,
        "include_genes": merged["include_genes"],
        "palette": palette,
        "tmb_data": tmb,
        "tmb_palette": palette,
        "metadata": metadata,
        "metadata_cols": merged.get("metadata_cols"),
        "metadata_palette": _readme_metadata_palette(),
        "metadata_sort_cols": merged.get("metadata_sort_cols"),
        "metadata_sort_by": merged.get("metadata_sort_by", "alphabetical"),
        "metadata_sort_desc": merged.get("metadata_sort_desc", False),
        "options": render_options,
    }
    result = oncoplot(params=oncoplot_params)
    if merged.get("title"):
        result.figure.suptitle(str(merged["title"]), y=0.985, fontweight="bold")
    dpi = _save_dpi(merged, default=120)
    result.figure.set_size_inches(render_options.width / dpi, render_options.height / dpi, forward=True)
    _save_exact(result, output_path, dpi=dpi)


def render_package_mark(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    merged = merge_params(params, allowed_keys=PACKAGE_MARK_KEYS, context="package mark gallery", **kwargs)
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
    ax.text(0.50, 0.075, str(merged.get("title", "ggoncoplot")), ha="center", va="center", fontsize=9, weight="bold")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_interactive_snapshot(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    merged = merge_params(params, allowed_keys=MULTIMODAL_KEYS, context="interactive snapshot gallery", **kwargs)
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
        ax_legend.text(0.11, y_cursor - 0.015, label, fontsize=8, va="center")
        y_cursor -= 0.09
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_comparison_table(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    merged = merge_params(params, allowed_keys=COMPARISON_TABLE_KEYS, context="comparison table gallery", **kwargs)
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


def render_lasso_scatter(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    merged = merge_params(params, allowed_keys=MULTIMODAL_KEYS, context="lasso scatter gallery", **kwargs)
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
    ax.legend(loc="upper right", frameon=True, fontsize=10)
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


def render_multimodal_panel(output_path: Path, *, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    merged = merge_params(params, allowed_keys=MULTIMODAL_KEYS, context="multimodal panel gallery", **kwargs)
    _samples, points, events, clinical, selection, palette = _load_multimodal()
    dpi = _save_dpi(merged, default=100)
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
            ax.scatter(group[f"{prefix}_x"], group[f"{prefix}_y"], s=12, color=palette.get(str(classification), "#999999"), alpha=0.72, edgecolor="none")
        selected = points[points["selected"]]
        ax.scatter(selected[f"{prefix}_x"], selected[f"{prefix}_y"], s=30, facecolor="none", edgecolor=palette["Selected"], linewidth=1.0)
        if prefix == "tsne" and merged.get("show_lasso", False):
            ax.add_patch(Polygon(selection[["x", "y"]].to_numpy(), closed=True, fill=False, edgecolor=palette["Selected"], linewidth=1.8, linestyle="--"))
        ax.set_title(title, fontsize=9, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, color="#EFEFEF", linewidth=0.4)

    _draw_multimodal_oncoplot(ax_main, events, ordered_samples, genes, palette, selected_samples)
    ax_main.set_title("Linked oncoplot", fontsize=9, weight="bold")

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
        ax_legend.text(0.09, y_cursor - 0.015, label, fontsize=7, va="center")
        y_cursor -= 0.10
    fig.suptitle(str(merged.get("title", "Multimodal sample selection")), fontsize=12, weight="bold", y=0.985)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


RENDERERS: Dict[str, Callable[..., None]] = {
    "aml_basic": render_aml_basic,
    "aml_metadata": render_aml_metadata,
    "brca_large": render_brca_large,
    "brca_large_reference_like": render_brca_large_reference_like,
    "brca_compact_complex": render_brca_compact_complex,
    "brca_compact_reference_like": render_brca_compact_reference_like,
    "cssc_compact": render_cssc_compact,
    "gbm_clinical_molecular": render_gbm_clinical_molecular,
    "sv_panel": render_sv_panel,
    "readme_oncoplot": render_readme_oncoplot,
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


def _build_reference_like_presets() -> Dict[str, GalleryVariant]:
    presets: Dict[str, GalleryVariant] = {}
    for name, run_config in CONFIGURED_RUNS.items():
        if run_config.get("style") != "reference_like":
            continue
        presets[name] = GalleryVariant(
            name=name,
            output_name=run_config["output_name"],
            renderer=_renderer_from_config(name, run_config),
            expected_size=tuple(run_config["expected_size"]),
            goal_plot=run_config["goal_plot"],
            params=run_config.get("params", {}),
            run=bool(run_config.get("run", False)),
        )
    return presets


GALLERY_PRESETS = _build_gallery_presets()
REFERENCE_LIKE_PRESETS = _build_reference_like_presets()


def _resolve_reference_like_name(name: str) -> str:
    if name in REFERENCE_LIKE_PRESETS:
        return name
    candidate = f"{name}_reference_like"
    if candidate in REFERENCE_LIKE_PRESETS:
        return candidate
    available = ", ".join(sorted(REFERENCE_LIKE_PRESETS))
    raise ValueError(f"Unknown reference-like gallery preset {name!r}. Available presets: {available}")


def _render_variant(variant: GalleryVariant, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / variant.output_name
    variant.renderer(output_path, params=variant.params)
    return output_path


def render_preset(name: str, out_dir: Optional[Path] = None, style: str = "clean") -> Path:
    if style == "reference_like":
        variant = REFERENCE_LIKE_PRESETS[_resolve_reference_like_name(name)]
        return _render_variant(variant, out_dir or REFERENCE_LIKE_OUT)
    if style == "comparison":
        return render_brca_comparison_sheet(name, out_dir or COMPARISON_OUT)
    if style != "clean":
        raise ValueError("style must be one of: clean, reference_like, comparison.")

    if name not in GALLERY_PRESETS:
        available = ", ".join(sorted(GALLERY_PRESETS))
        raise ValueError(f"Unknown gallery preset {name!r}. Available presets: {available}")
    out_dir = out_dir or CLEAN_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = GALLERY_PRESETS[name]
    output_path = out_dir / preset.output_name
    preset.renderer(output_path, params=preset.params)
    return output_path


def render_brca_comparison_sheet(name: str, out_dir: Optional[Path] = None) -> Path:
    comparison_config = GALLERY_CONFIG.get("comparison_runs", {})
    base_name = name.replace("_reference_like", "")
    if base_name not in comparison_config:
        if base_name == "brca_compact":
            base_name = "brca_compact_complex"
        elif base_name == "brca_large_reference_like":
            base_name = "brca_large"
    if base_name not in comparison_config:
        raise ValueError("Comparison sheets are only available for configured BRCA gallery presets.")
    run_config = _deep_merge(GALLERY_CONFIG.get("default_params", {}), comparison_config[base_name])
    clean_name = run_config["clean_preset"]
    reference_name = run_config["reference_like_preset"]
    clean_preset = GALLERY_PRESETS[clean_name]
    reference_variant = REFERENCE_LIKE_PRESETS[reference_name]
    out_dir = out_dir or COMPARISON_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_sources"
    clean_path = render_preset(clean_name, work_dir / "clean", style="clean")
    reference_like_path = render_preset(reference_name, work_dir / "reference_like", style="reference_like")
    original_path = GOAL_PLOTS / clean_preset.goal_plot
    output_path = out_dir / run_config["output_name"]

    labels = ["Original reference", "Clean generated", "Reference-like generated"]
    paths = [original_path, clean_path, reference_like_path]
    thumb_width = 620
    thumb_height = 360
    label_height = 38
    canvas = Image.new("RGB", (thumb_width * 3, thumb_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(zip(labels, paths)):
        if path.exists():
            image = Image.open(path).convert("RGB")
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
        elif name.endswith("_reference_like"):
            selected.append(name.removesuffix("_reference_like"))
        else:
            raise ValueError(f"Unknown gallery preset {name!r}.")
    return selected


def _selected_reference_presets(names: Optional[Sequence[str]]) -> list[str]:
    if not names:
        return [name for name, preset in REFERENCE_LIKE_PRESETS.items() if preset.run]
    selected = []
    for name in names:
        if name in GALLERY_PRESETS and GALLERY_PRESETS[name].clean_only:
            continue
        selected.append(_resolve_reference_like_name(name))
    return selected


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=["clean", "reference_like", "comparison", "both"], default="clean")
    parser.add_argument("--preset", action="append", help="Preset name to render. May be supplied more than once.")
    parser.add_argument("--out-dir", type=Path, help="Override the output directory for the selected style.")
    args = parser.parse_args(argv)

    if args.style in {"clean", "both"}:
        clean_out = args.out_dir if args.style == "clean" and args.out_dir else CLEAN_OUT
        for name in _selected_clean_presets(args.preset):
            render_preset(name, clean_out, style="clean")
        print(f"Wrote clean gallery images to {clean_out}")

    if args.style in {"reference_like", "both"}:
        reference_out = args.out_dir if args.style == "reference_like" and args.out_dir else REFERENCE_LIKE_OUT
        for name in _selected_reference_presets(args.preset):
            render_preset(name, reference_out, style="reference_like")
        print(f"Wrote reference-like gallery images to {reference_out}")

    if args.style in {"comparison", "both"}:
        comparison_out = args.out_dir if args.style == "comparison" and args.out_dir else COMPARISON_OUT
        comparison_names = args.preset or ["brca_large"]
        for name in comparison_names:
            render_preset(name, comparison_out, style="comparison")
        print(f"Wrote comparison sheets to {comparison_out}")


if __name__ == "__main__":
    main()
