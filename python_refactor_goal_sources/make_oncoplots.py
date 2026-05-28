"""Render gallery presets backed directly by the public ``oncoplot()`` API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

from pyoncoplot import oncoplot


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


def merged_run_config(name: str) -> dict[str, Any]:
    runs = GALLERY_CONFIG["plot_runs"]
    if name not in runs:
        available = ", ".join(sorted(run_name for run_name, run in runs.items() if run.get("style") == "clean"))
        raise ValueError(f"Unknown gallery preset {name!r}. Available presets: {available}")

    run_config = _deep_merge(GALLERY_CONFIG.get("default_params", {}), runs[name])
    params = _deep_merge(GALLERY_CONFIG.get("default_params", {}).get("params", {}), run_config.get("params", {}))
    if "save" in run_config:
        params = _deep_merge({"save": run_config["save"]}, params)
    if "backend" in run_config and isinstance(params.get("oncoplot"), Mapping):
        params["oncoplot"] = _deep_merge({"backend": run_config["backend"]}, params["oncoplot"])
    run_config["params"] = params
    return run_config


def clean_run_names(names: Optional[Sequence[str]] = None, *, renderer: Optional[str] = None) -> list[str]:
    runs = GALLERY_CONFIG["plot_runs"]
    if names:
        selected = []
        for name in names:
            run = runs.get(name)
            if run is None or run.get("style") != "clean":
                raise ValueError(f"Unknown gallery preset {name!r}.")
            if renderer is not None and run["renderer"] != renderer:
                raise ValueError(f"Preset {name!r} uses renderer {run['renderer']!r}, not {renderer!r}.")
            selected.append(name)
        return selected

    return [
        name
        for name, run in runs.items()
        if run.get("style") == "clean" and run.get("run", True) and (renderer is None or run["renderer"] == renderer)
    ]


def render_preset(name: str, out_dir: Optional[Path] = None) -> Path:
    run_config = merged_run_config(name)
    if run_config.get("style") != "clean" or run_config["renderer"] != "oncoplot":
        raise ValueError(f"Preset {name!r} is not an oncoplot gallery preset.")

    out_dir = out_dir or CLEAN_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / run_config["output_name"]
    result = oncoplot(params=CONFIG_PATH, params_key=f"gallery_params.plot_runs.{name}.params.oncoplot")
    dpi = int(run_config["params"]["save"]["dpi"])
    if result.backend == "matplotlib":
        import matplotlib.pyplot as plt

        options = run_config["params"]["oncoplot"]["options"]
        result.figure.set_size_inches(float(options["width"]) / dpi, float(options["height"]) / dpi, forward=True)
        result.save(output_path, dpi=dpi, bbox_inches=None)
        plt.close(result.figure)
    else:
        result.save(output_path, dpi=dpi)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", help="Oncoplot preset name to render. May be supplied more than once.")
    parser.add_argument("--out-dir", type=Path, help="Override the clean output directory.")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or CLEAN_OUT
    for name in clean_run_names(args.preset, renderer="oncoplot"):
        render_preset(name, out_dir)
    print(f"Wrote oncoplot gallery images to {out_dir}")


if __name__ == "__main__":
    main()
