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

# Generated here with public oncoplot(params=CONFIG_PATH, params_key=...):
# goal plots 01-05, 13-19, and 21. Custom plots 06-12 and 20 are generated
# by make_other_gallery_plots.py.


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", help="Oncoplot preset name to render. May be supplied more than once.")
    parser.add_argument("--out-dir", type=Path, help="Override the clean output directory.")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or CLEAN_OUT
    runs = GALLERY_CONFIG["plot_runs"]
    selected_names = args.preset or [
        name
        for name, run in runs.items()
        if run.get("style") == "clean" and run.get("run", True) and run["renderer"] == "oncoplot"
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in selected_names:
        run_config = runs.get(name)
        if run_config is None:
            raise ValueError(f"Unknown gallery preset {name!r}.")
        if run_config.get("style") != "clean" or run_config["renderer"] != "oncoplot":
            raise ValueError(f"Preset {name!r} is not an oncoplot gallery preset.")
        save = dict(run_config["params"]["save"])
        save["path"] = out_dir / run_config["output_name"]
        result = oncoplot(
            params=CONFIG_PATH,
            params_key=f"gallery_params.plot_runs.{name}.params.oncoplot",
            save=save,
        )
        if result.backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.close(result.figure)
    print(f"Wrote oncoplot gallery images to {out_dir}")


if __name__ == "__main__":
    main()
