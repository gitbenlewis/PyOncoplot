"""Recreate the full gallery by dispatching to the split renderer scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from pyoncoplot import oncoplot

try:
    from python_refactor_goal_sources.make_oncoplots import (
        CLEAN_OUT,
        COMPARISON_OUT,
        CONFIG_PATH,
        GALLERY_CONFIG,
        GENERATED_ROOT,
        GOAL_PLOTS,
        INPUTS,
    )
    from python_refactor_goal_sources.make_other_gallery_plots import render_preset as render_other_preset
except ModuleNotFoundError:  # pragma: no cover - supports ``python path/to/script.py``.
    from make_oncoplots import (  # type: ignore
        CLEAN_OUT,
        COMPARISON_OUT,
        CONFIG_PATH,
        GALLERY_CONFIG,
        GENERATED_ROOT,
        GOAL_PLOTS,
        INPUTS,
    )
    from make_other_gallery_plots import render_preset as render_other_preset  # type: ignore


def render_preset(name: str, out_dir: Optional[Path] = None, style: str = "clean") -> Path:
    if style == "comparison":
        return render_other_preset(name, out_dir or COMPARISON_OUT, style="comparison")
    if style != "clean":
        raise ValueError("style must be one of: clean, comparison.")

    run_config = GALLERY_CONFIG["plot_runs"].get(name)
    if run_config is None:
        raise ValueError(f"Unknown gallery preset {name!r}.")
    out_dir = out_dir or CLEAN_OUT
    if run_config["renderer"] == "oncoplot":
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / run_config["output_name"]
        save = dict(run_config["params"]["save"])
        save["path"] = output_path
        result = oncoplot(
            params=CONFIG_PATH,
            params_key=f"gallery_params.plot_runs.{name}.params.oncoplot",
            save=save,
        )
        if result.backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.close(result.figure)
        return output_path
    return render_other_preset(name, out_dir, style="clean")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=["clean", "comparison", "both"], default="clean")
    parser.add_argument("--preset", action="append", help="Preset name to render. May be supplied more than once.")
    parser.add_argument("--out-dir", type=Path, help="Override the output directory for the selected style.")
    args = parser.parse_args(argv)

    if args.style in {"clean", "both"}:
        clean_out = args.out_dir if args.style == "clean" and args.out_dir else CLEAN_OUT
        runs = GALLERY_CONFIG["plot_runs"]
        selected_names = args.preset or [
            name for name, run in runs.items() if run.get("style") == "clean" and run.get("run", True)
        ]
        for name in selected_names:
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
