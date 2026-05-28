"""Recreate the full gallery by dispatching to the split renderer scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

try:
    from python_refactor_goal_sources.make_oncoplots import (
        CLEAN_OUT,
        COMPARISON_OUT,
        CONFIG_PATH,
        GALLERY_CONFIG,
        GENERATED_ROOT,
        GOAL_PLOTS,
        INPUTS,
        clean_run_names,
        merged_run_config,
        render_preset as render_oncoplot_preset,
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
        clean_run_names,
        merged_run_config,
        render_preset as render_oncoplot_preset,
    )
    from make_other_gallery_plots import render_preset as render_other_preset  # type: ignore


def render_preset(name: str, out_dir: Optional[Path] = None, style: str = "clean") -> Path:
    if style == "comparison":
        return render_other_preset(name, out_dir or COMPARISON_OUT, style="comparison")
    if style != "clean":
        raise ValueError("style must be one of: clean, comparison.")

    run_config = merged_run_config(name)
    out_dir = out_dir or CLEAN_OUT
    if run_config["renderer"] == "oncoplot":
        return render_oncoplot_preset(name, out_dir)
    return render_other_preset(name, out_dir, style="clean")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=["clean", "comparison", "both"], default="clean")
    parser.add_argument("--preset", action="append", help="Preset name to render. May be supplied more than once.")
    parser.add_argument("--out-dir", type=Path, help="Override the output directory for the selected style.")
    args = parser.parse_args(argv)

    if args.style in {"clean", "both"}:
        clean_out = args.out_dir if args.style == "clean" and args.out_dir else CLEAN_OUT
        for name in clean_run_names(args.preset):
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
