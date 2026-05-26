# Development

## Local Setup

```bash
python3 -m pip install -e ".[test,export]"
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

## Repository Layout

```text
src/pyoncoplot/                         package code
tests/                                   pytest suite
docs/                                    Markdown documentation
plans/                                   implementation plans
python_refactor_goal_sources/            gallery scripts and training source tree
python_refactor_goal_sources/config.yaml config-driven gallery run definitions
python_refactor_goal_sources/goal_plots/ numbered source/reference plots
python_refactor_goal_sources/syntheitic_goal_data/ deterministic synthetic TSV/JSON inputs
```

## Regenerate Gallery Inputs

```bash
python3 python_refactor_goal_sources/generate_synthetic_inputs.py
```

This rewrites deterministic TSV/JSON files under:

```text
python_refactor_goal_sources/syntheitic_goal_data/
```

## Regenerate Gallery Images

```bash
python3 python_refactor_goal_sources/recreate_gallery.py
python3 python_refactor_goal_sources/recreate_gallery.py --style comparison --preset brca_large
```

Generated gallery images are tracked when intentionally updated.

## Adding A Gallery Preset

1. Add or extend synthetic inputs in `python_refactor_goal_sources/generate_synthetic_inputs.py`.
2. Commit the generated TSV/JSON inputs.
3. Add a renderer function in `python_refactor_goal_sources/recreate_gallery.py` only when an existing renderer cannot express the plot.
4. Add the named run to `python_refactor_goal_sources/config.yaml` under `gallery_params.plot_runs`.
5. Add or update tests in `tests/test_gallery.py`.
6. Document the preset in [Gallery](gallery.md).

Renderer functions accept `params={...}` plus explicit overrides. Keep data
loading in named loader functions and put runtime choices such as genes,
metadata tracks, expected dimensions, output names, and save DPI in YAML.

## Documentation Changes

Docs are Markdown-first. Keep examples runnable from the repository root unless
the page says otherwise.

After editing docs:

```bash
.venv/bin/python -m pytest -q
```

The docs test checks that required pages exist and that local Markdown links
resolve.
