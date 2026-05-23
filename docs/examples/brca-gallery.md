# BRCA Gallery Example

The BRCA gallery uses deterministic synthetic inputs with large-cohort ordering,
subtype/receptor metadata, numeric age metadata, TMB bars, recurrence labels,
and mutation legends.

## Render The Default BRCA Outputs

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --preset brca_large
python3 python_refactor_goal_sources/recreate_gallery.py --preset brca_compact_complex
```

Outputs:

```text
python_refactor_goal_sources/generated_plots/clean/gen.goal_plot_4.png
python_refactor_goal_sources/generated_plots/clean/gen.goal_plot_5.png
```

## Render Reference-Like Variants

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --style reference_like --preset brca_large
```

The large reference-like variant emphasizes:

- wider matrix usage across the canvas.
- larger left gene labels.
- stacked top burden bars.
- right-side recurrence bars with percentages.
- bottom metadata tracks.
- a full right legend column.

## Inputs

```text
python_refactor_goal_sources/syntheitic_goal_data/brca_mutations.tsv
python_refactor_goal_sources/syntheitic_goal_data/brca_metadata.tsv
python_refactor_goal_sources/syntheitic_goal_data/brca_tmb.tsv
python_refactor_goal_sources/syntheitic_goal_data/brca_palette.json
```

## Inspect The Preset Code

The gallery renderer lives in:

```text
python_refactor_goal_sources/recreate_gallery.py
```

Use this file as a recipe for larger Matplotlib layouts that are too bespoke for
the core `oncoplot()` API.

