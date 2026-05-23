# Structural Variation Panel

The structural-variation panel recreates `gen.goal_plot_8.png` from synthetic depth,
allele-fraction, and gene-model inputs.

It is gallery-specific and is not rendered through `oncoplot()`, because it is a
depth and allele-fraction panel rather than a mutation oncoplot.

## Render It

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --preset sv_panel
```

Output:

```text
python_refactor_goal_sources/generated_plots/clean/gen.goal_plot_8.png
```

## Inputs

```text
python_refactor_goal_sources/syntheitic_goal_data/sv_depth.tsv
python_refactor_goal_sources/syntheitic_goal_data/sv_allele_fraction.tsv
python_refactor_goal_sources/syntheitic_goal_data/sv_gene_models.tsv
```

## Input Shapes

`sv_depth.tsv`:

| Column | Meaning |
| --- | --- |
| `sample` | sample identifier |
| `title` | panel title |
| `position` | genomic position |
| `depth` | read depth |

`sv_allele_fraction.tsv`:

| Column | Meaning |
| --- | --- |
| `sample` | sample identifier |
| `position` | genomic position |
| `allele` | `REF` or `ALT` |
| `allele_fraction` | observed allele fraction |

`sv_gene_models.tsv`:

| Column | Meaning |
| --- | --- |
| `gene` | gene label |
| `start` | interval start |
| `end` | interval end |
| `strand` | gene strand |

