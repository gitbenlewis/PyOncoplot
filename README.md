# pyoncoplot

`pyoncoplot` is a Pythonic implementation of oncoplots inspired by the
R package `ggoncoplot`. It accepts mutation-level tabular data and can render
interactive Plotly oncoplots or static Matplotlib figures.

## Documentation

Full documentation lives in [docs/index.md](docs/index.md), including:

- [quickstart examples](docs/quickstart.md)
- [input schemas](docs/data-inputs.md)
- [metadata and TMB usage](docs/metadata-and-tmb.md)
- [gallery recreation](docs/gallery.md)
- [migration notes for ggoncoplot users](docs/migration-from-ggoncoplot.md)

## Install for Development

```bash
python3 -m pip install -e ".[test,export]"
```

The `export` extra adds `kaleido` for Plotly image export. HTML export does not
need it.

## Quick Start

```python
import pandas as pd
from pyoncoplot import oncoplot

mutations = pd.DataFrame(
    {
        "sample": ["S1", "S1", "S2", "S3"],
        "gene": ["TP53", "EGFR", "TP53", "PTEN"],
        "mutation_type": [
            "Missense_Mutation",
            "Frame_Shift_Del",
            "Nonsense_Mutation",
            "Splice_Site",
        ],
    }
)

result = oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    top_n=10,
    draw_gene_bar=True,
    draw_tmb_bar=True,
)

result.save("oncoplot.html")
```

The same call can be made from a reusable parameter dictionary; explicit
keywords override dictionary values:

```python
params = {
    "data": mutations,
    "gene_col": "gene",
    "sample_col": "sample",
    "mutation_type_col": "mutation_type",
    "top_n": 5,
}

result = oncoplot(params=params, top_n=10)
```

For a static image:

```python
result = oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    backend="matplotlib",
    draw_gene_bar=True,
)
result.save("oncoplot.png", dpi=120)
```

## Recreate the Example Gallery

The reference PNGs in `python_refactor_goal_sources/goal_plots/` can be recreated with:

```bash
python3 python_refactor_goal_sources/recreate_gallery.py
```

Generated files are written to `python_refactor_goal_sources/generated_plots/clean/` as
`gen.goal_plot_1.png` through `gen.goal_plot_21.png`, so the original reference
images remain untouched. Gallery runs are configured in
`python_refactor_goal_sources/config.yaml` under `gallery_params.plot_runs`.
The gallery uses deterministic synthetic inputs stored in `python_refactor_goal_sources/syntheitic_goal_data/`; regenerate those fixtures
with:

```bash
python3 python_refactor_goal_sources/generate_synthetic_inputs.py
```

The AML metadata gallery outputs in `generated_plots/clean/gen.goal_plot_2.png` and
`generated_plots/clean/gen.goal_plot_3.png` are treated as the approved clean baseline.
BRCA-specific reference-like variants and side-by-side comparison sheets can be
rendered separately:

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --style reference_like --preset brca_large
python3 python_refactor_goal_sources/recreate_gallery.py --style comparison --preset brca_large
```

## Pythonic API

The public API intentionally uses Python names rather than preserving R names:

| R `ggoncoplot` argument | Python `pyoncoplot` argument |
| --- | --- |
| `col_genes` | `gene_col` |
| `col_samples` | `sample_col` |
| `col_mutation_type` | `mutation_type_col` |
| `col_tooltip` | `tooltip_col` |
| `genes_to_include` | `include_genes` |
| `genes_to_ignore` | `ignore_genes` |
| `topn` | `top_n` |
| `draw_gene_barplot` | `draw_gene_bar` |
| `draw_tmb_barplot` | `draw_tmb_bar` |
| `copy` | `copy_on_click` |
| `cols_to_plot_metadata` | `metadata_cols` |
| `col_samples_metadata` | `metadata_sample_col` |
| `col_genes_pathway` | `pathway_gene_col` |

## Attribution

This package ports behavior from the MIT-licensed
[`ggoncoplot`](https://github.com/selkamand/ggoncoplot) R package. The original
R implementation is retained for reference in this fork's Git history and as a
pinned submodule at `python_refactor_goal_sources/ggoncoplot`. The Python
implementation is not intended to be pixel-identical to the ggplot output, but
it follows the same core data semantics: top-gene selection, multi-hit collapse,
sample sorting, metadata handling, pathway grouping, TMB bars, and mutation
palettes.
