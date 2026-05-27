# API Reference

This is a human-written reference for the public API. It focuses on stable
behavior and examples rather than generated internals.

## `oncoplot`

```python
oncoplot(
    data=None,
    *,
    params=None,
    **kwargs,
)
```

Create an oncoplot from mutation-level data and return an `OncoplotResult`.
The function accepts normal explicit keyword arguments, a `params` dictionary,
or both. Explicit keywords override values in `params`.

Common call:

```python
from pyoncoplot import oncoplot

result = oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    draw_gene_bar=True,
    draw_tmb_bar=True,
    backend="matplotlib",
)
```

Config-driven or reusable calls can pass the same arguments as a dictionary:

```python
params = {
    "data": mutations,
    "gene_col": "gene",
    "sample_col": "sample",
    "mutation_type_col": "mutation_type",
    "backend": "matplotlib",
    "options": {"width": 900, "height": 520},
}

result = oncoplot(params=params, top_n=20)
```

Here `top_n=20` overrides any `params["top_n"]` value.

### Core Arguments

| Argument | Purpose |
| --- | --- |
| `data` | mutation-level `pandas.DataFrame` |
| `gene_col`, `sample_col` | required column names for gene and sample identifiers |
| `mutation_type_col` | optional mutation category column used for tile colors and legends |
| `tooltip_col` | optional tooltip text column; defaults to `sample_col` |
| `include_genes`, `ignore_genes`, `top_n` | choose the displayed gene panel |
| `draw_gene_bar`, `draw_tmb_bar` | add recurrence and mutation burden side panels |
| `palette`, `tmb_palette`, `metadata_palette` | color mappings for mutation tiles, typed TMB bars, categorical metadata, and numeric metadata colormaps |
| `metadata`, `metadata_cols`, `metadata_sample_col` | clinical annotation input and selected tracks |
| `metadata_require_mutations`, `show_all_samples` | sample inclusion controls |
| `pathway`, `pathway_gene_col` | pathway grouping input |
| `sample_order`, `metadata_sort_cols` | explicit or metadata-driven sample sorting |
| `tmb_data` | optional 2- or 3-column custom TMB table |
| `backend`, `interactive` | choose Plotly or Matplotlib rendering |
| `copy_on_click` | Plotly clipboard payload behavior |
| `options` | `OncoplotOptions` instance or mapping for visual controls |

## `OncoplotResult`

Returned by `oncoplot()`.

| Attribute or method | Purpose |
| --- | --- |
| `figure` | backend-specific Plotly or Matplotlib figure |
| `backend` | `"plotly"` or `"matplotlib"` |
| `prepared_data` | transformed data shared by renderers |
| `show()` | call the backend's display method |
| `save(path, **kwargs)` | save HTML or image output |
| `to_html(...)` | Plotly-only HTML string export |

### Save Behavior

`OncoplotResult.save()` chooses behavior from the file suffix. Plotly results
save `.html` directly, while Plotly image suffixes such as `.png`, `.svg`, and
`.pdf` require the export extra. Matplotlib results save through
`figure.savefig()` and support the image/vector formats available in the local
Matplotlib installation.

## `prepare_oncoplot_data`

Prepares mutation, metadata, pathway, and TMB inputs without rendering.

Use it when you want to inspect selected genes, sample order, collapsed tiles, or
metadata filtering:

```python
from pyoncoplot import prepare_oncoplot_data

prepared = prepare_oncoplot_data(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    top_n=10,
)

print(prepared.genes)
print(prepared.samples)
print(prepared.tiles.head())
```

### `PreparedOncoplotData` Fields

| Field | Summary |
| --- | --- |
| `tiles` | one displayed row per sample/gene tile, with collapsed `MutationType` and tooltip text |
| `samples`, `genes` | final display order used by both renderers |
| `total_samples` | denominator used for recurrence percentages |
| `mutation_type_col` | original mutation type column name, when supplied |
| `metadata`, `metadata_cols`, `metadata_tracks` | filtered metadata table and renderer-neutral track summaries |
| `pathway`, `pathway_by_gene`, `pathway_groups` | pathway input and contiguous display groups |
| `tmb` | filtered TMB input or inferred mutation burden table |
| `tmb_sample_col`, `tmb_value_col`, `tmb_type_col` | resolved TMB sample, numeric value, and optional subtype columns |
| `tmb_render_stacked`, `tmb_is_custom` | flags used by renderers to decide stacked TMB behavior |
| `mutation_counts`, `tmb_totals`, `tmb_type_counts` | summary tables for testing, debugging, and downstream inspection |

## `identify_top_genes`

Rank genes by the number of distinct mutated samples.

```python
from pyoncoplot import identify_top_genes

genes = identify_top_genes(
    mutations,
    gene_col="gene",
    sample_col="sample",
    top_n=20,
)
```

## `score_sample_by_gene_rank`

Assign one sample a score based on mutations in higher-ranked genes. This is the
low-level helper used to produce a useful default sample ordering.

```python
from pyoncoplot import score_sample_by_gene_rank

score = score_sample_by_gene_rank(
    mutated_genes=["TP53", "PTEN"],
    genes_informing_score=["TP53", "PIK3CA", "PTEN"],
    gene_rank=[1, 2, 3],
)
```

## `rank_genes_by_pathway`

Rank genes while respecting pathway order.

```python
from pyoncoplot import rank_genes_by_pathway

gene_pathway_map = pd.DataFrame(
    {
        "gene": ["TP53", "RB1", "PIK3CA", "PTEN"],
        "pathway": ["Cell cycle", "Cell cycle", "PI3K", "PI3K"],
    }
)

ranked = rank_genes_by_pathway(
    gene_pathway_map,
    gene_ranks=["PIK3CA", "TP53", "PTEN", "RB1"],
    pathway_ranks=["Cell cycle", "PI3K"],
)
```

## `prettify`

Convert machine-style labels into display labels.

```python
from pyoncoplot import prettify

prettify("Missense_Mutation")
# "Missense Mutation"
```

## Palette Helpers

```python
from pyoncoplot import assert_palette_is_sensible, get_sensible_default_palette

palette = get_sensible_default_palette(mutations["mutation_type"])
assert_palette_is_sensible(palette, mutations["mutation_type"])
```
