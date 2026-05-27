# Data Inputs

`pyoncoplot` expects mutation-level data: one row per mutation event.

## Required Columns

The public API lets you name your columns explicitly:

```python
oncoplot(
    data,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
)
```

Required mutation table fields:

| Concept | API argument | Required | Notes |
| --- | --- | --- | --- |
| sample identifier | `sample_col` | yes | cannot be missing or empty |
| gene identifier | `gene_col` | yes | cannot be missing or empty |
| mutation type | `mutation_type_col` | no | required for mutation-specific colors |
| variant value | `variant_value_col`, `variant_value_cols`, `main_grid_rows` | no | numeric values used for continuous main-grid coloring |
| tooltip text | `tooltip_col` | no | defaults to `sample_col` when omitted |

## Example Mutation Table

```python
mutations = pd.DataFrame(
    {
        "sample": ["S1", "S1", "S2"],
        "gene": ["TP53", "TP53", "PTEN"],
        "mutation_type": ["Missense_Mutation", "Splice_Site", "Frame_Shift_Del"],
        "tooltip": ["TP53 p.R175H", "TP53 splice", "PTEN frameshift"],
    }
)
```

## Multi-Hit Behavior

If one sample has multiple rows for the same gene, the plot collapses those rows
into a single tile. With a mutation type column, multi-hit cells are marked as
`Multi_Hit` when more than one mutation row is present for that sample/gene
cell, even if the repeated rows have the same mutation type.

The tooltip content is aggregated so the original mutation-level evidence is not
lost in interactive output.

When `variant_value_col` is supplied, collapsed sample/gene tiles also aggregate
that numeric column. Use `variant_value_agg` to choose `"max"` (default),
`"mean"`, `"median"`, or `"min"`. Missing values in displayed variants are
rejected so heatmap colors are deterministic.

Multiple numeric variant values can be plotted as separate subrows under each
gene. The concise form keeps mutation type visible first and then adds one row
per value column:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    variant_value_cols=["VAF_pct", "VAF_abs"],
)
```

Use `main_grid_rows` for custom labels, per-row aggregation, or per-row
palettes:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    main_grid_rows=[
        {"kind": "mutation_type", "label": "Variant type"},
        {"kind": "variant_value", "column": "VAF_pct", "label": "VAF %"},
        {"kind": "variant_value", "column": "deltaVAF_pct", "label": "delta VAF %", "agg": "mean"},
    ],
)
```

## Gene Selection

Use `top_n` to choose the most recurrent genes by distinct mutated samples:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    top_n=20,
)
```

Use explicit genes when you need a fixed panel:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    include_genes=["TP53", "PIK3CA", "PTEN"],
)
```

Other gene-selection controls:

| Argument | Behavior |
| --- | --- |
| `ignore_genes` | removes genes before ranking |
| `include_genes` | uses this list and order when possible |
| `return_extra_genes_if_tied` | returns all genes tied at the cutoff |
| `top_n=None` | uses all eligible genes |

## Sample Selection

By default, samples with no selected-gene mutations are not shown. Use
`show_all_samples=True` to keep all samples available from mutation data and,
when supplied, custom TMB input. Metadata-only samples are retained only when
`metadata_require_mutations=False` is also set.

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    show_all_samples=True,
)
```

Use `sample_order` for a fixed display order:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    sample_order=["S3", "S1", "S2"],
)
```

## Validation Rules

The data preparation layer checks:

- mutation data must be a non-empty `pandas.DataFrame`.
- sample and gene columns must exist.
- sample and gene identifiers cannot be missing or empty strings.
- mutation type values cannot be missing when `mutation_type_col` is supplied.
- metadata sample identifiers must be unique.
- pathway input must have exactly two columns and no duplicate genes.
- palette coverage must include all displayed mutation types.
