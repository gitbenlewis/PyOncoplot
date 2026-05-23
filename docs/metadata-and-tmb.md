# Metadata and TMB

Clinical metadata and mutation burden bars are optional, but they are often what
turns an oncoplot from a mutation grid into an interpretable cohort summary.

## Metadata Table Schema

Metadata input needs one unique row per sample.

```python
metadata = pd.DataFrame(
    {
        "sample": ["S1", "S2", "S3"],
        "Subtype": ["A", "B", "A"],
        "Age_years": [62, 47, 71],
    }
)
```

Use `metadata_cols` to choose displayed tracks:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    metadata=metadata,
    metadata_cols=["Subtype", "Age_years"],
)
```

If the metadata sample column is not named the same as `sample_col`, pass
`metadata_sample_col`.

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="Tumor_Sample_Barcode",
    mutation_type_col="Variant_Classification",
    metadata=metadata,
    metadata_sample_col="patient_id",
    metadata_cols=["Subtype"],
)
```

## Metadata Filtering

By default, `metadata_require_mutations=True`, so metadata rows without displayed
mutation samples are filtered out unless `show_all_samples=True` is used.

Use:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    metadata=metadata,
    metadata_cols=["Subtype"],
    show_all_samples=True,
)
```

to show samples even when they do not have selected-gene mutations.

## Numeric Metadata

Numeric metadata can be shown as a heatmap or as mini bars.

```python
from pyoncoplot import OncoplotOptions

options = OncoplotOptions(metadata_numeric_plot_type="bar")
```

Use `"bar"` for continuous values where relative magnitude matters, such as age
or follow-up days. Use `"heatmap"` for compact display.

## TMB Inference

When `draw_tmb_bar=True` and no `tmb_data` is supplied, TMB is inferred from the
mutation table.

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    draw_tmb_bar=True,
)
```

## Custom TMB Data

Custom TMB input should include sample identifiers and count values. When it also
includes mutation types, the renderers can draw stacked mutation-type bars when
`log10_transform_tmb=False`.

```python
tmb = pd.DataFrame(
    {
        "sample": ["S1", "S1", "S2"],
        "mutation_type": ["Missense_Mutation", "Splice_Site", "Missense_Mutation"],
        "mutations": [12, 3, 7],
    }
)

oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    draw_tmb_bar=True,
    tmb_data=tmb,
    tmb_palette=palette,
    backend="matplotlib",
    options=OncoplotOptions(log10_transform_tmb=False),
)
```

If `log10_transform_tmb=True` with custom stacked TMB data, Plotly renders
sample totals instead of stacked subtype bars and records a warning. Custom TMB
categories must be covered by `tmb_palette` unless they are already present in
the mutation palette fallback.
