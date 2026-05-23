# Troubleshooting

## Missing Column Errors

Error pattern:

```text
Could not find column: ...
```

Fix: check the column name and pass the correct API argument:

```python
oncoplot(
    mutations,
    gene_col="Hugo_Symbol",
    sample_col="Tumor_Sample_Barcode",
    mutation_type_col="Variant_Classification",
)
```

## Empty Or Missing Sample/Gene Identifiers

Sample and gene columns cannot contain missing values or empty strings.

```python
mutations = mutations.dropna(subset=["sample", "gene"])
mutations = mutations[(mutations["sample"] != "") & (mutations["gene"] != "")]
```

## Palette Coverage Errors

If a mutation palette is supplied, it must cover displayed mutation types.

```python
set(mutations["mutation_type"]) - set(palette)
```

Add missing colors or let `pyoncoplot` create a default palette.

## Metadata Duplicate Sample Errors

Metadata must have one row per sample.

```python
metadata[metadata["sample"].duplicated()]
```

Deduplicate or aggregate metadata before plotting.

## Samples Disappear

By default, samples without selected-gene mutations are filtered out. Use:

```python
show_all_samples=True
```

when metadata or TMB should keep the full cohort visible.

## Plotly Image Export Fails

HTML export works without Kaleido:

```python
result.save("plot.html")
```

For PNG/SVG/PDF export from Plotly, install the export extra:

```bash
python3 -m pip install -e ".[export]"
```

## Matplotlib Font Warnings

Matplotlib may warn about local font discovery or parser deprecations. These are
usually environment warnings and do not necessarily mean the figure failed.

## Very Large Cohorts

Rendering cost should scale mostly with displayed samples times displayed genes,
not raw mutation row count. Use `include_genes`, `ignore_genes`, or `top_n` to
control the displayed matrix size.

