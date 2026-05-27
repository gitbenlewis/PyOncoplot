# Palettes

Palettes are plain Python mappings from values to CSS-style colors.

## Mutation Palette

```python
palette = {
    "Missense_Mutation": "#2CA02C",
    "Frame_Shift_Del": "#1F77B4",
    "Nonsense_Mutation": "#17BECF",
    "Splice_Site": "#FF7F0E",
    "Multi_Hit": "#000000",
}
```

Use it with:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    palette=palette,
)
```

The palette must cover every mutation type that appears in displayed tiles.
Ampersand-delimited Sequence Ontology values such as
`missense_variant&intron_variant` are rejected; preselect the most severe
consequence before plotting.

## Metadata Palette

Metadata palettes map metadata columns to explicit category mappings, named
palettes, or numeric colormaps:

```python
metadata_palette = {
    "FAB_classification": "tol_colors",
    "Overall_Survival_Status": "Iridescent",
    "Mean VAF%": "viridis_greyzero",
}
```

Use it with:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    metadata=metadata,
    metadata_cols=["FAB_classification", "Overall_Survival_Status"],
    metadata_palette=metadata_palette,
)
```

Categorical palette names are case-sensitive. PyOncoplot first resolves names
from `pyoncoplot.palettes`, such as `tol_colors` and `Iridescent`, then falls
back to Matplotlib colormap names such as `"Dark2"`. Named palettes and color
sequences assign colors in observed category order. Finite color lists wrap
around from the first color again when a track has more categories than colors.

You can still pass explicit category mappings. Categorical levels not listed in
an explicit metadata palette receive fallback colors.

For numeric metadata, `metadata_palette` can specify a true continuous colormap
per column. Use a Matplotlib colormap name such as `"viridis"`, a PyOncoplot
palette name such as `"Iridescent"`, or a sequence of colors:

```python
metadata_palette = {
    "CAF%": "viridis_greyzero",
    "Mean VAF%": "magma",
    "Tumor purity": "Iridescent",
}

oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    metadata=metadata,
    metadata_cols=["CAF%", "Mean VAF%", "Tumor purity"],
    metadata_palette=metadata_palette,
)
```

Numeric metadata columns must have numeric dtype. Convert strings such as
`"42%"` before plotting:

```python
metadata["CAF%"] = pd.to_numeric(metadata["CAF%"].astype(str).str.rstrip("%"), errors="coerce")
```

Plotly shows continuous colorbars for numeric metadata when
`options={"show_metadata_legends": True}`. Set
`options={"metadata_legend_orientation_heatmap": "horizontal"}` to place
numeric metadata colorbars horizontally.

PyOncoplot also exports reusable color cycles and ramps from
`pyoncoplot.palettes`: `tol_colors`, `Iridescent`, `vega_10`,
`vega_10_scanpy`, `vega_20`, `vega_20_scanpy`, `default_20`, `zeileis_28`,
`default_28`, `godsnot_102`, `default_102`, and the gray-zero colormaps
listed below.

For sparse continuous tracks where exact zero should stand out as gray,
PyOncoplot registers gray-zero colormaps named `viridis_greyzero`,
`plasma_greyzero`, `magma_greyzero`, `inferno_greyzero`, `cividis_greyzero`,
and `turbo_greyzero`. These are built as a gray first color followed by the
256-color base Matplotlib ramp.

## TMB Palette

When TMB data includes a mutation type column, pass a matching palette to stack
TMB bars by mutation type:

```python
oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    draw_tmb_bar=True,
    tmb_data=tmb,
    tmb_palette=palette,
    options=OncoplotOptions(log10_transform_tmb=False),
)
```

Stacked TMB bars are most useful with `log10_transform_tmb=False`.
Palette coverage for custom TMB categories is required only when stacked subtype
bars are rendered with `log10_transform_tmb=False`. If those categories are not
covered by `tmb_palette` or by the mutation palette fallback, `pyoncoplot`
raises a clear error instead of drawing uncolored bars.

Default-generated mutation palettes use `OncoplotOptions.multi_hit_color` for
`Multi_Hit`. An explicit `palette={"Multi_Hit": ...}` entry takes precedence.

## Default Palettes

If no mutation palette is provided, `pyoncoplot` creates a sensible default for
the observed mutation types.

You can inspect or validate palettes directly:

```python
from pyoncoplot import assert_palette_is_sensible, get_sensible_default_palette

palette = get_sensible_default_palette(mutations["mutation_type"])
assert_palette_is_sensible(palette, mutations["mutation_type"])
```

## Practical Color Guidance

- Use high-contrast mutation colors.
- Reserve black or near-black for multi-hit or truncating events.
- Use white carefully in metadata palettes because white can look like missing data.
- Keep numeric metadata bars neutral unless the value has an obvious semantic direction.
