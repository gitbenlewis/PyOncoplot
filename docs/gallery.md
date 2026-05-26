# Example Gallery

The gallery recreates reference-style plots from deterministic synthetic inputs.
Generated images are written outside the reference image paths so originals stay
untouched.

Runtime choices live in `python_refactor_goal_sources/config.yaml` under the
`gallery_params` block. The script loads that YAML once, merges `default_params`
into named `plot_runs`, skips runs with `run: false`, and dispatches through a
small renderer registry. Input filenames are also declared there under
`gallery_params.input_files`; the loader functions read the TSV/JSON files from
those configured names.

## Render The Clean Gallery

```bash
python3 python_refactor_goal_sources/recreate_gallery.py
```

Outputs are written to:

```text
python_refactor_goal_sources/generated_plots/clean/
```

## Regenerate Synthetic Inputs

```bash
python3 python_refactor_goal_sources/generate_synthetic_inputs.py
```

Synthetic inputs are checked in under:

```text
python_refactor_goal_sources/syntheitic_goal_data/
```

## BRCA Comparison Sheets

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --style comparison --preset brca_large
```

Comparison sheets contain two panels: the original source image and the clean
generated image.

Outputs are written to:

```text
python_refactor_goal_sources/generated_plots/comparison/
```

## Accepted Clean Baselines

The generated AML metadata plots are treated as approved clean baselines:

- `python_refactor_goal_sources/generated_plots/clean/gen.goal_plot_2.png`
- `python_refactor_goal_sources/generated_plots/clean/gen.goal_plot_3.png`

Do not tune these toward the originals if it makes the generated versions worse.

## Presets

| Preset | Output | Size | Notes |
| --- | --- | --- | --- |
| `aml_basic` | `gen.goal_plot_1.png` | `1080 x 720` | basic AML oncoplot |
| `aml_metadata_unsorted` | `gen.goal_plot_2.png` | `1080 x 720` | accepted clean baseline |
| `aml_metadata_sorted` | `gen.goal_plot_3.png` | `1080 x 720` | accepted clean baseline |
| `brca_large` | `gen.goal_plot_4.png` | `3600 x 1800` | large BRCA plot |
| `brca_compact_complex` | `gen.goal_plot_5.png` | `850 x 683` | compact BRCA plot |
| `cssc_compact` | `gen.goal_plot_6.png` | `1400 x 700` | compact alteration matrix |
| `gbm_clinical_molecular` | `gen.goal_plot_7.png` | `1080 x 436` | GBM-style clinical/molecular heatmap |
| `sv_panel` | `gen.goal_plot_8.png` | `1296 x 864` | structural-variation panel |
| `ggoncoplot_readme_small` | `gen.goal_plot_9.png` | `672 x 480` | small README oncoplot |
| `ggoncoplot_readme_basic` | `gen.goal_plot_10.png` | `7200 x 3000` | README basic oncoplot |
| `ggoncoplot_readme_marginal` | `gen.goal_plot_11.png` | `7200 x 3600` | README oncoplot with marginal bars |
| `ggoncoplot_readme_metadata` | `gen.goal_plot_12.png` | `7200 x 3600` | README oncoplot with clinical metadata |
| `ggoncoplot_package_mark` | `gen.goal_plot_13.png` | `125 x 144` | package mark recreation |
| `ggoncoplot_interactive_snapshot` | `gen.goal_plot_14.png` | `1566 x 1036` | static interactive snapshot |
| `ggoncoplot_comparison_table` | `gen.goal_plot_15.png` | `800 x 533` | package comparison table |
| `ggoncoplot_comparison_table_jats` | `gen.goal_plot_16.png` | `800 x 513` | compact comparison table |
| `lasso_select` | `gen.goal_plot_17.png` | `1408 x 922` | lasso-selection scatterplot |
| `multimodal_selection_old` | `gen.goal_plot_18.png` | `7620 x 5204` | multimodal linked panel |
| `multimodal_selection` | `gen.goal_plot_19.png` | `7620 x 5204` | multimodal linked panel |
| `multimodal_selection_with_lasso` | `gen.goal_plot_20.png` | `2281 x 1520` | multimodal panel with lasso |
| `paper_gbm_oncoplot` | `gen.goal_plot_21.png` | `864 x 432` | compact paper-style GBM oncoplot |

## Config-Driven Runs

Each `plot_runs` entry declares its renderer, output file, source goal plot,
expected size, run toggle, and renderer params:

```yaml
gallery_params:
  plot_runs:
    aml_basic:
      run: true
      renderer: aml_basic
      style: clean
      output_name: gen.goal_plot_1.png
      goal_plot: goal_plot_1.png
      expected_size: [1080, 720]
      params:
        include_genes: [FLT3, DNMT3A, NPM1]
        oncoplot:
          gene_col: gene
          sample_col: sample
```

Generated outputs keep the numbered naming convention:
`gen.goal_plot_N.png` for generated plots and `compare.goal_plot_N.png` for
comparison sheets.

## Synthetic Input Families

| Family | Files |
| --- | --- |
| AML | `aml_mutations.tsv`, `aml_metadata.tsv`, `aml_tmb.tsv`, `aml_palette.json` |
| BRCA | `brca_mutations.tsv`, `brca_metadata.tsv`, `brca_tmb.tsv`, `brca_palette.json` |
| CSSC | `cssc_mutations.tsv`, `cssc_tmb.tsv`, `cssc_palette.json` |
| GBM | `gbm_clinical_tracks.tsv`, `gbm_events.tsv`, `gbm_palette.json` |
| SV | `sv_depth.tsv`, `sv_allele_fraction.tsv`, `sv_gene_models.tsv` |
| ggoncoplot README | `ggoncoplot_readme_mutations.tsv`, `ggoncoplot_readme_metadata.tsv`, `ggoncoplot_readme_tmb.tsv`, `ggoncoplot_readme_palette.json` |
| Multimodal paper | `paper_multimodal_samples.tsv`, `paper_multimodal_points.tsv`, `paper_multimodal_events.tsv`, `paper_multimodal_clinical.tsv`, `paper_multimodal_selection.tsv`, `paper_multimodal_palette.json` |
| Comparison table | `ggoncoplot_comparison_table.tsv` |

## Related Examples

- [Metadata Example](examples/metadata.md)
- [BRCA Gallery Example](examples/brca-gallery.md)
- [Structural Variation Panel](examples/structural-variation-panel.md)
