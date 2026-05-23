# Python Refactor Goal Sources

This directory is the canonical source area for reproducing the training and
gallery plots used by `pyoncoplot`.

## Folder Layout

```text
python_refactor_goal_sources/
  config.yaml                      config-driven named gallery runs
  recreate_gallery.py              gallery renderer and CLI
  generate_synthetic_inputs.py     deterministic synthetic input generator
  source_info_for_training.md      this source map
  goal_plots/                      immutable numbered source/reference plots
  syntheitic_goal_data/            deterministic TSV/JSON inputs
  generated_plots/                 disposable generated outputs, ignored by git
  ggoncoplot/                      cloned R reference implementation
```

The folder names `python_refactor_goal_sources` and `syntheitic_goal_data` keep
their current spelling intentionally.

## Source Rules

- `goal_plots/` is immutable source/reference material.
- Generated images must never be written to `goal_plots/`.
- `generated_plots/` is disposable output and ignored by git.
- All source reference plots are PNG files named `goal_plot_N.png`.
- Existing plots without a reliable public source are kept with
  `source_status: unresolved`; new imported plots require a public source URL.
- Clean generated counterparts are configured in `config.yaml` as
  `gen.goal_plot_1.png` through `gen.goal_plot_21.png`.

## Generated Counterparts

| Goal plot | Clean generated plot | Renderer intent |
| --- | --- | --- |
| `goal_plot_1.png` | `gen.goal_plot_1.png` | basic AML oncoplot |
| `goal_plot_2.png` | `gen.goal_plot_2.png` | AML metadata oncoplot, unsorted |
| `goal_plot_3.png` | `gen.goal_plot_3.png` | AML metadata oncoplot, sorted |
| `goal_plot_4.png` | `gen.goal_plot_4.png` | large BRCA oncoplot |
| `goal_plot_5.png` | `gen.goal_plot_5.png` | compact BRCA oncoplot |
| `goal_plot_6.png` | `gen.goal_plot_6.png` | compact CSSC alteration matrix |
| `goal_plot_7.png` | `gen.goal_plot_7.png` | GBM clinical/molecular panel |
| `goal_plot_8.png` | `gen.goal_plot_8.png` | structural-variation panel |
| `goal_plot_9.png` | `gen.goal_plot_9.png` | small README oncoplot |
| `goal_plot_10.png` | `gen.goal_plot_10.png` | README basic oncoplot |
| `goal_plot_11.png` | `gen.goal_plot_11.png` | README oncoplot with marginal bars |
| `goal_plot_12.png` | `gen.goal_plot_12.png` | README oncoplot with clinical metadata |
| `goal_plot_13.png` | `gen.goal_plot_13.png` | package mark recreation |
| `goal_plot_14.png` | `gen.goal_plot_14.png` | static interactive oncoplot snapshot |
| `goal_plot_15.png` | `gen.goal_plot_15.png` | package comparison table |
| `goal_plot_16.png` | `gen.goal_plot_16.png` | compact package comparison table |
| `goal_plot_17.png` | `gen.goal_plot_17.png` | lasso-selection scatterplot |
| `goal_plot_18.png` | `gen.goal_plot_18.png` | multimodal selection panel |
| `goal_plot_19.png` | `gen.goal_plot_19.png` | multimodal selection panel |
| `goal_plot_20.png` | `gen.goal_plot_20.png` | multimodal panel with lasso highlight |
| `goal_plot_21.png` | `gen.goal_plot_21.png` | compact paper-style GBM oncoplot |

## Goal Plot Source Table

| Goal plot | Source status | Original local source path | Original filename | Source project/page | Source URL | License/attribution | Dimensions | SHA256 | Import action | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `goal_plot_1.png` | resolved | `previous example_oncoplots/oncoplot.png` | `oncoplot.png` | fuc tutorials, Create customized oncoplots | https://sbslee-fuc.readthedocs.io/en/latest/tutorials.html | fuc documentation by Seung-been Steven Lee; original license not bundled here | 1080 x 720 | `97e920006300cc26d5d301018863b4bd482f008061164fd832f4336972381710` | preserved existing PNG | Standard oncoplot tutorial image. |
| `goal_plot_2.png` | resolved | `previous example_oncoplots/customized_oncoplot_1.png` | `customized_oncoplot_1.png` | fuc tutorials, Create customized oncoplots | https://sbslee-fuc.readthedocs.io/en/latest/tutorials.html | fuc documentation by Seung-been Steven Lee; original license not bundled here | 1080 x 720 | `1589f9f9cc3899fb88738ea401f560c69b8f6ea648c44551f07bf7cd31885359` | preserved existing PNG | Customized oncoplot with annotations. |
| `goal_plot_3.png` | resolved | `previous example_oncoplots/customized_oncoplot_2.png` | `customized_oncoplot_2.png` | fuc tutorials, Create customized oncoplots | https://sbslee-fuc.readthedocs.io/en/latest/tutorials.html | fuc documentation by Seung-been Steven Lee; original license not bundled here | 1080 x 720 | `c838ebc206baa7ee3548472884fa1ce7c7baecbac1115803d757713cb8f33660` | preserved existing PNG | Customized oncoplot with sorted annotations. |
| `goal_plot_4.png` | resolved | `/ggoncoplot/paper/oncoplot.png` | `73b90cf5-8b0b-4787-b38e-f0b1a2d6.png`; matches `oncoplot.png` | ggoncoplot paper Figure 1 | https://github.com/selkamand/ggoncoplot/tree/main/paper | ggoncoplot; MIT + file LICENSE; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 3600 x 1800 | `a062b5f1868a18eb5754b149c91f0609b1c127d4207de4f0e7dfab48dadd85c1` | preserved existing PNG | Exact SHA match to nested `paper/oncoplot.png`; duplicate was not re-imported. |
| `goal_plot_5.png` | resolved | `previous example_oncoplots/9c4c2e0f-74b4-4992-82be-57048160.png` | `9c4c2e0f-74b4-4992-82be-57048160.png` | Zhang et al. Annals of Translational Medicine 2019, Figure 1 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6526269/ | Figure 1 from "Characterization of frequently mutated cancer genes in Chinese breast tumors: a comparison of Chinese and TCGA cohorts"; DOI 10.21037/atm.2019.04.23 | 850 x 683 | `be239584d040d28222dc7606c337ffdc118b90cdb26365ea57a77966ad6f18a8` | preserved existing PNG | Figure caption: "The mutational landscape of 33 cancer genes in Chinese breast tumors (n=305)." |
| `goal_plot_6.png` | resolved | `previous example_oncoplots/7a1ffb71-a4d9-4f42-8cf8-d992b037.png` | `7a1ffb71-a4d9-4f42-8cf8-d992b037.png` | Thind et al. Frontiers in Oncology 2022, Figure 2G | https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.919118/full | Figure panel from "Whole genome analysis reveals the genomic complexity in metastatic cutaneous squamous cell carcinoma"; Frontiers open-access CC BY; DOI 10.3389/fonc.2022.919118 | 1400 x 700 | `3c02828a7f23f6364828aebc42f25b9ce68825cbcb1ce88f05373abcc54573ad` | preserved existing PNG | Figure 2G is described as detailed sample-level SNV and variant-type information for top altered genes in metastatic CSCC. |
| `goal_plot_7.png` | resolved | `previous example_oncoplots/how-to-go-about-making-a-plot-like-this-v0-7ixn5xszvh3d1.webp` | `how-to-go-about-making-a-plot-like-this-v0-7ixn5xszvh3d1.webp` | Hoogstrate et al. Cancer Cell 2023 | https://www.sciencedirect.com/science/article/pii/S1535610823000478 | Oncoprint/clinical molecular panel from "Transcriptome analysis reveals tumor microenvironment changes in glioblastoma"; Cancer Cell / Elsevier open-access CC BY; DOI 10.1016/j.ccell.2023.02.019 | 1080 x 436 | `f4563ea506d1c5dd27de72b77c6ed7b20a6faa5d306d14a2173ea906b000f99f` | preserved existing converted PNG | User-identified source; article metadata also indexed by PubMed PMID 36898379 and Cardiff ORCA. |
| `goal_plot_8.png` | resolved | `previous example_oncoplots/vcf_sv.png` | `vcf_sv.png` | fuc tutorials and Biostars VCF structural-variation example | https://www.biostars.org/p/9493544/ | fuc tutorial example by Seung-been Steven Lee; Biostars post attribution to sbstevenlee | 1296 x 864 | `266ef3e2747c7e8227202418f9b9469083b00bddb5747dfef0dfdeb77d08d7e7` | preserved existing PNG | Also shown in fuc tutorials at https://sbslee-fuc.readthedocs.io/en/latest/tutorials.html. |
| `goal_plot_9.png` | resolved | `/ggoncoplot/man/figures/README-example-1.png` | `README-example-1.png` | ggoncoplot GitHub man/figures | https://github.com/selkamand/ggoncoplot/tree/main/man/figures | ggoncoplot; MIT + file LICENSE; Sam El-Kamand | 672 x 480 | `4f7bd78b8c0b7b7c92f1fd6397e3d0bb2b92098a0314721bcd9ebd6495c33451` | copied PNG | README example figure. |
| `goal_plot_10.png` | resolved | `/ggoncoplot/man/figures/README-unnamed-chunk-2-1.png` | `README-unnamed-chunk-2-1.png` | ggoncoplot README/pkgdown basic example | https://selkamand.github.io/ggoncoplot/ | ggoncoplot; MIT + file LICENSE; Sam El-Kamand | 7200 x 3000 | `630e24bdd8e29193901dea8db2a4691fe0281043f11f2cc88264d9246c494a87` | copied PNG | Basic static example from package README. |
| `goal_plot_11.png` | resolved | `/ggoncoplot/man/figures/README-unnamed-chunk-3-1.png` | `README-unnamed-chunk-3-1.png` | ggoncoplot README/pkgdown marginal plots example | https://selkamand.github.io/ggoncoplot/ | ggoncoplot; MIT + file LICENSE; Sam El-Kamand | 7200 x 3600 | `40aebfb68a06b726dd0618c097bad0951a04c881d50f95b4821c9bdb093dd51c` | copied PNG | README marginal bar plots example. |
| `goal_plot_12.png` | resolved | `/ggoncoplot/man/figures/README-unnamed-chunk-4-1.png` | `README-unnamed-chunk-4-1.png` | ggoncoplot README/pkgdown clinical metadata example | https://selkamand.github.io/ggoncoplot/ | ggoncoplot; MIT + file LICENSE; Sam El-Kamand | 7200 x 3600 | `aac710c8a364b6b27fecedf483c9122fc293238bb127030fa1d8fed6deae5b2e` | copied PNG | README clinical metadata example. |
| `goal_plot_13.png` | resolved | `/ggoncoplot/man/figures/ggoncoplot.pdf` | `ggoncoplot.pdf` | ggoncoplot GitHub man/figures | https://github.com/selkamand/ggoncoplot/tree/main/man/figures | ggoncoplot; MIT + file LICENSE; Sam El-Kamand | 125 x 144 | `7290f97f8778b5122c2077c6e18d97725acedc77bd8cedffea5b73711cceaa3f` | converted first page PDF to PNG with `sips` | Small vector package figure asset retained because it was in the requested figure import list. |
| `goal_plot_14.png` | resolved | `/ggoncoplot/man/figures/interactive_oncoplot.gif` | `interactive_oncoplot.gif` | ggoncoplot README/pkgdown interactive example | https://selkamand.github.io/ggoncoplot/ | ggoncoplot; MIT + file LICENSE; Sam El-Kamand | 1566 x 1036 | `15e434d36ed01e2ff5eb9d973c5f9b9acec4b69d93d339a4f2f08e68c31c7919` | converted representative GIF frame to PNG with `sips` | Static PNG snapshot of the interactive GIF. |
| `goal_plot_15.png` | resolved | `/ggoncoplot/paper/ggoncoplot_comparison.pdf` | `ggoncoplot_comparison.pdf` | ggoncoplot paper comparison figure | https://github.com/selkamand/ggoncoplot/tree/main/paper | ggoncoplot; MIT + file LICENSE; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 800 x 533 | `dac748c1a3e589f54c0f8367489d1d58060ae9ee39bab3c7b62b22b5c000819c` | converted first page PDF to PNG with `sips` | Paper comparison table/figure. |
| `goal_plot_16.png` | resolved | `/ggoncoplot/paper/jats/ggoncoplot_comparision.pdf` | `ggoncoplot_comparision.pdf` | ggoncoplot JATS paper assets | https://github.com/selkamand/ggoncoplot/tree/main/paper | ggoncoplot; MIT + file LICENSE; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 800 x 513 | `9214d0092d4ec0215f305e28eadb764c41adb85c3746583a32b791ec4ba88bb4` | converted first page PDF to PNG with `sips` | Kept because converted output is distinct from `goal_plot_15.png`. |
| `goal_plot_17.png` | resolved | `/ggoncoplot/paper/lasso_select.png` | `lasso_select.png` | ggoncoplot paper assets | https://github.com/selkamand/ggoncoplot/tree/main/paper | ggoncoplot; MIT + file LICENSE; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 1408 x 922 | `40a4764b79b62884dff4388a522008acf3c54a9869ea34959c8e813e61652d59` | copied PNG | Duplicate JATS copy skipped by SHA. |
| `goal_plot_18.png` | resolved | `/ggoncoplot/paper/multimodal_selection.old.png` | `multimodal_selection.old.png` | ggoncoplot paper assets | https://github.com/selkamand/ggoncoplot/tree/main/paper | ggoncoplot; MIT + file LICENSE; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 7620 x 5204 | `93de1d63c82e6d817bd6e254f560b740c7e7ec02cdec63018e501abc6c3e9b5e` | copied PNG | Historical version of multimodal selection figure. |
| `goal_plot_19.png` | resolved | `/ggoncoplot/paper/multimodal_selection.png` | `multimodal_selection.png` | ggoncoplot paper assets | https://github.com/selkamand/ggoncoplot/tree/main/paper | ggoncoplot; MIT + file LICENSE; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 7620 x 5204 | `f5db277e852c715f14e1be5293ae99da251511335661001ef4686568fecfff16` | copied PNG | Duplicate JATS copy skipped by SHA. |
| `goal_plot_20.png` | resolved | `/ggoncoplot/paper/multimodal_selection_with_lasso.png` | `multimodal_selection_with_lasso.png` | ggoncoplot JOSS paper Figure 2 | https://www.theoj.org/joss-papers/joss.07390/10.21105.joss.07390.pdf | ggoncoplot JOSS paper; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 2281 x 1520 | `d77ce72b70450f3180ecfa65a9643d13b9c12729d4e0710b1be3e1e250476b2b` | copied PNG | Duplicate JATS copy skipped by SHA. |
| `goal_plot_21.png` | resolved | `/ggoncoplot/paper/oncoplot_gbm.png` | `oncoplot_gbm.png` | ggoncoplot paper assets | https://github.com/selkamand/ggoncoplot/tree/main/paper | ggoncoplot; MIT + file LICENSE; Sam El-Kamand, Julian M. W. Quinn, Mark J. Cowley | 864 x 432 | `569397c0aab009dd92dabe077cac8de87d38e1ef3ee3c1807f3a1bd27726ac11` | copied PNG | GBM oncoplot paper asset. |

## Synthetic Inputs

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

## Regeneration Commands

Regenerate deterministic synthetic input data:

```bash
python3 python_refactor_goal_sources/generate_synthetic_inputs.py
```

Render clean generated plots:

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --style clean
```

Render BRCA reference-like plots:

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --style reference_like
```

Render a BRCA comparison sheet:

```bash
python3 python_refactor_goal_sources/recreate_gallery.py --style comparison --preset brca_large
```
