# Metadata Example

This example uses the deterministic AML synthetic inputs.

```python
import json
from pathlib import Path

import pandas as pd

from pyoncoplot import OncoplotOptions, oncoplot

root = Path("python_refactor_goal_sources/syntheitic_goal_data")
mutations = pd.read_csv(root / "aml_mutations.tsv", sep="\t")
metadata = pd.read_csv(root / "aml_metadata.tsv", sep="\t")
tmb = pd.read_csv(root / "aml_tmb.tsv", sep="\t")
palette = json.loads((root / "aml_palette.json").read_text())

metadata["FAB_classification"] = metadata["FAB_classification"].astype(str)
metadata["Overall_Survival_Status"] = metadata["Overall_Survival_Status"].astype(str)

metadata_palette = {
    "FAB_classification": {
        "M0": "#1B9E77",
        "M1": "#D95F02",
        "M2": "#7570B3",
        "M3": "#E7298A",
        "M4": "#66A61E",
        "M5": "#E6AB02",
        "M6": "#A6761D",
        "M7": "#666666",
    },
    "Overall_Survival_Status": {"0": "#FDB7B4", "1": "#BBD7EA"},
}

result = oncoplot(
    mutations,
    gene_col="gene",
    sample_col="sample",
    mutation_type_col="mutation_type",
    tooltip_col="tooltip",
    include_genes=["FLT3", "DNMT3A", "NPM1", "IDH2", "IDH1", "TET2", "RUNX1", "NRAS", "TP53", "CEBPA"],
    draw_gene_bar=True,
    draw_tmb_bar=True,
    palette=palette,
    tmb_data=tmb,
    tmb_palette=palette,
    metadata=metadata,
    metadata_cols=["FAB_classification", "days_to_last_followup", "Overall_Survival_Status"],
    metadata_palette=metadata_palette,
    show_all_samples=True,
    backend="matplotlib",
    options=OncoplotOptions(
        width=1080,
        height=720,
        log10_transform_tmb=False,
        metadata_numeric_plot_type="bar",
        mutation_legend_position="bottom",
        metadata_legend_position="right",
    ),
)
result.save("aml_metadata.png", dpi=120)
```

To sort by FAB classification, add:

```python
metadata_sort_cols=["FAB_classification"],
metadata_sort_by="alphabetical",
metadata_sort_desc=False,
```

