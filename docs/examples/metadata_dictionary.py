"""Synthetic, dictionary-only cohort filtering and source styles; no axes edits.
Run: PYTHONPATH=src .venv/bin/python docs/examples/metadata_dictionary.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyoncoplot import oncoplot


def dictionary_example():
    samples = [f"S{i:02}" for i in range(12)]
    data = pd.DataFrame({"sample": samples + samples[:5],
                         "gene": ["TP53", "KRAS", "PIK3CA"] * 4 + ["APC"] * 5,
                         "event": ["Missense"] * 12 + ["Truncating"] * 5})
    metadata = pd.DataFrame({"sample": samples, "treatment": list("ABC") * 4,
                             "response": list("AB") * 6, "age": list(range(30, 90, 5))})
    return oncoplot(params={
        "data": data, "gene_col": "gene", "sample_col": "sample", "mutation_type_col": "event",
        "metadata": metadata, "metadata_cols": ["treatment", "response", "age"],
        "metadata_filter_mode": "analysis_cohort", "metadata_filters": {"age": {"min": 30, "max": 75}},
        "backend": "matplotlib", "draw_tmb_bar": True, "draw_gene_bar": True,
        "options": {"compact_layout": True, "width": 1500, "height": 850,
                    "font_family": "DejaVu Sans", "show_sample_ids": True,
                    "metadata_legend_position": "bottom", "log10_transform_tmb": False,
                    "title_text": "Synthetic analysis cohort: ages 30–75",
                    "metadata_styles": {
                        "treatment": {"legend": {"ncol": 2, "alignment": "top"},
                                      "labels": {"A": "Treatment A", "B": "Treatment B", "C": "Treatment C"}},
                        "response": {"legend": {"ncol": 1}, "labels": {"A": "Response", "B": "Stable"}}}}})


def main():
    output = Path(".venv/dictionary-api")
    output.mkdir(parents=True, exist_ok=True)
    result = dictionary_example()
    try:
        for suffix in ("png", "pdf", "svg"):
            path = output / ("metadata-dictionary." + suffix)
            result.save(path, dpi=150, bbox_inches=None)
            print(path)
    finally:
        plt.close(result.figure)


if __name__ == "__main__":
    main()
