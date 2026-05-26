"""Build compact PyOncoplot fixtures from upstream fuc example data.

This script intentionally does not import fuc. It mirrors the relevant fuc
example data transformations so the gallery can render from committed TSV/JSON
fixtures without requiring fuc or large raw MAF/VCF files at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "syntheitic_goal_data"
GOAL_PLOTS = ROOT / "goal_plots"

NONSYN_NAMES = [
    "Missense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Splice_Site",
    "Translation_Start_Site",
]

NONSYN_COLORS = [
    "#2ca02c",
    "#1f77b4",
    "#9467bd",
    "#bcbd22",
    "#d62728",
    "#17becf",
    "#e377c2",
    "#ff7f0e",
    "#8c564b",
]

SOURCE_URLS = {
    "tcga_laml.maf.gz": "https://raw.githubusercontent.com/sbslee/fuc-data/main/tcga-laml/tcga_laml.maf.gz",
    "tcga_laml_annot.tsv": "https://raw.githubusercontent.com/sbslee/fuc-data/main/tcga-laml/tcga_laml_annot.tsv",
    "getrm-cyp2d6-vdr.vcf": "https://raw.githubusercontent.com/sbslee/fuc-data/main/pyvcf/getrm-cyp2d6-vdr.vcf",
}

SOURCE_SUBDIRS = {
    "tcga_laml.maf.gz": "tcga-laml",
    "tcga_laml_annot.tsv": "tcga-laml",
    "getrm-cyp2d6-vdr.vcf": "pyvcf",
}

SV_SAMPLE_MAP = {
    "NA18973": "NA18973",
    "NA10831": "HG00276",
    "NA19109": "NA19109",
}

SV_TITLES = {
    "NA18973": "NA18973 (no structural variation)",
    "NA10831": "NA10831 (CYP2D6 deletion)",
    "NA19109": "NA19109 (CYP2D6 duplication)",
}

CYP2D6_EXONS = [
    (42522500, 42522754),
    (42522852, 42522994),
    (42523448, 42523636),
    (42523843, 42523985),
    (42524175, 42524352),
    (42524785, 42524946),
    (42525034, 42525187),
    (42525739, 42525911),
    (42526613, 42526883),
]

CYP2D7_EXONS = [
    (42536213, 42536467),
    (42536565, 42536707),
    (42537161, 42537349),
    (42537543, 42537685),
    (42537877, 42538054),
    (42538479, 42538640),
    (42538728, 42538881),
    (42539410, 42539582),
    (42540284, 42540576),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file(source_dir: Path, name: str) -> Path:
    flat_path = source_dir / name
    if flat_path.exists():
        return flat_path
    nested_path = source_dir / SOURCE_SUBDIRS[name] / name
    if nested_path.exists():
        return nested_path
    raise FileNotFoundError(f"Could not find {name} in {source_dir} or {nested_path.parent}.")


def convert_num2cat(series: pd.Series, n: int = 5, decimals: int = 0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    boundaries = list(np.linspace(numeric.min(), numeric.max(), n + 1, endpoint=True))
    intervals = list(zip(boundaries[:-1], boundaries[1:]))

    def convert(value: float) -> float:
        if pd.isna(value):
            return np.nan
        for lower, upper in intervals:
            if lower <= value <= upper:
                return round(upper, decimals)
        return np.nan

    return numeric.apply(convert)


def waterfall_matrix(maf: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    nonsyn = maf[maf["Variant_Classification"].isin(NONSYN_NAMES)].copy()
    grouped = (
        nonsyn.groupby(["Hugo_Symbol", "Tumor_Sample_Barcode"])["Variant_Classification"]
        .apply(lambda values: values.iloc[0] if len(values) == 1 else "Multi_Hit")
        .reset_index()
    )
    matrix = grouped.pivot(
        index="Hugo_Symbol",
        columns="Tumor_Sample_Barcode",
        values="Variant_Classification",
    )
    row_order = matrix.isnull().sum(axis=1).sort_values(ascending=True).index
    matrix = matrix.reindex(index=row_order).head(count)
    matrix = matrix.dropna(axis=1, how="all")
    presence = matrix.notnull().astype(int)
    column_order = presence.sort_values(matrix.index.to_list(), axis=1, ascending=False).columns
    return matrix.reindex(columns=column_order).fillna("None").rename_axis(None, axis=1)


def sorted_annotation_order(metadata: pd.DataFrame, samples: Iterable[str]) -> list[str]:
    indexed = metadata.set_index("sample")
    return indexed.loc[list(samples)].sort_values(["FAB_classification", "Overall_Survival_Status"], kind="mergesort").index.astype(str).tolist()


def build_aml(source_dir: Path) -> dict[str, object]:
    maf_path = source_file(source_dir, "tcga_laml.maf.gz")
    annot_path = source_file(source_dir, "tcga_laml_annot.tsv")
    maf = pd.read_csv(maf_path, sep="\t", compression="gzip")
    annot = pd.read_csv(annot_path, sep="\t")
    annot = annot.rename(columns={"Tumor_Sample_Barcode": "sample"})
    annot["FAB_classification"] = annot["FAB_classification"].astype(str)
    annot["Overall_Survival_Status"] = annot["Overall_Survival_Status"].astype(str)
    annot["days_to_last_followup"] = convert_num2cat(annot["days_to_last_followup"])

    nonsyn = maf[maf["Variant_Classification"].isin(NONSYN_NAMES)].copy()
    mutations = pd.DataFrame(
        {
            "sample": nonsyn["Tumor_Sample_Barcode"].astype(str),
            "gene": nonsyn["Hugo_Symbol"].astype(str),
            "mutation_type": nonsyn["Variant_Classification"].astype(str),
            "tooltip": (
                nonsyn["Hugo_Symbol"].astype(str)
                + " "
                + nonsyn["Variant_Classification"].astype(str)
                + " "
                + nonsyn["Protein_Change"].fillna("").astype(str)
            ),
            "cluster": "tcga-laml",
        }
    )
    mutations.to_csv(INPUTS / "aml_mutations.tsv", sep="\t", index=False)

    tmb = (
        mutations.groupby(["sample", "mutation_type"], as_index=False)
        .size()
        .rename(columns={"size": "mutations"})
        .sort_values(["sample", "mutation_type"], kind="mergesort")
    )
    tmb.to_csv(INPUTS / "aml_tmb.tsv", sep="\t", index=False)

    full_matrix = waterfall_matrix(maf)
    full_samples = full_matrix.columns.astype(str).tolist()
    survival_samples = annot.loc[annot["Overall_Survival_Status"] == "1", "sample"].astype(str).tolist()
    survival_matrix = waterfall_matrix(maf[maf["Tumor_Sample_Barcode"].isin(survival_samples)])
    survival_order = survival_matrix.columns.astype(str).tolist()
    sorted_order = sorted_annotation_order(annot, full_samples)

    order_maps = {
        "waterfall_order": {sample: index + 1 for index, sample in enumerate(full_samples)},
        "sorted_order": {sample: index + 1 for index, sample in enumerate(sorted_order)},
        "survival_filtered_order": {sample: index + 1 for index, sample in enumerate(survival_order)},
    }
    for column, mapping in order_maps.items():
        annot[column] = annot["sample"].map(mapping)
    annot[["waterfall_order", "sorted_order", "survival_filtered_order"]] = annot[
        ["waterfall_order", "sorted_order", "survival_filtered_order"]
    ].fillna("NA")
    annot.to_csv(INPUTS / "aml_metadata.tsv", sep="\t", index=False)

    palette = dict(zip(NONSYN_NAMES, NONSYN_COLORS))
    palette["Multi_Hit"] = "#000000"
    (INPUTS / "aml_palette.json").write_text(json.dumps(palette, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    params = {
        "full_top_genes": full_matrix.index.astype(str).tolist(),
        "survival_top_genes": survival_matrix.index.astype(str).tolist(),
        "waterfall_sample_order": full_samples,
        "sorted_sample_order": sorted_order,
        "survival_filtered_sample_order": survival_order,
    }
    (INPUTS / "aml_gallery_params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    return {
        "raw_rows": int(len(maf)),
        "nonsyn_rows": int(len(mutations)),
        "samples": int(annot["sample"].nunique()),
        "full_top_genes": params["full_top_genes"],
        "survival_top_genes": params["survival_top_genes"],
    }


def parse_sample(format_keys: list[str], value: str) -> dict[str, str]:
    parts = value.split(":")
    return {key: parts[index] if index < len(parts) else "." for index, key in enumerate(format_keys)}


def numeric_format_value(parsed: dict[str, str], key: str) -> float:
    value = parsed.get(key, ".")
    if value in {"", "."}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def allele_fraction(parsed: dict[str, str], allele: str) -> float:
    value = parsed.get("AD", ".")
    if value in {"", "."}:
        return np.nan
    try:
        depths = [int(part) for part in value.split(",")]
    except ValueError:
        return np.nan
    total = sum(depths)
    if total == 0:
        return np.nan
    if allele == "REF":
        return depths[0] / total
    return sum(depths[1:]) / total


def build_sv(source_dir: Path) -> dict[str, object]:
    vcf_path = source_file(source_dir, "getrm-cyp2d6-vdr.vcf")
    header: list[str] | None = None
    depth_rows = []
    allele_rows = []
    with vcf_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.lstrip("#").split("\t")
                continue
            if not line:
                continue
            if header is None:
                raise ValueError("VCF header was not found.")
            row = dict(zip(header, line.split("\t")))
            format_keys = row["FORMAT"].split(":")
            position = int(row["POS"])
            for display_sample, source_sample in SV_SAMPLE_MAP.items():
                parsed = parse_sample(format_keys, row[source_sample])
                depth_rows.append(
                    {
                        "sample": display_sample,
                        "source_sample": source_sample,
                        "title": SV_TITLES[display_sample],
                        "position": position,
                        "depth": numeric_format_value(parsed, "DP"),
                    }
                )
                for allele in ("REF", "ALT"):
                    allele_rows.append(
                        {
                            "sample": display_sample,
                            "source_sample": source_sample,
                            "position": position,
                            "allele": allele,
                            "allele_fraction": allele_fraction(parsed, allele),
                        }
                    )

    pd.DataFrame(depth_rows).to_csv(INPUTS / "sv_depth.tsv", sep="\t", index=False)
    pd.DataFrame(allele_rows).to_csv(INPUTS / "sv_allele_fraction.tsv", sep="\t", index=False)

    gene_rows = []
    for gene, exons in {"CYP2D6": CYP2D6_EXONS, "CYP2D7": CYP2D7_EXONS}.items():
        for index, (start, end) in enumerate(exons, start=1):
            gene_rows.append({"gene": gene, "exon": index, "start": start, "end": end, "strand": "-"})
    pd.DataFrame(gene_rows).to_csv(INPUTS / "sv_gene_models.tsv", sep="\t", index=False)
    return {"positions": int(len(depth_rows) / len(SV_SAMPLE_MAP)), "samples": list(SV_SAMPLE_MAP)}


def fixture_checksums() -> dict[str, str]:
    names = [
        "aml_mutations.tsv",
        "aml_metadata.tsv",
        "aml_tmb.tsv",
        "aml_palette.json",
        "aml_gallery_params.json",
        "sv_depth.tsv",
        "sv_allele_fraction.tsv",
        "sv_gene_models.tsv",
    ]
    return {name: sha256(INPUTS / name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / "fuc-data",
        help="Directory containing tcga_laml.maf.gz, tcga_laml_annot.tsv, and getrm-cyp2d6-vdr.vcf.",
    )
    args = parser.parse_args()
    INPUTS.mkdir(exist_ok=True)
    source_dir = args.source_dir.expanduser()
    aml = build_aml(source_dir)
    sv = build_sv(source_dir)
    manifest = {
        "source_urls": SOURCE_URLS,
        "source_checksums": {
            name: sha256(source_file(source_dir, name))
            for name in ["tcga_laml.maf.gz", "tcga_laml_annot.tsv", "getrm-cyp2d6-vdr.vcf"]
        },
        "source_scripts": {
            "goal_plot_18.png": "oncoplot.py",
            "goal_plot_19.png": "customized_oncoplot_1.py",
            "goal_plot_20.png": "customized_oncoplot_2.py",
            "goal_plot_21.png": "vcf_sv.py",
            "goal_plot_22.png": "customized_oncoplot_3.py",
        },
        "goal_plot_22_sha256": sha256(GOAL_PLOTS / "goal_plot_22.png") if (GOAL_PLOTS / "goal_plot_22.png").exists() else None,
        "derived_fixture_checksums": fixture_checksums(),
        "summary": {"aml": aml, "sv": sv},
    }
    (ROOT / "fuc_sources" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
