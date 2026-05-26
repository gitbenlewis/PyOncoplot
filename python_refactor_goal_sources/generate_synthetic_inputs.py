"""Generate deterministic synthetic inputs for non-fuc gallery examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "syntheitic_goal_data"

MUTATION_PALETTE = {
    "Missense_Mutation": "#2CA02C",
    "Frame_Shift_Del": "#1F77B4",
    "Frame_Shift_Ins": "#9467BD",
    "In_Frame_Del": "#BCBD22",
    "In_Frame_Ins": "#D62728",
    "Nonsense_Mutation": "#17BECF",
    "Nonstop_Mutation": "#E377C2",
    "Splice_Site": "#FF7F0E",
    "Translation_Start_Site": "#8C564B",
    "Silent": "#F2C16B",
    "3'UTR": "#F2C16B",
    "5'UTR": "#F2C16B",
    "Intron": "#F2C16B",
    "Multi_Hit": "#000000",
}

AML_GENES = ["FLT3", "DNMT3A", "NPM1", "IDH2", "IDH1", "TET2", "RUNX1", "NRAS", "TP53", "CEBPA"]
AML_TYPES = [
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

BRCA_GENES = [
    "PIK3CA",
    "TP53",
    "CDH1",
    "GATA3",
    "KMT2C",
    "MAP3K1",
    "SPTA1",
    "ZFHX4",
    "PTEN",
    "ABCA13",
    "BRCA2",
    "FOXA1",
    "AKT1",
    "NF1",
    "NCOR1",
    "MYC",
    "CCND1",
    "ERBB2",
    "EGFR",
    "FGFR1",
    "RB1",
    "BRAF",
    "BRCA1",
    "PIK3R1",
    "FGFR2",
    "MAP2K4",
]
BRCA_TYPES = [
    "Missense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Nonsense_Mutation",
    "Splice_Site",
    "Silent",
    "3'UTR",
    "5'UTR",
    "Intron",
]

CSSC_SAMPLES = [
    "CSSC_0126-M1",
    "CSSC_0014-M1",
    "CSSC_0001-M1",
    "CSSC_0134-M1",
    "CSSC_0007-M1",
    "CSSC_0066-M1",
    "CSSC_0012-M1",
    "CSSC_0125-M1",
    "CSSC_0022-M1",
    "CSSC_0130-M1",
    "CSSC_0013-M1",
    "CSSC_0133-M1",
    "CSSC_0011-M1",
    "CSSC_0005-M1",
    "CSSC_0132-M1",
    "CSSC_0024-M1",
    "CSSC_0124-M1",
    "CSSC_0025-M1",
    "CSSC_0006-M1",
    "CSSC_0010-M1",
    "CSSC_0135-M1",
    "CSSC_0004-M1",
    "CSSC_0003-M1",
    "CSSC_0002-M1",
    "CSSC_0009-M1",
]
CSSC_GENES = ["TP53", "CDKN2A", "C9", "KHDRBS2", "SLC22A6", "COLEC12", "LINGO2", "CDHR5", "ZNF442", "PRLR", "DHRS4"]
CSSC_TYPES = [
    "missense_variant",
    "synonymous_variant",
    "stop_gained",
    "complex_substitution",
    "splice_site_variant",
    "frameshift_truncation",
    "inframe_deletion",
]
CSSC_PALETTE = {
    "missense_variant": "#F5A000",
    "synonymous_variant": "#8E2F17",
    "stop_gained": "#F20D0D",
    "complex_substitution": "#1E3CFF",
    "splice_site_variant": "#19E33A",
    "frameshift_truncation": "#000000",
    "inframe_deletion": "#62D7F2",
}

GBM_GENES = [
    "TERT",
    "PDGFRA",
    "TSC2",
    "PIK3CA",
    "PIK3R1",
    "TP53",
    "TP53BP1",
    "EGFR",
    "ERBB3",
    "NF1",
    "PTEN",
    "KMT2C",
    "MTOR",
    "ARID2",
    "SETD2",
    "ATM",
    "BRCA2",
    "BRCA1",
    "MSH2",
    "RB1",
    "JAK2",
    "APC",
    "AXIN2",
    "HM",
    "MGMT meth",
]
GBM_TRACKS = [
    "Recurrence",
    "Primary",
    "Sex",
    "Age above 50",
    "KPS 70 or above",
    "Deceased",
    "Treatment: RT",
    "Treatment: TMZ",
    "Treatment: Beva",
    "GITS subtype R2",
    "GITS subtype R1",
    "CN status MDM2",
    "CN status CDK4",
    "CN status NF1",
    "CN status RB1",
    "CN status EGFR",
]
GBM_PALETTE = {
    "Classical": "#4F8DC3",
    "Mesenchymal": "#F0C400",
    "Proneural": "#EF5A6E",
    "No / wildtype / biopsy": "#FFFFFF",
    "Yes / stable / resection": "#1E355C",
    "NA / Beva: random trial": "#AFAFAF",
    "Gained / increased / female": "#BE5362",
    "Lost / decreased / male": "#70B7B4",
}

README_GENES = ["PIK3CA", "TP53", "CDH1", "GATA3", "MAP3K1", "PTEN", "KMT2C", "NF1", "ERBB2", "BRCA1", "RB1", "EGFR"]
README_TYPES = ["missense", "nonsense", "frameshift", "splice", "inframe", "silent"]
README_PALETTE = {
    "missense": "#4DAF4A",
    "nonsense": "#E41A1C",
    "frameshift": "#377EB8",
    "splice": "#FF7F00",
    "inframe": "#984EA3",
    "silent": "#BDBDBD",
    "Multi_Hit": "#111111",
}

PAPER_MULTIMODAL_GENES = ["PIK3CA", "TP53", "CDH1", "GATA3", "MAP3K1", "PTEN", "NF1", "BRCA1", "RB1", "EGFR"]
PAPER_MULTIMODAL_TYPES = ["Mutation", "Amplification", "Deletion", "Fusion"]
PAPER_MULTIMODAL_PALETTE = {
    "Mutation": "#244A7F",
    "Amplification": "#B74A5A",
    "Deletion": "#5DA8A3",
    "Fusion": "#D1A12B",
    "Triple Negative": "#111111",
    "Not Triple Negative": "#E43D30",
    "Ambiguous": "#BDBDBD",
    "Selected": "#E43D30",
    "Unselected": "#BDBDBD",
}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_type(rng: np.random.Generator, types: list[str], weights: list[float]) -> str:
    return str(rng.choice(types, p=np.array(weights) / np.sum(weights)))


def _tmb_from_mutations(mutations: pd.DataFrame, sample_col: str = "sample") -> pd.DataFrame:
    base = (
        mutations.groupby([sample_col, "mutation_type"], dropna=False)
        .size()
        .rename("mutations")
        .reset_index()
    )
    return base.sort_values([sample_col, "mutation_type"]).reset_index(drop=True)


def generate_aml() -> None:
    rng = np.random.default_rng(101)
    samples = [f"AML-{i:03d}" for i in range(1, 136)]
    block_names = ["FLT3/NPM1", "DNMT3A/TET2", "IDH", "TP53/CEBPA"]
    blocks = {sample: block_names[min(len(block_names) - 1, i // 34)] for i, sample in enumerate(samples)}
    rows = []
    base_probs = {
        "FLT3": 0.36,
        "DNMT3A": 0.31,
        "NPM1": 0.25,
        "IDH2": 0.15,
        "IDH1": 0.13,
        "TET2": 0.11,
        "RUNX1": 0.08,
        "NRAS": 0.075,
        "TP53": 0.06,
        "CEBPA": 0.055,
    }
    boosts = {
        "FLT3/NPM1": {"FLT3": 1.8, "NPM1": 1.8, "DNMT3A": 1.25},
        "DNMT3A/TET2": {"DNMT3A": 1.75, "TET2": 1.8, "RUNX1": 1.4},
        "IDH": {"IDH1": 2.0, "IDH2": 2.0, "NRAS": 1.25},
        "TP53/CEBPA": {"TP53": 2.2, "CEBPA": 2.1, "RUNX1": 1.35},
    }
    for sample in samples:
        block = blocks[sample]
        for gene in AML_GENES:
            probability = min(0.92, base_probs[gene] * boosts[block].get(gene, 0.85))
            if rng.random() < probability:
                hits = 2 if rng.random() < (0.13 if gene in {"FLT3", "DNMT3A", "TP53"} else 0.06) else 1
                for hit in range(hits):
                    mutation_type = _sample_type(rng, AML_TYPES, [0.52, 0.08, 0.04, 0.04, 0.03, 0.13, 0.02, 0.12, 0.02])
                    rows.append(
                        {
                            "sample": sample,
                            "gene": gene,
                            "mutation_type": mutation_type,
                            "tooltip": f"{sample} {gene} {mutation_type} hit {hit + 1}",
                            "cluster": block,
                        }
                    )
    mutations = pd.DataFrame(rows)
    fab_by_block = {
        "FLT3/NPM1": ["M1", "M2", "M4"],
        "DNMT3A/TET2": ["M0", "M5", "M7"],
        "IDH": ["M2", "M3", "M6"],
        "TP53/CEBPA": ["M4", "M5", "M7"],
    }
    metadata = pd.DataFrame(
        {
            "sample": samples,
            "cluster": [blocks[sample] for sample in samples],
            "FAB_classification": [str(rng.choice(fab_by_block[blocks[sample]])) for sample in samples],
            "days_to_last_followup": rng.choice([572, 1144, 1717, 2289, 2861], size=len(samples)),
            "Overall_Survival_Status": rng.choice([0, 1], p=[0.58, 0.42], size=len(samples)),
        }
    )
    tmb = _tmb_from_mutations(mutations)
    extra_rows = []
    for sample in samples:
        for mutation_type in rng.choice(AML_TYPES, size=rng.integers(1, 4), replace=False):
            extra_rows.append({"sample": sample, "mutation_type": mutation_type, "mutations": int(rng.integers(1, 7))})
    tmb = pd.concat([tmb, pd.DataFrame(extra_rows)], ignore_index=True)
    tmb = tmb.groupby(["sample", "mutation_type"], as_index=False)["mutations"].sum()

    mutations.to_csv(OUT / "aml_mutations.tsv", sep="\t", index=False)
    metadata.to_csv(OUT / "aml_metadata.tsv", sep="\t", index=False)
    tmb.to_csv(OUT / "aml_tmb.tsv", sep="\t", index=False)
    _write_json(OUT / "aml_palette.json", {key: MUTATION_PALETTE[key] for key in AML_TYPES + ["Multi_Hit"]})


def generate_brca() -> None:
    rng = np.random.default_rng(202)
    samples = [f"BRCA-{i:04d}" for i in range(1, 421)]
    subtypes = ["HR+/HER2-", "HR-/HER2+", "HR+/HER2+", "HR-/HER2-"]
    subtype_blocks = np.repeat(subtypes, [190, 70, 55, 105])
    rng.shuffle(subtype_blocks)
    base_probs = {
        "PIK3CA": 0.37,
        "TP53": 0.34,
        "CDH1": 0.14,
        "GATA3": 0.13,
        "KMT2C": 0.10,
        "MAP3K1": 0.09,
        "SPTA1": 0.07,
        "ZFHX4": 0.06,
        "PTEN": 0.06,
        "ABCA13": 0.055,
        "BRCA2": 0.045,
        "FOXA1": 0.045,
        "AKT1": 0.04,
        "NF1": 0.04,
        "NCOR1": 0.035,
        "MYC": 0.032,
        "CCND1": 0.03,
        "ERBB2": 0.026,
        "EGFR": 0.025,
        "FGFR1": 0.024,
        "RB1": 0.024,
        "BRAF": 0.022,
        "BRCA1": 0.02,
        "PIK3R1": 0.02,
        "FGFR2": 0.018,
        "MAP2K4": 0.018,
    }
    rows = []
    for sample, subtype in zip(samples, subtype_blocks):
        for gene in BRCA_GENES:
            probability = base_probs[gene]
            if subtype == "HR-/HER2-" and gene in {"TP53", "BRCA1", "RB1"}:
                probability *= 1.9
            if subtype == "HR+/HER2-" and gene in {"PIK3CA", "GATA3", "MAP3K1", "CDH1"}:
                probability *= 1.4
            if subtype in {"HR-/HER2+", "HR+/HER2+"} and gene in {"ERBB2", "MYC", "CCND1", "EGFR", "FGFR1"}:
                probability *= 2.2
            if rng.random() < min(probability, 0.85):
                hits = 2 if rng.random() < 0.05 else 1
                for hit in range(hits):
                    mutation_type = _sample_type(rng, BRCA_TYPES, [0.58, 0.07, 0.035, 0.12, 0.08, 0.04, 0.025, 0.015, 0.035])
                    rows.append(
                        {
                            "sample": sample,
                            "gene": gene,
                            "mutation_type": mutation_type,
                            "tooltip": f"{sample} {gene} {mutation_type} hit {hit + 1}",
                            "subtype": subtype,
                        }
                    )
    mutations = pd.DataFrame(rows)
    metadata = pd.DataFrame({"sample": samples, "Subtype": subtype_blocks})
    metadata["ER_status"] = np.where(metadata["Subtype"].str.startswith("HR+"), "Positive", "Negative")
    metadata["PR_status"] = np.where(metadata["Subtype"].str.startswith("HR+"), rng.choice(["Positive", "Negative"], size=len(samples), p=[0.72, 0.28]), "Negative")
    metadata["HER2_status"] = np.where(metadata["Subtype"].str.contains("HER2\\+"), "Positive", "Negative")
    metadata["Age"] = rng.choice(["<=50", ">50"], size=len(samples), p=[0.38, 0.62])
    metadata["Age_years"] = rng.integers(28, 91, size=len(samples))
    metadata["Menopause"] = np.where(metadata["Age"] == ">50", "Postmenopausal", rng.choice(["Premenopausal", "Postmenopausal"], size=len(samples), p=[0.75, 0.25]))
    metadata["LN_stage"] = rng.choice(["Positive", "Negative"], size=len(samples), p=[0.42, 0.58])
    metadata["Grade"] = rng.choice(["I", "II", "III", "Unknown"], size=len(samples), p=[0.12, 0.38, 0.42, 0.08])
    metadata["TNM_stage"] = rng.choice(["I", "II", "III", "IV"], size=len(samples), p=[0.24, 0.46, 0.25, 0.05])
    metadata["Histological_type"] = rng.choice(
        ["Infiltrating Ductal Carcinoma", "Infiltrating Lobular Carcinoma", "Others"],
        size=len(samples),
        p=[0.72, 0.16, 0.12],
    )
    metadata["Classification"] = np.where(
        metadata["Subtype"] == "HR-/HER2-",
        "Triple Negative",
        np.where(metadata["Subtype"] == "HR+/HER2-", "Not Triple Negative", "Ambiguous"),
    )
    tmb = _tmb_from_mutations(mutations)
    for sample in samples:
        if rng.random() < 0.65:
            extra = pd.DataFrame(
                {
                    "sample": [sample],
                    "mutation_type": [str(rng.choice(BRCA_TYPES))],
                    "mutations": [int(rng.integers(3, 25))],
                }
            )
            tmb = pd.concat([tmb, extra], ignore_index=True)
    tmb = tmb.groupby(["sample", "mutation_type"], as_index=False)["mutations"].sum()

    mutations.to_csv(OUT / "brca_mutations.tsv", sep="\t", index=False)
    metadata.to_csv(OUT / "brca_metadata.tsv", sep="\t", index=False)
    tmb.to_csv(OUT / "brca_tmb.tsv", sep="\t", index=False)
    _write_json(OUT / "brca_palette.json", {key: MUTATION_PALETTE[key] for key in BRCA_TYPES + ["Multi_Hit"]})


def generate_sv() -> None:
    rng = np.random.default_rng(303)
    samples = ["NA18973", "NA10831", "NA19109"]
    titles = {
        "NA18973": "NA18973 (no structural variation)",
        "NA10831": "NA10831 (CYP2D6 deletion)",
        "NA19109": "NA19109 (CYP2D6 duplication)",
    }
    positions = np.linspace(42_510_000, 42_555_000, 260, dtype=int)
    depth_rows = []
    allele_rows = []
    for sample in samples:
        depth = rng.normal(40, 7, size=len(positions))
        cyp2d6 = (positions >= 42_520_400) & (positions <= 42_526_900)
        if sample == "NA10831":
            depth[cyp2d6] *= 0.45
        if sample == "NA19109":
            depth[cyp2d6] *= 1.9
        for position, value in zip(positions, depth):
            depth_rows.append(
                {
                    "sample": sample,
                    "title": titles[sample],
                    "position": int(position),
                    "depth": round(float(max(1, value)), 3),
                }
            )
        ref = np.clip(rng.beta(12, 1.25, size=len(positions)), 0, 1)
        alt = np.clip(rng.beta(0.7, 9.5, size=len(positions)), 0, 1)
        het = rng.choice(len(positions), size=46, replace=False)
        ref[het] = np.clip(rng.normal(0.50, 0.11, size=len(het)), 0, 1)
        alt[het] = np.clip(rng.normal(0.50, 0.11, size=len(het)), 0, 1)
        for position, ref_value, alt_value in zip(positions, ref, alt):
            allele_rows.append({"sample": sample, "position": int(position), "allele": "REF", "allele_fraction": round(float(ref_value), 4)})
            allele_rows.append({"sample": sample, "position": int(position), "allele": "ALT", "allele_fraction": round(float(alt_value), 4)})
    gene_models = pd.DataFrame(
        [
            {"gene": "CYP2D6", "start": 42_520_400, "end": 42_526_900, "strand": "-"},
            {"gene": "CYP2D7", "start": 42_533_700, "end": 42_540_000, "strand": "-"},
        ]
    )
    pd.DataFrame(depth_rows).to_csv(OUT / "sv_depth.tsv", sep="\t", index=False)
    pd.DataFrame(allele_rows).to_csv(OUT / "sv_allele_fraction.tsv", sep="\t", index=False)
    gene_models.to_csv(OUT / "sv_gene_models.tsv", sep="\t", index=False)


def generate_cssc() -> None:
    rng = np.random.default_rng(404)
    target_counts = {
        "TP53": 24,
        "CDKN2A": 20,
        "C9": 13,
        "KHDRBS2": 12,
        "SLC22A6": 11,
        "COLEC12": 11,
        "LINGO2": 10,
        "CDHR5": 8,
        "ZNF442": 8,
        "PRLR": 7,
        "DHRS4": 5,
    }
    type_weights = np.array([0.50, 0.09, 0.18, 0.09, 0.03, 0.08, 0.03])
    type_weights = type_weights / type_weights.sum()
    rows = []
    for gene_index, gene in enumerate(CSSC_GENES):
        sample_order = CSSC_SAMPLES[gene_index:] + CSSC_SAMPLES[:gene_index]
        selected = sample_order[: target_counts[gene]]
        rng.shuffle(selected)
        for sample_index, sample in enumerate(selected):
            alteration = str(rng.choice(CSSC_TYPES, p=type_weights))
            rows.append(
                {
                    "sample": sample,
                    "gene": gene,
                    "alteration": alteration,
                    "tooltip": f"{sample} {gene} {alteration}",
                }
            )
            if (gene_index + sample_index) % 8 == 0:
                second = str(rng.choice(["complex_substitution", "synonymous_variant", "frameshift_truncation", "inframe_deletion"]))
                rows.append(
                    {
                        "sample": sample,
                        "gene": gene,
                        "alteration": second,
                        "tooltip": f"{sample} {gene} {second} secondary",
                    }
                )

    mutations = pd.DataFrame(rows)
    tmb = _tmb_from_mutations(mutations.rename(columns={"alteration": "mutation_type"})).rename(columns={"mutation_type": "alteration"})
    extra_rows = []
    for sample in CSSC_SAMPLES:
        for alteration in rng.choice(CSSC_TYPES, size=rng.integers(1, 4), replace=False):
            extra_rows.append({"sample": sample, "alteration": alteration, "mutations": int(rng.integers(1, 5))})
    tmb = pd.concat([tmb, pd.DataFrame(extra_rows)], ignore_index=True)
    tmb = tmb.groupby(["sample", "alteration"], as_index=False)["mutations"].sum()

    mutations.to_csv(OUT / "cssc_mutations.tsv", sep="\t", index=False)
    tmb.to_csv(OUT / "cssc_tmb.tsv", sep="\t", index=False)
    _write_json(OUT / "cssc_palette.json", CSSC_PALETTE)


def generate_gbm() -> None:
    rng = np.random.default_rng(505)
    samples = [f"G-SAM-{i:03d}" for i in range(1, 148)]
    subtype_values = ["Classical", "Mesenchymal", "Proneural"]
    gits_r1 = rng.choice(subtype_values, size=len(samples), p=[0.34, 0.36, 0.30])
    gits_r2 = np.where(rng.random(len(samples)) < 0.18, "NA / Beva: random trial", gits_r1)
    recurrence = rng.choice(["Yes / stable / resection", "No / wildtype / biopsy"], size=len(samples), p=[0.83, 0.17])
    tracks = pd.DataFrame(
        {
            "sample": samples,
            "Recurrence": recurrence,
            "Primary": np.where(recurrence == "Yes / stable / resection", "No / wildtype / biopsy", "Yes / stable / resection"),
            "Sex": rng.choice(["Gained / increased / female", "Lost / decreased / male"], size=len(samples), p=[0.45, 0.55]),
            "Age above 50": rng.choice(["Yes / stable / resection", "No / wildtype / biopsy"], size=len(samples), p=[0.62, 0.38]),
            "KPS 70 or above": rng.choice(["Yes / stable / resection", "No / wildtype / biopsy"], size=len(samples), p=[0.74, 0.26]),
            "Deceased": rng.choice(["Yes / stable / resection", "No / wildtype / biopsy"], size=len(samples), p=[0.55, 0.45]),
            "Treatment: RT": rng.choice(["Yes / stable / resection", "No / wildtype / biopsy"], size=len(samples), p=[0.86, 0.14]),
            "Treatment: TMZ": rng.choice(["Yes / stable / resection", "No / wildtype / biopsy"], size=len(samples), p=[0.78, 0.22]),
            "Treatment: Beva": rng.choice(["NA / Beva: random trial", "Yes / stable / resection", "No / wildtype / biopsy"], size=len(samples), p=[0.55, 0.22, 0.23]),
            "GITS subtype R2": gits_r2,
            "GITS subtype R1": gits_r1,
            "CN status MDM2": rng.choice(["Gained / increased / female", "No / wildtype / biopsy"], size=len(samples), p=[0.10, 0.90]),
            "CN status CDK4": rng.choice(["Gained / increased / female", "No / wildtype / biopsy"], size=len(samples), p=[0.11, 0.89]),
            "CN status NF1": rng.choice(["Lost / decreased / male", "No / wildtype / biopsy"], size=len(samples), p=[0.16, 0.84]),
            "CN status RB1": rng.choice(["Lost / decreased / male", "No / wildtype / biopsy"], size=len(samples), p=[0.13, 0.87]),
            "CN status EGFR": rng.choice(["Gained / increased / female", "No / wildtype / biopsy"], size=len(samples), p=[0.22, 0.78]),
        }
    )

    gene_probs = {
        "TERT": 0.52,
        "PDGFRA": 0.28,
        "TSC2": 0.13,
        "PIK3CA": 0.16,
        "PIK3R1": 0.19,
        "TP53": 0.23,
        "TP53BP1": 0.08,
        "EGFR": 0.30,
        "ERBB3": 0.06,
        "NF1": 0.16,
        "PTEN": 0.22,
        "KMT2C": 0.12,
        "MTOR": 0.09,
        "ARID2": 0.07,
        "SETD2": 0.06,
        "ATM": 0.08,
        "BRCA2": 0.06,
        "BRCA1": 0.05,
        "MSH2": 0.06,
        "RB1": 0.08,
        "JAK2": 0.04,
        "APC": 0.05,
        "AXIN2": 0.04,
        "HM": 0.10,
        "MGMT meth": 0.18,
    }
    event_types = ["Mutation", "Gained / increased / female", "Lost / decreased / male"]
    event_weights = np.array([0.70, 0.16, 0.14])
    event_rows = []
    for sample, subtype in zip(samples, gits_r1):
        for gene in GBM_GENES:
            probability = gene_probs[gene]
            if subtype == "Proneural" and gene in {"TP53", "PIK3CA", "PIK3R1"}:
                probability *= 1.45
            if subtype == "Classical" and gene in {"EGFR", "PTEN", "PDGFRA"}:
                probability *= 1.45
            if subtype == "Mesenchymal" and gene in {"NF1", "TERT", "KMT2C"}:
                probability *= 1.35
            if rng.random() < min(probability, 0.82):
                event = str(rng.choice(event_types, p=event_weights))
                event_rows.append({"sample": sample, "gene": gene, "alteration": event, "tooltip": f"{sample} {gene} {event}"})
                if rng.random() < 0.07:
                    second = "Gained / increased / female" if event != "Gained / increased / female" else "Mutation"
                    event_rows.append({"sample": sample, "gene": gene, "alteration": second, "tooltip": f"{sample} {gene} {second} secondary"})

    tracks.to_csv(OUT / "gbm_clinical_tracks.tsv", sep="\t", index=False)
    pd.DataFrame(event_rows).to_csv(OUT / "gbm_events.tsv", sep="\t", index=False)
    _write_json(
        OUT / "gbm_palette.json",
        {
            "tracks": GBM_PALETTE,
            "alterations": {
                "Mutation": "#1E355C",
                "Gained / increased / female": "#BE5362",
                "Lost / decreased / male": "#70B7B4",
            },
        },
    )


def generate_readme_examples() -> None:
    rng = np.random.default_rng(606)
    samples = [f"README-{index:03d}" for index in range(1, 181)]
    groups = np.repeat(["Luminal", "Basal", "HER2", "Normal-like"], [74, 50, 34, 22])
    rng.shuffle(groups)
    probabilities = {
        "PIK3CA": 0.42,
        "TP53": 0.36,
        "CDH1": 0.16,
        "GATA3": 0.14,
        "MAP3K1": 0.12,
        "PTEN": 0.10,
        "KMT2C": 0.10,
        "NF1": 0.08,
        "ERBB2": 0.07,
        "BRCA1": 0.06,
        "RB1": 0.06,
        "EGFR": 0.05,
    }
    rows = []
    for sample, group in zip(samples, groups):
        for gene in README_GENES:
            probability = probabilities[gene]
            if group == "Basal" and gene in {"TP53", "BRCA1", "RB1", "EGFR"}:
                probability *= 1.75
            if group == "Luminal" and gene in {"PIK3CA", "GATA3", "MAP3K1", "CDH1"}:
                probability *= 1.45
            if group == "HER2" and gene in {"ERBB2", "TP53", "PIK3CA"}:
                probability *= 1.55
            if rng.random() < min(probability, 0.90):
                hits = 2 if rng.random() < 0.06 else 1
                for hit in range(hits):
                    mutation_type = str(rng.choice(README_TYPES, p=[0.55, 0.14, 0.12, 0.08, 0.07, 0.04]))
                    rows.append(
                        {
                            "sample": sample,
                            "gene": gene,
                            "mutation_type": mutation_type,
                            "tooltip": f"{sample} {gene} {mutation_type} hit {hit + 1}",
                            "cohort": group,
                        }
                    )
    mutations = pd.DataFrame(rows)
    metadata = pd.DataFrame({"sample": samples, "Subtype": groups})
    metadata["ER_status"] = np.where(metadata["Subtype"].isin(["Luminal", "HER2"]), "Positive", "Negative")
    metadata["PR_status"] = np.where(metadata["Subtype"] == "Luminal", rng.choice(["Positive", "Negative"], len(samples), p=[0.78, 0.22]), "Negative")
    metadata["HER2_status"] = np.where(metadata["Subtype"] == "HER2", "Positive", "Negative")
    metadata["TMB_score"] = rng.gamma(shape=2.2, scale=1.5, size=len(samples)).round(2)
    tmb = _tmb_from_mutations(mutations)
    extra = []
    for sample in samples:
        for mutation_type in rng.choice(README_TYPES, size=rng.integers(1, 3), replace=False):
            extra.append({"sample": sample, "mutation_type": mutation_type, "mutations": int(rng.integers(1, 9))})
    tmb = pd.concat([tmb, pd.DataFrame(extra)], ignore_index=True)
    tmb = tmb.groupby(["sample", "mutation_type"], as_index=False)["mutations"].sum()

    mutations.to_csv(OUT / "ggoncoplot_readme_mutations.tsv", sep="\t", index=False)
    metadata.to_csv(OUT / "ggoncoplot_readme_metadata.tsv", sep="\t", index=False)
    tmb.to_csv(OUT / "ggoncoplot_readme_tmb.tsv", sep="\t", index=False)
    _write_json(OUT / "ggoncoplot_readme_palette.json", README_PALETTE)


def generate_paper_multimodal() -> None:
    rng = np.random.default_rng(707)
    samples = [f"GSM-{index:03d}" for index in range(1, 148)]
    selected = set(samples[:30])
    classes = rng.choice(["Triple Negative", "Not Triple Negative", "Ambiguous"], size=len(samples), p=[0.24, 0.64, 0.12])
    classes[:30] = "Triple Negative"
    centers = {
        "Triple Negative": (-2.2, 1.4),
        "Not Triple Negative": (1.2, -0.8),
        "Ambiguous": (0.0, 1.8),
    }
    point_rows = []
    clinical_rows = []
    for sample, classification in zip(samples, classes):
        cx, cy = centers[str(classification)]
        tsne_x = rng.normal(cx, 0.75)
        tsne_y = rng.normal(cy, 0.72)
        umap_x = rng.normal(cx * 0.62, 1.1)
        umap_y = rng.normal(cy * 0.62, 1.0)
        point_rows.append(
            {
                "sample": sample,
                "classification": classification,
                "tsne_x": round(float(tsne_x), 4),
                "tsne_y": round(float(tsne_y), 4),
                "umap_x": round(float(umap_x), 4),
                "umap_y": round(float(umap_y), 4),
                "selected": sample in selected,
            }
        )
        clinical_rows.append(
            {
                "sample": sample,
                "Classification": classification,
                "PR_status": "Negative" if classification == "Triple Negative" else str(rng.choice(["Positive", "Negative"], p=[0.70, 0.30])),
                "ER_status": "Negative" if classification == "Triple Negative" else str(rng.choice(["Positive", "Negative"], p=[0.76, 0.24])),
                "HER2_status": "Negative" if classification != "Ambiguous" else str(rng.choice(["Positive", "Negative"])),
            }
        )
    event_rows = []
    probabilities = {
        "PIK3CA": 0.33,
        "TP53": 0.31,
        "CDH1": 0.12,
        "GATA3": 0.11,
        "MAP3K1": 0.09,
        "PTEN": 0.08,
        "NF1": 0.07,
        "BRCA1": 0.06,
        "RB1": 0.06,
        "EGFR": 0.05,
    }
    for sample, classification in zip(samples, classes):
        for gene in PAPER_MULTIMODAL_GENES:
            probability = probabilities[gene]
            if sample in selected and gene in {"TP53", "BRCA1", "RB1", "EGFR"}:
                probability *= 2.0
            if classification == "Not Triple Negative" and gene in {"PIK3CA", "GATA3", "MAP3K1"}:
                probability *= 1.45
            if rng.random() < min(probability, 0.88):
                alteration = str(rng.choice(PAPER_MULTIMODAL_TYPES, p=[0.70, 0.13, 0.13, 0.04]))
                event_rows.append({"sample": sample, "gene": gene, "alteration": alteration, "tooltip": f"{sample} {gene} {alteration}"})

    pd.DataFrame({"sample": samples, "selected": [sample in selected for sample in samples]}).to_csv(OUT / "paper_multimodal_samples.tsv", sep="\t", index=False)
    pd.DataFrame(point_rows).to_csv(OUT / "paper_multimodal_points.tsv", sep="\t", index=False)
    pd.DataFrame(event_rows).to_csv(OUT / "paper_multimodal_events.tsv", sep="\t", index=False)
    pd.DataFrame(clinical_rows).to_csv(OUT / "paper_multimodal_clinical.tsv", sep="\t", index=False)
    pd.DataFrame({"vertex": [1, 2, 3, 4, 5], "x": [-3.45, -2.50, -1.15, -0.80, -3.45], "y": [0.45, 2.65, 2.25, 0.95, 0.45]}).to_csv(
        OUT / "paper_multimodal_selection.tsv", sep="\t", index=False
    )
    _write_json(OUT / "paper_multimodal_palette.json", PAPER_MULTIMODAL_PALETTE)


def generate_comparison_table() -> None:
    rows = [
        {"package": "ggoncoplot", "tidy input": "Yes", "interactive": "Yes", "linked selection": "Yes", "metadata": "Yes", "auto palette": "Yes", "notes": "Python target"},
        {"package": "ComplexHeatmap", "tidy input": "No", "interactive": "Shiny", "linked selection": "Limited", "metadata": "Yes", "auto palette": "Manual", "notes": "Matrix-first"},
        {"package": "maftools", "tidy input": "MAF", "interactive": "No", "linked selection": "No", "metadata": "Yes", "auto palette": "MAF classes", "notes": "Cancer-focused"},
        {"package": "genVisR", "tidy input": "Wide", "interactive": "No", "linked selection": "No", "metadata": "Yes", "auto palette": "Manual", "notes": "Static"},
        {"package": "cBioPortal", "tidy input": "Portal", "interactive": "Yes", "linked selection": "Partial", "metadata": "Yes", "auto palette": "Preset", "notes": "Web app"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "ggoncoplot_comparison_table.tsv", sep="\t", index=False)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    generate_brca()
    generate_cssc()
    generate_gbm()
    generate_readme_examples()
    generate_paper_multimodal()
    generate_comparison_table()
    print(f"Wrote synthetic inputs to {OUT}")
    print("Skipped AML and SV: those fuc-backed fixtures are rebuilt with fuc_sources/rebuild_fuc_fixtures.py.")


if __name__ == "__main__":
    main()
