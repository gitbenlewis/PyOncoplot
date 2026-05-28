"""Generate deterministic synthetic inputs for non-fuc gallery examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
OUT = HERE / "syntheitic_goal_data"

GALLERY_CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["gallery_params"]
INPUT_FILES = GALLERY_CONFIG["input_files"]
SHARED = GALLERY_CONFIG["shared"]
SYNTHETIC = GALLERY_CONFIG["synthetic_inputs"]


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_type(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    values = list(weights)
    probabilities = np.array(list(weights.values()), dtype=float)
    return str(rng.choice(values, p=probabilities / probabilities.sum()))


def _tmb_from_mutations(mutations: pd.DataFrame, sample_col: str = "sample") -> pd.DataFrame:
    base = (
        mutations.groupby([sample_col, "mutation_type"], dropna=False)
        .size()
        .rename("mutations")
        .reset_index()
    )
    return base.sort_values([sample_col, "mutation_type"]).reset_index(drop=True)


def generate_brca() -> None:
    settings = SYNTHETIC["brca"]
    files = INPUT_FILES["brca"]
    rng = np.random.default_rng(settings["seed"])
    samples = [f"{settings['sample_prefix']}-{index:04d}" for index in range(1, settings["sample_count"] + 1)]
    subtype_blocks = np.repeat(list(settings["subtype_counts"]), list(settings["subtype_counts"].values()))
    rng.shuffle(subtype_blocks)

    rows = []
    for sample, subtype in zip(samples, subtype_blocks):
        for gene in settings["genes"]:
            probability = float(settings["gene_probabilities"][gene])
            multiplier = settings["subtype_gene_multipliers"].get(str(subtype))
            if multiplier and gene in multiplier["genes"]:
                probability *= float(multiplier["multiplier"])
            if rng.random() < min(probability, float(settings["max_probability"])):
                hits = 2 if rng.random() < float(settings["multi_hit_probability"]) else 1
                for hit in range(hits):
                    mutation_type = _sample_type(rng, settings["mutation_type_weights"])
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
    metadata["PR_status"] = np.where(
        metadata["Subtype"].str.startswith("HR+"),
        rng.choice(settings["pr_status_weights"]["values"], size=len(samples), p=settings["pr_status_weights"]["weights"]),
        "Negative",
    )
    metadata["HER2_status"] = np.where(metadata["Subtype"].str.contains("HER2\\+"), "Positive", "Negative")
    metadata["Age"] = rng.choice(settings["age_values"], size=len(samples), p=settings["age_weights"])
    metadata["Age_years"] = rng.integers(*settings["age_year_range"], size=len(samples))
    metadata["Menopause"] = np.where(
        metadata["Age"] == ">50",
        "Postmenopausal",
        rng.choice(settings["menopause_weights"]["values"], size=len(samples), p=settings["menopause_weights"]["weights"]),
    )
    metadata["LN_stage"] = rng.choice(settings["ln_stage_weights"]["values"], size=len(samples), p=settings["ln_stage_weights"]["weights"])
    metadata["Grade"] = rng.choice(settings["grade_weights"]["values"], size=len(samples), p=settings["grade_weights"]["weights"])
    metadata["TNM_stage"] = rng.choice(settings["tnm_stage_weights"]["values"], size=len(samples), p=settings["tnm_stage_weights"]["weights"])
    metadata["Histological_type"] = rng.choice(
        settings["histological_type_weights"]["values"],
        size=len(samples),
        p=settings["histological_type_weights"]["weights"],
    )
    metadata["Classification"] = np.where(
        metadata["Subtype"] == "HR-/HER2-",
        "Triple Negative",
        np.where(metadata["Subtype"] == "HR+/HER2-", "Not Triple Negative", "Ambiguous"),
    )

    tmb = _tmb_from_mutations(mutations)
    for sample in samples:
        if rng.random() < float(settings["extra_tmb_probability"]):
            low, high = settings["extra_tmb_mutations_range"]
            extra = pd.DataFrame(
                {
                    "sample": [sample],
                    "mutation_type": [str(rng.choice(list(settings["mutation_type_weights"])))],
                    "mutations": [int(rng.integers(low, high))],
                }
            )
            tmb = pd.concat([tmb, extra], ignore_index=True)
    tmb = tmb.groupby(["sample", "mutation_type"], as_index=False)["mutations"].sum()

    mutations.to_csv(OUT / files["mutations"], sep="\t", index=False)
    metadata.to_csv(OUT / files["metadata"], sep="\t", index=False)
    tmb.to_csv(OUT / files["tmb"], sep="\t", index=False)
    _write_json(OUT / files["palette"], SHARED["brca_palette"])


def generate_cssc() -> None:
    settings = SYNTHETIC["cssc"]
    files = INPUT_FILES["cssc"]
    rng = np.random.default_rng(settings["seed"])
    samples = SHARED["cssc_samples"]
    genes = SHARED["cssc_genes"]
    alteration_weights = settings["alteration_weights"]
    probabilities = np.array(list(alteration_weights.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()

    rows = []
    for gene_index, gene in enumerate(genes):
        sample_order = samples[gene_index:] + samples[:gene_index]
        selected = sample_order[: settings["target_counts"][gene]]
        rng.shuffle(selected)
        for sample_index, sample in enumerate(selected):
            alteration = str(rng.choice(list(alteration_weights), p=probabilities))
            rows.append({"sample": sample, "gene": gene, "alteration": alteration, "tooltip": f"{sample} {gene} {alteration}"})
            if (gene_index + sample_index) % int(settings["secondary_modulo"]) == 0:
                second = str(rng.choice(settings["secondary_alterations"]))
                rows.append({"sample": sample, "gene": gene, "alteration": second, "tooltip": f"{sample} {gene} {second} secondary"})

    mutations = pd.DataFrame(rows)
    tmb = _tmb_from_mutations(mutations.rename(columns={"alteration": "mutation_type"})).rename(columns={"mutation_type": "alteration"})
    extra_rows = []
    count_low, count_high = settings["extra_tmb_type_count_range"]
    mutation_low, mutation_high = settings["extra_tmb_mutations_range"]
    for sample in samples:
        for alteration in rng.choice(list(alteration_weights), size=rng.integers(count_low, count_high), replace=False):
            extra_rows.append({"sample": sample, "alteration": alteration, "mutations": int(rng.integers(mutation_low, mutation_high))})
    tmb = pd.concat([tmb, pd.DataFrame(extra_rows)], ignore_index=True)
    tmb = tmb.groupby(["sample", "alteration"], as_index=False)["mutations"].sum()

    mutations.to_csv(OUT / files["mutations"], sep="\t", index=False)
    tmb.to_csv(OUT / files["tmb"], sep="\t", index=False)
    _write_json(OUT / files["palette"], {key: value for key, value in SHARED["cssc_palette"].items() if key != "Multi_Hit"})


def generate_gbm() -> None:
    settings = SYNTHETIC["gbm"]
    files = INPUT_FILES["gbm"]
    rng = np.random.default_rng(settings["seed"])
    samples = [f"{settings['sample_prefix']}-{index:03d}" for index in range(1, settings["sample_count"] + 1)]
    gits_r1 = rng.choice(settings["subtype_values"], size=len(samples), p=settings["subtype_weights"])
    gits_r2 = np.where(rng.random(len(samples)) < float(settings["beva_random_probability"]), "NA / Beva: random trial", gits_r1)
    recurrence_weights = settings["track_weights"]["Recurrence"]
    recurrence = rng.choice(recurrence_weights["values"], size=len(samples), p=recurrence_weights["weights"])

    track_data = {"sample": samples}
    for track in SHARED["gbm_tracks"]:
        if track == "Recurrence":
            track_data[track] = recurrence
        elif track == "Primary":
            track_data[track] = np.where(recurrence == "Yes / stable / resection", "No / wildtype / biopsy", "Yes / stable / resection")
        elif track == "GITS subtype R2":
            track_data[track] = gits_r2
        elif track == "GITS subtype R1":
            track_data[track] = gits_r1
        else:
            weights = settings["track_weights"][track]
            track_data[track] = rng.choice(weights["values"], size=len(samples), p=weights["weights"])
    tracks = pd.DataFrame(track_data, columns=["sample"] + SHARED["gbm_tracks"])

    event_rows = []
    event_weights = settings["event_weights"]
    event_probabilities = np.array(list(event_weights.values()), dtype=float)
    event_probabilities = event_probabilities / event_probabilities.sum()
    for sample, subtype in zip(samples, gits_r1):
        for gene in SHARED["gbm_genes"]:
            probability = float(settings["gene_probabilities"][gene])
            multiplier = settings["subtype_gene_multipliers"].get(str(subtype))
            if multiplier and gene in multiplier["genes"]:
                probability *= float(multiplier["multiplier"])
            if rng.random() < min(probability, float(settings["max_probability"])):
                event = str(rng.choice(list(event_weights), p=event_probabilities))
                event_rows.append({"sample": sample, "gene": gene, "alteration": event, "tooltip": f"{sample} {gene} {event}"})
                if rng.random() < float(settings["secondary_event_probability"]):
                    second = "Gained / increased / female" if event != "Gained / increased / female" else "Mutation"
                    event_rows.append({"sample": sample, "gene": gene, "alteration": second, "tooltip": f"{sample} {gene} {second} secondary"})

    tracks.to_csv(OUT / files["tracks"], sep="\t", index=False)
    pd.DataFrame(event_rows).to_csv(OUT / files["events"], sep="\t", index=False)
    _write_json(
        OUT / files["palette"],
        {
            "tracks": SHARED["gbm_track_palette"],
            "alterations": {key: value for key, value in SHARED["gbm_alteration_palette"].items() if key != "Multi_Hit"},
        },
    )


def generate_readme_examples() -> None:
    settings = SYNTHETIC["readme"]
    files = INPUT_FILES["readme"]
    rng = np.random.default_rng(settings["seed"])
    samples = [f"{settings['sample_prefix']}-{index:03d}" for index in range(1, settings["sample_count"] + 1)]
    groups = np.repeat(list(settings["subtype_counts"]), list(settings["subtype_counts"].values()))
    rng.shuffle(groups)

    rows = []
    for sample, group in zip(samples, groups):
        for gene in SHARED["readme_genes"]:
            probability = float(settings["gene_probabilities"][gene])
            multiplier = settings["subtype_gene_multipliers"].get(str(group))
            if multiplier and gene in multiplier["genes"]:
                probability *= float(multiplier["multiplier"])
            if rng.random() < min(probability, float(settings["max_probability"])):
                hits = 2 if rng.random() < float(settings["multi_hit_probability"]) else 1
                for hit in range(hits):
                    mutation_type = _sample_type(rng, settings["mutation_type_weights"])
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
    metadata["PR_status"] = np.where(
        metadata["Subtype"] == "Luminal",
        rng.choice(settings["pr_status_weights"]["values"], len(samples), p=settings["pr_status_weights"]["weights"]),
        "Negative",
    )
    metadata["HER2_status"] = np.where(metadata["Subtype"] == "HER2", "Positive", "Negative")
    metadata["TMB_score"] = rng.gamma(shape=settings["tmb_gamma_shape"], scale=settings["tmb_gamma_scale"], size=len(samples)).round(2)

    tmb = _tmb_from_mutations(mutations)
    extra = []
    count_low, count_high = settings["extra_tmb_type_count_range"]
    mutation_low, mutation_high = settings["extra_tmb_mutations_range"]
    for sample in samples:
        for mutation_type in rng.choice(list(settings["mutation_type_weights"]), size=rng.integers(count_low, count_high), replace=False):
            extra.append({"sample": sample, "mutation_type": mutation_type, "mutations": int(rng.integers(mutation_low, mutation_high))})
    tmb = pd.concat([tmb, pd.DataFrame(extra)], ignore_index=True)
    tmb = tmb.groupby(["sample", "mutation_type"], as_index=False)["mutations"].sum()

    mutations.to_csv(OUT / files["mutations"], sep="\t", index=False)
    metadata.to_csv(OUT / files["metadata"], sep="\t", index=False)
    tmb.to_csv(OUT / files["tmb"], sep="\t", index=False)
    _write_json(OUT / files["palette"], SHARED["readme_palette"])


def generate_paper_multimodal() -> None:
    settings = SYNTHETIC["multimodal"]
    files = INPUT_FILES["multimodal"]
    rng = np.random.default_rng(settings["seed"])
    samples = [f"{settings['sample_prefix']}-{index:03d}" for index in range(1, settings["sample_count"] + 1)]
    selected = set(samples[: settings["selected_count"]])
    classes = rng.choice(settings["classes"], size=len(samples), p=settings["class_weights"])
    classes[: settings["selected_count"]] = settings["selected_class"]

    point_rows = []
    clinical_rows = []
    for sample, classification in zip(samples, classes):
        cx, cy = settings["centers"][str(classification)]
        tsne_x = rng.normal(cx, settings["tsne_sd"][0])
        tsne_y = rng.normal(cy, settings["tsne_sd"][1])
        umap_x = rng.normal(cx * settings["umap_center_scale"], settings["umap_sd"][0])
        umap_y = rng.normal(cy * settings["umap_center_scale"], settings["umap_sd"][1])
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
                "PR_status": "Negative"
                if classification == "Triple Negative"
                else str(rng.choice(settings["pr_status_weights"]["values"], p=settings["pr_status_weights"]["weights"])),
                "ER_status": "Negative"
                if classification == "Triple Negative"
                else str(rng.choice(settings["er_status_weights"]["values"], p=settings["er_status_weights"]["weights"])),
                "HER2_status": "Negative" if classification != "Ambiguous" else str(rng.choice(settings["ambiguous_her2_values"])),
            }
        )

    event_rows = []
    alteration_weights = settings["alteration_weights"]
    alteration_probabilities = np.array(list(alteration_weights.values()), dtype=float)
    alteration_probabilities = alteration_probabilities / alteration_probabilities.sum()
    selected_multiplier = settings["selected_gene_multiplier"]
    class_multipliers = settings["class_gene_multipliers"]
    for sample, classification in zip(samples, classes):
        for gene in SHARED["paper_multimodal_genes"]:
            probability = float(settings["gene_probabilities"][gene])
            if sample in selected and gene in selected_multiplier["genes"]:
                probability *= float(selected_multiplier["multiplier"])
            multiplier = class_multipliers.get(str(classification))
            if multiplier and gene in multiplier["genes"]:
                probability *= float(multiplier["multiplier"])
            if rng.random() < min(probability, float(settings["max_probability"])):
                alteration = str(rng.choice(list(alteration_weights), p=alteration_probabilities))
                event_rows.append({"sample": sample, "gene": gene, "alteration": alteration, "tooltip": f"{sample} {gene} {alteration}"})

    pd.DataFrame({"sample": samples, "selected": [sample in selected for sample in samples]}).to_csv(OUT / files["samples"], sep="\t", index=False)
    pd.DataFrame(point_rows).to_csv(OUT / files["points"], sep="\t", index=False)
    pd.DataFrame(event_rows).to_csv(OUT / files["events"], sep="\t", index=False)
    pd.DataFrame(clinical_rows).to_csv(OUT / files["clinical"], sep="\t", index=False)
    pd.DataFrame(settings["selection_vertices"]).to_csv(OUT / files["selection"], sep="\t", index=False)
    _write_json(OUT / files["palette"], settings["palette"])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    generate_brca()
    generate_cssc()
    generate_gbm()
    generate_readme_examples()
    generate_paper_multimodal()
    comparison_files = INPUT_FILES["comparison_table"]
    pd.DataFrame(SYNTHETIC["comparison_table"]["rows"]).to_csv(OUT / comparison_files["table"], sep="\t", index=False)
    print(f"Wrote synthetic inputs to {OUT}")
    print("Skipped AML and SV: those fuc-backed fixtures are rebuilt with fuc_sources/rebuild_fuc_fixtures.py.")


if __name__ == "__main__":
    main()
