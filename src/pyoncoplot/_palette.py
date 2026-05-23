"""Mutation type palette handling."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import pandas as pd


MAF_PALETTE: Dict[str, str] = {
    "Missense_Mutation": "#1F78B4",
    "Nonsense_Mutation": "#E31A1C",
    "Nonstop_Mutation": "#B15928",
    "Splice_Site": "#6A3D9A",
    "Splice_Region": "#CAB2D6",
    "Translation_Start_Site": "#FB9A99",
    "Frame_Shift_Del": "#33A02C",
    "Frame_Shift_Ins": "#B2DF8A",
    "In_Frame_Del": "#FF7F00",
    "In_Frame_Ins": "#FDBF6F",
    "De_novo_Start_InFrame": "#A6CEE3",
    "De_novo_Start_OutOfFrame": "#B3DE69",
    "Silent": "#BDBDBD",
    "RNA": "#8DD3C7",
    "Intron": "#FFFFB3",
    "IGR": "#BEBADA",
    "3'UTR": "#80B1D3",
    "5'UTR": "#FCCDE5",
    "5'Flank": "#D9D9D9",
    "3'Flank": "#BC80BD",
    "Targeted_Region": "#CCEBC5",
    "Multi_Hit": "black",
}

SO_PALETTE: Dict[str, str] = {
    "missense_variant": "#1F78B4",
    "stop_gained": "#E31A1C",
    "stop_lost": "#B15928",
    "splice_acceptor_variant": "#6A3D9A",
    "splice_donor_variant": "#6A3D9A",
    "splice_region_variant": "#CAB2D6",
    "frameshift_variant": "#33A02C",
    "inframe_deletion": "#FF7F00",
    "inframe_insertion": "#FDBF6F",
    "synonymous_variant": "#BDBDBD",
    "start_lost": "#FB9A99",
    "intron_variant": "#FFFFB3",
    "upstream_gene_variant": "#D9D9D9",
    "downstream_gene_variant": "#BC80BD",
    "3_prime_UTR_variant": "#80B1D3",
    "5_prime_UTR_variant": "#FCCDE5",
    "Multi_Hit": "black",
}

PAVE_PALETTE: Dict[str, str] = {
    "Loss of Function": "#E31A1C",
    "Damaging Missense": "#1F78B4",
    "Other Missense": "#A6CEE3",
    "Synonymous": "#BDBDBD",
    "Splice": "#6A3D9A",
    "Inframe Indel": "#FF7F00",
    "Other": "#8DD3C7",
    "Multi_Hit": "black",
}

FALLBACK_COLORS = [
    "#A6CEE3",
    "#1F78B4",
    "#B2DF8A",
    "#33A02C",
    "#FB9A99",
    "#E31A1C",
    "#FDBF6F",
    "#FF7F00",
    "#CAB2D6",
    "#6A3D9A",
    "#FFFF99",
    "#B15928",
]


def _clean_terms(values: Iterable[object]) -> list:
    series = pd.Series(list(values), dtype="object")
    series = series[~series.isna()]
    return [str(value) for value in pd.unique(series)]


def _reject_compound_so_terms(terms: Iterable[str]) -> None:
    compound = [term for term in terms if "&" in term]
    if compound:
        raise ValueError(
            "Mutation type values cannot contain ampersand-delimited Sequence Ontology terms. "
            "Please preselect the most severe consequence before plotting. "
            f"Examples: {', '.join(compound[:3])}"
        )


def assert_palette_is_sensible(
    palette: Mapping[str, str],
    mutation_types: Iterable[object],
) -> Dict[str, str]:
    """Validate and return a mutation-type palette."""

    if not isinstance(palette, Mapping):
        raise TypeError("palette must be a mapping of mutation type to color.")
    unique_impacts = _clean_terms(mutation_types)
    _reject_compound_so_terms(unique_impacts)
    missing = [term for term in unique_impacts if term not in palette]
    if missing:
        raise ValueError(
            "Please add colour mappings for the following terms: "
            + ", ".join(map(str, missing))
        )
    out = dict(palette)
    if "Multi_Hit" not in out:
        out["Multi_Hit"] = "black"
    return out


def get_sensible_default_palette(
    mutation_types: Iterable[object],
    verbose: bool = False,
) -> Optional[Dict[str, str]]:
    """Choose a deterministic default palette for mutation types."""

    unique_impacts = _clean_terms(mutation_types)
    if not unique_impacts:
        return None
    _reject_compound_so_terms(unique_impacts)

    terms_without_multi = [term for term in unique_impacts if term != "Multi_Hit"]

    if all(term in MAF_PALETTE for term in terms_without_multi):
        return {term: MAF_PALETTE[term] for term in unique_impacts if term in MAF_PALETTE}

    if all(term in SO_PALETTE for term in terms_without_multi):
        return {term: SO_PALETTE[term] for term in unique_impacts if term in SO_PALETTE}

    if all(term in PAVE_PALETTE for term in terms_without_multi):
        return {term: PAVE_PALETTE[term] for term in unique_impacts if term in PAVE_PALETTE}

    if len(unique_impacts) > len(FALLBACK_COLORS):
        raise ValueError(
            "Too many unique mutation types for automatic palette generation "
            f"(need <= {len(FALLBACK_COLORS)}, not {len(unique_impacts)}). "
            "Please supply a custom mutation type to colour mapping."
        )

    palette = {term: FALLBACK_COLORS[index] for index, term in enumerate(unique_impacts)}
    palette.setdefault("Multi_Hit", "black")
    return palette
