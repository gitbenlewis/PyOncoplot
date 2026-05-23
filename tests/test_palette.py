import pandas as pd
import pytest

from pyoncoplot import assert_palette_is_sensible, get_sensible_default_palette


def test_default_palette_detects_maf_and_adds_multi_hit():
    palette = get_sensible_default_palette(["Missense_Mutation", "Frame_Shift_Del", "Multi_Hit"])
    assert palette["Missense_Mutation"]
    assert palette["Multi_Hit"] == "black"


def test_palette_validation_requires_all_observed_terms():
    with pytest.raises(ValueError, match="Nonsense_Mutation"):
        assert_palette_is_sensible({"Missense_Mutation": "green"}, ["Missense_Mutation", "Nonsense_Mutation"])


def test_fallback_palette_errors_for_too_many_categories():
    terms = [f"type_{index}" for index in range(20)]
    with pytest.raises(ValueError, match="Too many"):
        get_sensible_default_palette(terms)


def test_palette_rejects_ampersand_delimited_so_terms():
    with pytest.raises(ValueError, match="ampersand"):
        get_sensible_default_palette(["missense_variant&intron_variant"])
