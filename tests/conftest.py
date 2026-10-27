"""Shared figure ownership isolation; retain figures that predate each test."""
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_test_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)
