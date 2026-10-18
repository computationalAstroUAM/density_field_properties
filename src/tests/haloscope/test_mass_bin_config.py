"""Tests for Haloscope mass-bin configuration."""

import numpy as np
import pytest

from density_field_properties.haloscope.mass_bin_config import (
    HaloscopeMassBinConfig,
    parse_haloscope_mass_bin_config,
    resolve_mass_bin_edges,
)


def test_parse_mass_bin_from_fastpm_min():
    """
    JSON mass_bin should enable auto lower edge from FastPM masses.
    """
    config = parse_haloscope_mass_bin_config(
        {"mass_bin": {"log_min_from_fastpm": True, "log_max": 13.0}}
    )
    assert config.log_min_from_fastpm is True
    assert config.log_max == pytest.approx(13.0)


def test_resolve_mass_bin_edges_uses_fastpm_minimum():
    """
    Custom single-bin edges should start at the smallest LR mass.
    """
    lr_masses = np.array([4.0e12, 5.0e12, 8.0e12])
    edges = resolve_mass_bin_edges(
        log_m200b_hr_max=13.0,
        lr_masses=lr_masses,
        mass_bin=HaloscopeMassBinConfig(
            log_max=13.0,
            log_min_from_fastpm=True,
        ),
    )
    expected_low = np.log10(4.0e12)
    assert edges.tolist() == [expected_low, 13.0]
