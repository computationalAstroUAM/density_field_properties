"""Tests for tidal anisotropy feature attachment."""

import sys

import pandas as pd
import pytest

if sys.version_info < (3, 11):
    pytest.skip("tidal anisotropy tests require Python 3.11+", allow_module_level=True)

from density_field_properties.preprocessing.context import SimulationRunContext
from density_field_properties.preprocessing.input_features.registry import resolve_attachers
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    HaloEnvironmentDescriptorsAttacher,
    _load_halo_environment_descriptor_table,
    input_features_need_tidal_descriptors,
)


def test_load_halo_environment_descriptor_table_reads_first_batch(tmp_path):
    """
    Load a single descriptor batch file with the expected column names.
    """
    descriptor_file = tmp_path / "0_halo_environment_descriptors.txt"
    descriptor_file.write_text(
        "# Halo ID, X, Y, Z, M200b, R_G, Tidal Anisotropy, Overdensity\n"
        "1 10 20 30 1.0e12 1.5 0.8 -0.2\n"
        "2 11 21 31 2.0e12 1.6 0.9 -0.1\n"
    )
    frame = _load_halo_environment_descriptor_table(tmp_path, max_batch_files=1)
    assert "tidal_anisotropy" in frame.columns
    assert len(frame) == 2


def _catalog_and_context(tmp_path, boxsize=1000.0, n_grid=512):
    cell_size = boxsize / n_grid
    descriptor_file = tmp_path / "0_halo_environment_descriptors.txt"
    descriptor_file.write_text("# header\n1 10 20 30 1.0e12 1.5 0.42 -0.17\n")
    catalog = pd.DataFrame(
        {
            "x": [10 * cell_size + 0.1],
            "y": [20 * cell_size + 0.1],
            "z": [30 * cell_size + 0.1],
            "M200b": [1.0e12],
        }
    )
    run_context = SimulationRunContext(
        boxsize_mpc_h=boxsize,
        tidal_descriptors_dir=tmp_path,
        tidal_n_grid=n_grid,
        max_descriptor_batch_files=1,
    )
    return catalog, run_context


def test_tidal_anisotropy_feature_merges_by_grid_cell(tmp_path):
    """
    Attach tidal anisotropy using periodic grid-cell coordinates.
    """
    catalog, run_context = _catalog_and_context(tmp_path)
    attacher = HaloEnvironmentDescriptorsAttacher(("tidal_anisotropy",))
    enriched = attacher.attach(catalog, run_context)
    assert enriched["tidal_anisotropy"].iloc[0] == pytest.approx(0.42)


def test_overdensity_feature_merges_by_grid_cell(tmp_path):
    """
    Attach overdensity using periodic grid-cell coordinates.
    """
    catalog, run_context = _catalog_and_context(tmp_path)
    attacher = HaloEnvironmentDescriptorsAttacher(("overdensity",))
    enriched = attacher.attach(catalog, run_context)
    assert enriched["overdensity"].iloc[0] == pytest.approx(-0.17)


def test_resolve_attachers_attaches_both_descriptors_in_one_step(tmp_path, monkeypatch):
    """
    Both descriptor INPUT features should use a single merge attacher.
    """
    catalog, run_context = _catalog_and_context(tmp_path)
    attachers = resolve_attachers(("tidal_anisotropy", "overdensity"))
    assert len(attachers) == 1
    assert isinstance(attachers[0], HaloEnvironmentDescriptorsAttacher)

    read_calls = 0
    original_read_csv = pd.read_csv

    def counting_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counting_read_csv)
    enriched = attachers[0].attach(catalog, run_context)
    assert read_calls == 1
    assert enriched["tidal_anisotropy"].iloc[0] == pytest.approx(0.42)
    assert enriched["overdensity"].iloc[0] == pytest.approx(-0.17)


@pytest.mark.parametrize(
    "input_features,expected",
    [
        (("env",), False),
        (("tidal_anisotropy",), True),
        (("overdensity",), True),
        (("t_over_u", "overdensity"), True),
    ],
)
def test_input_features_need_tidal_descriptors(input_features, expected):
    """
    Descriptor directories are required for tidal environment INPUT columns.
    """
    assert input_features_need_tidal_descriptors(input_features) is expected
