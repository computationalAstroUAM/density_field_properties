"""Tests for assembly-bias panel helpers."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "input_features,expected_feature_text",
    [
        (("env",), "env"),
        (("t_over_u", "tidal_anisotropy"), "T/|U| + tidal anisotropy"),
        (("env", "t_over_u"), "env + T/|U|"),
    ],
)
def test_format_assembly_bias_panel_title(input_features, expected_feature_text):
    """
    Panel titles should reflect the Haloscope INPUT features of the run.
    """
    pytest.importorskip("bigfile")
    from density_field_properties.validation.assembly_bias_panel import (
        format_assembly_bias_panel_title,
    )

    title = format_assembly_bias_panel_title(
        input_features,
        sim_delta_mode="saved CIC",
        fastpm_delta_mode="halo CIC",
    )

    assert title == (f"input: {expected_feature_text} " "(HR δ: saved CIC, FastPM δ: halo CIC)")


def test_resolve_saved_cic_density_paths_use_tidal_work_dirs(tmp_path):
    """
    Assembly bias should prefer CIC grids under pipeline tidal work directories.
    """
    from density_field_properties.pipelines.config import (
        HaloscopeEnrichmentConfig,
        HaloscopePipelineStages,
        TidalAnisotropyStageConfig,
        TidalAnisotropyTargetConfig,
    )
    from density_field_properties.validation.assembly_bias_panel import (
        _resolve_saved_cic_density_paths,
    )

    unit_work = tmp_path / "output" / "tidal_mn5" / "unit"
    fastpm_work = tmp_path / "output" / "tidal_mn5" / "fastpm"
    unit_work.mkdir(parents=True)
    fastpm_work.mkdir(parents=True)
    unit_density = unit_work / "dm_particles_0.5_128_density"
    unit_info = unit_work / "dm_particles_0.5_128_density_info.txt"
    fastpm_density = fastpm_work / "snap_1.0000_density"
    fastpm_info = fastpm_work / "snap_1.0000_density_info.txt"
    unit_density.write_bytes(b"u")
    unit_info.write_text("#box_size,n_grid,n_particles,mass_particle\n", encoding="utf-8")
    fastpm_density.write_bytes(b"f")
    fastpm_info.write_text("#box_size,n_grid,n_particles,mass_particle\n", encoding="utf-8")

    config = HaloscopeEnrichmentConfig(
        sim_hlist_path=tmp_path / "sim.list",
        fastpm_list_path=tmp_path / "fastpm.list",
        repo_root=tmp_path,
        pipeline_stages=HaloscopePipelineStages(
            tidal_anisotropy=TidalAnisotropyStageConfig(
                unit=TidalAnisotropyTargetConfig(work_dir=Path("output/tidal_mn5/unit")),
                fastpm=TidalAnisotropyTargetConfig(work_dir=Path("output/tidal_mn5/fastpm")),
            ),
        ),
    )

    (sim_density_path, sim_info_path), (fastpm_density_path, fastpm_info_path) = (
        _resolve_saved_cic_density_paths(tmp_path, config)
    )

    assert sim_density_path == unit_density
    assert sim_info_path == unit_info
    assert fastpm_density_path == fastpm_density
    assert fastpm_info_path == fastpm_info


def test_resolve_assembly_bias_context_uses_run_config(tmp_path):
    """
    Assembly-bias context should honor box size and DM paths from the run config.
    """
    from density_field_properties.pipelines.config import (
        HaloscopeEnrichmentConfig,
        HaloscopePipelineStages,
        TidalAnisotropyStageConfig,
        TidalAnisotropyTargetConfig,
    )
    from density_field_properties.validation.assembly_bias_panel import (
        _resolve_assembly_bias_context,
    )

    unit_dm = tmp_path / "unit_dm.dat"
    fastpm_dm = tmp_path / "fastpm_dm"
    config = HaloscopeEnrichmentConfig(
        sim_hlist_path=tmp_path / "sim.list",
        fastpm_list_path=tmp_path / "fastpm.list",
        box_size_mpc_h=200.0,
        pipeline_stages=HaloscopePipelineStages(),
    )
    config.pipeline_stages.tidal_anisotropy = TidalAnisotropyStageConfig(
        enabled=True,
        mass_particle=9.9e8,
        unit=TidalAnisotropyTargetConfig(dm_particles_file=unit_dm),
        fastpm=TidalAnisotropyTargetConfig(dm_particles_file=fastpm_dm),
    )

    sim_box, fastpm_box, sim_dm, fastpm_dm_path, dm_mass = _resolve_assembly_bias_context(config)

    assert sim_box == 200.0
    assert fastpm_box == 200.0
    assert sim_dm == unit_dm
    assert fastpm_dm_path == fastpm_dm
    assert dm_mass == 9.9e8
