"""Tests for multi-stage Haloscope pipeline orchestration."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from density_field_properties.pipelines.config import (
    HaloscopeEnrichmentConfig,
    HaloscopePipelineStages,
    HaloscopeStageConfig,
    PreprocessStageConfig,
    TidalAnisotropyStageConfig,
    TidalAnisotropyTargetConfig,
)
from density_field_properties.pipelines.haloscope_pipeline import (
    _resolve_descriptor_dirs_from_pipeline,
    run_assembly_bias_from_cached_tables,
    run_haloscope_pipeline,
)


def _minimal_config(tmp_path: Path, stages: HaloscopePipelineStages) -> HaloscopeEnrichmentConfig:
    """
    Build a minimal enrichment config for pipeline orchestration tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for catalog and output paths.
    stages : HaloscopePipelineStages
        Pipeline stage toggles under test.

    Returns
    -------
    HaloscopeEnrichmentConfig
        Config pointing at placeholder catalog files.
    """
    sim_path = tmp_path / "sim.list"
    fastpm_path = tmp_path / "fastpm.list"
    sim_path.write_text("# header\n", encoding="utf-8")
    fastpm_path.write_text("# header\n", encoding="utf-8")
    return HaloscopeEnrichmentConfig(
        sim_hlist_path=sim_path,
        fastpm_list_path=fastpm_path,
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
        box_size_mpc_h=200.0,
        pipeline_stages=stages,
        input_features=("env",),
        run_assembly_bias_plot=False,
    )


def test_run_haloscope_pipeline_executes_tidal_targets_sequentially(tmp_path):
    """
    Enabled tidal-anisotropy stage should process each configured target once.
    """
    stages = HaloscopePipelineStages(
        tidal_anisotropy=TidalAnisotropyStageConfig(
            enabled=True,
            targets=("unit", "fastpm"),
            unit=TidalAnisotropyTargetConfig(
                dm_particles_file=tmp_path / "unit_dm.dat",
                work_dir=tmp_path / "unit_work",
            ),
            fastpm=TidalAnisotropyTargetConfig(
                dm_particles_file=tmp_path / "fastpm_snap",
                work_dir=tmp_path / "fastpm_work",
                catalog_layout="rockstar_list",
            ),
        ),
        preprocess=PreprocessStageConfig(enabled=False),
        haloscope=HaloscopeStageConfig(enabled=False),
    )
    config = _minimal_config(tmp_path, stages)
    processed_targets = []

    def fake_tidal_stage(active_config):
        processed_targets.extend(active_config.pipeline_stages.tidal_anisotropy.targets)

    with patch(
        "density_field_properties.pipelines.haloscope_pipeline.run_tidal_anisotropy_stage",
        side_effect=fake_tidal_stage,
    ):
        result = run_haloscope_pipeline(config)

    assert processed_targets == ["unit", "fastpm"]
    assert result is None


def test_resolve_descriptor_dirs_from_pipeline_work_dirs(tmp_path):
    """
    Tidal descriptor dirs should resolve from pipeline work dirs when unset.
    """
    unit_work = tmp_path / "unit_work"
    fastpm_work = tmp_path / "fastpm_work"
    stages = HaloscopePipelineStages(
        tidal_anisotropy=TidalAnisotropyStageConfig(
            enabled=False,
            unit=TidalAnisotropyTargetConfig(work_dir=unit_work),
            fastpm=TidalAnisotropyTargetConfig(work_dir=fastpm_work),
        ),
    )
    config = _minimal_config(tmp_path, stages)
    config.input_features = ("t_over_u", "tidal_anisotropy")

    _resolve_descriptor_dirs_from_pipeline(config)

    assert config.unit_descriptors_dir == unit_work / "tidal_anisotropy"
    assert config.fastpm_descriptors_dir == fastpm_work / "tidal_anisotropy"


def test_run_haloscope_pipeline_assembly_bias_only(tmp_path):
    """
    Assembly-bias-only mode should load Parquet tables and skip enrichment.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    hr_path = out_dir / "hr.parquet"
    enriched_path = out_dir / "enriched.parquet"
    hr = pd.DataFrame({"M200b": [1.0e12]})
    lr = pd.DataFrame({"M200b_cal": [9.0e11], "cv": [1.0], "Spin": [0.1], "ca": [0.9], "ba": [0.8]})
    hr.to_parquet(hr_path, index=False)
    lr.to_parquet(enriched_path, index=False)

    stages = HaloscopePipelineStages(
        preprocess=PreprocessStageConfig(hr_parquet_name="hr.parquet"),
        haloscope=HaloscopeStageConfig(enabled=False),
    )
    config = _minimal_config(tmp_path, stages)
    config.output_dir = out_dir
    config.run_assembly_bias_plot = True
    config.assembly_bias_only = True
    config.enriched_parquet_name = "enriched.parquet"
    config.input_features = ("env",)

    with patch(
        "density_field_properties.pipelines.haloscope_pipeline.write_tidal_assembly_bias_panel",
        return_value=out_dir / "assembly_bias_tidal_input.pdf",
    ) as write_panel:
        result = run_haloscope_pipeline(config)

    assert result == out_dir / "assembly_bias_tidal_input.pdf"
    write_panel.assert_called_once()
    called_hr = write_panel.call_args[0][0]
    assert called_hr["M200b"].iloc[0] == 1.0e12


def test_run_assembly_bias_from_cached_tables_requires_enabled_flag(tmp_path):
    """
    Cached assembly-bias runs should fail when validation assembly bias is disabled.
    """
    config = _minimal_config(tmp_path, HaloscopePipelineStages())
    config.run_assembly_bias_plot = False
    config.assembly_bias_only = True

    with pytest.raises(ValueError, match="assembly_bias.only requires"):
        run_assembly_bias_from_cached_tables(config)


def test_run_haloscope_pipeline_preprocess_only(tmp_path):
    """
    Preprocess stage should persist HR/LR tables without running Haloscope.
    """
    stages = HaloscopePipelineStages(
        preprocess=PreprocessStageConfig(enabled=True),
        haloscope=HaloscopeStageConfig(enabled=False),
    )
    config = _minimal_config(tmp_path, stages)
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2]})

    with patch(
        "density_field_properties.pipelines.haloscope_pipeline.build_haloscope_feature_tables",
        return_value=(hr, lr, "M200b_cal"),
    ):
        result = run_haloscope_pipeline(config)

    assert result is not None
    hr_path, lr_path = result
    assert hr_path.is_file()
    assert lr_path.is_file()
    assert pd.read_parquet(hr_path).shape[0] == 1
