"""Mass calibration between LR and HR halo catalogs."""

import numpy as np
import pandas as pd
from scipy import stats

from density_field_properties.preprocessing.halo_matching import (
    UNMATCHED_INDEX,
    hr_masses_for_matched_lr,
    match_hr_to_lr_one_to_one,
)
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    MASS_CALIBRATION_METHOD_MATCHING_ML,
    MassCalibrationConfig,
    MatchingHyperparameters,
)
from density_field_properties.preprocessing.mass_calibration_ml import apply_matching_ml


def abundance_match_mass_cdf(mass_target: np.ndarray, mass_reference: np.ndarray) -> np.ndarray:
    """
    Monotonic abundance matching of ``mass_target`` onto the mass function of ``mass_reference``.

    Parameters
    ----------
    mass_target : np.ndarray
        Masses to remap (e.g. FastPM M200b).
    mass_reference : np.ndarray
        Reference mass sample (e.g. UNIT SIM M200b).

    Returns
    -------
    np.ndarray
        Masses on the reference cumulative distribution, same shape as ``mass_target``.
    """
    quantile_target = stats.rankdata(mass_target) / (len(mass_target) + 1)
    reference_sorted = np.sort(mass_reference)
    quantile_reference = (np.arange(len(reference_sorted)) + 1) / (len(reference_sorted) + 1)
    return np.interp(quantile_target, quantile_reference, reference_sorted)


def abundance_match_mass(
    mass_target: np.ndarray,
    mass_reference: np.ndarray,
    volume_reference: float | int | None = None,
    volume_target: float | int | None = None,
) -> np.ndarray:
    """
    Map target cumulative densities onto reference masses via log-log interpolation.

    Parameters
    ----------
    mass_target : np.ndarray
        Masses to remap (e.g. FastPM M200b).
    mass_reference : np.ndarray
        Reference mass sample (e.g. UNIT SIM M200b).
    volume_reference : float or int, optional
        Simulation volume in (Mpc/h)^3 for the reference sample.
    volume_target : float or int, optional
        Simulation volume in (Mpc/h)^3 for the target sample.

    Returns
    -------
    np.ndarray
        Calibrated masses with the same shape as ``mass_target``.
    """
    mass_reference_sorted = np.sort(mass_reference)[::-1]
    mass_target_argsort = np.argsort(mass_target)[::-1]

    n_reference_density = np.arange(len(mass_reference_sorted), dtype=np.float64) + 1.0
    n_target_density = np.arange(len(mass_target), dtype=np.float64) + 1.0

    condition_volume = int(volume_reference is None) + int(volume_target is None)
    if condition_volume == 1:
        raise ValueError("volume_reference and volume_target must both be set or both omitted.")
    elif condition_volume == 0:
        n_reference_density /= float(volume_reference)
        n_target_density /= float(volume_target)

    log_density_reference = np.log10(n_reference_density)
    log_density_target = np.log10(n_target_density)
    log_mass_reference = np.log10(mass_reference_sorted)
    log_mass_new = np.interp(
        log_density_target,
        log_density_reference,
        log_mass_reference,
    )
    new_masses = np.empty_like(mass_target)
    new_masses[mass_target_argsort] = np.power(10.0, log_mass_new)
    return new_masses


def _apply_abundance_matching(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCalibrationConfig,
    box_size_mpc_h: float,
) -> pd.DataFrame:
    """
    Write abundance-matched masses on the LR catalog.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog.
    hr_catalog : pd.DataFrame
        High-resolution reference catalog.
    config : MassCalibrationConfig
        Calibration settings including column names.
    box_size_mpc_h : float
        Periodic box side in Mpc/h for number-density normalization.

    Returns
    -------
    pd.DataFrame
        ``lr_catalog`` with ``config.calibrated_column`` populated.

    Raises
    ------
    ValueError
        If either mass sample is empty.
    """
    volume = None
    if box_size_mpc_h > 0.0:
        volume = float(box_size_mpc_h) ** 3

    lr_masses = lr_catalog[config.matching.lr_mass_column].to_numpy()
    hr_masses = hr_catalog[config.matching.hr_mass_column].to_numpy()
    if len(lr_masses) == 0 or len(hr_masses) == 0:
        raise ValueError("Empty HR or LR mass sample for abundance matching.")
    lr_catalog[config.calibrated_column] = abundance_match_mass(
        lr_masses, hr_masses, volume, volume
    )
    lr_catalog["mass_calib_method"] = MASS_CALIBRATION_METHOD_ABUNDANCE
    return lr_catalog


def _apply_matching_1to1(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCalibrationConfig,
    box_size_mpc_h: float,
) -> pd.DataFrame:
    """
    Assign HR masses to matched LR halos via spatial 1:1 matching.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog.
    hr_catalog : pd.DataFrame
        High-resolution reference catalog.
    config : MassCalibrationConfig
        Calibration settings including matching hyperparameters.
    box_size_mpc_h : float
        Periodic box side length in Mpc/h.

    Returns
    -------
    pd.DataFrame
        ``lr_catalog`` with calibrated mass and optional trace columns.

    Raises
    ------
    ValueError
        If every LR calibrated mass is NaN after matching.
    """
    matching = match_hr_to_lr_one_to_one(
        hr_catalog,
        lr_catalog,
        box_size_mpc_h=box_size_mpc_h,
        params=config.matching,
    )
    hr_mass_column = config.matching.hr_mass_column
    hr_masses = hr_catalog[hr_mass_column].to_numpy(dtype=float)
    calibrated = hr_masses_for_matched_lr(matching, hr_masses)

    lr_catalog[config.calibrated_column] = calibrated
    matched_mask = matching.lr_to_hr_index != UNMATCHED_INDEX
    lr_catalog["mass_calib_matched"] = matched_mask
    lr_catalog["mass_calib_method"] = MASS_CALIBRATION_METHOD_MATCHING_1TO1

    if "id" in hr_catalog.columns:
        hr_ids = hr_catalog["id"].to_numpy()
        partner_ids = np.full(len(lr_catalog), np.nan, dtype=float)
        partner_ids[matched_mask] = hr_ids[matching.lr_to_hr_index[matched_mask]]
        lr_catalog["mass_calib_hr_id"] = partner_ids

    if not np.any(np.isfinite(calibrated)):
        raise ValueError("All LR calibrated masses are NaN after matching_1to1.")

    return lr_catalog


def apply_mass_calibration(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCalibrationConfig,
    box_size_mpc_h: float,
) -> tuple[pd.DataFrame, str]:
    """
    Calibrate LR halo masses for Haloscope bin assignment.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog (FastPM).
    hr_catalog : pd.DataFrame
        High-resolution training catalog (UNIT).
    config : MassCalibrationConfig
        Method key and hyperparameters.
    box_size_mpc_h : float
        Periodic box side in Mpc/h for spatial and abundance matching.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The same ``lr_catalog`` instance and the mass column for bin assignment.

    Raises
    ------
    ValueError
        If the method is unknown or not yet implemented.
    """
    lr_mass_column = config.matching.lr_mass_column
    if not config.enabled:
        return lr_catalog, lr_mass_column

    if config.method == MASS_CALIBRATION_METHOD_ABUNDANCE:
        _apply_abundance_matching(lr_catalog, hr_catalog, config, box_size_mpc_h)
        return lr_catalog, config.calibrated_column

    if config.method == MASS_CALIBRATION_METHOD_MATCHING_1TO1:
        _apply_matching_1to1(lr_catalog, hr_catalog, config, box_size_mpc_h)
        return lr_catalog, config.calibrated_column

    if config.method == MASS_CALIBRATION_METHOD_MATCHING_ML:
        apply_matching_ml(lr_catalog, hr_catalog, config, box_size_mpc_h)
        return lr_catalog, config.calibrated_column

    raise ValueError(f"Unknown mass calibration method: {config.method}")


def calibrate_lr_mass(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    calibrate_mass: bool,
    lr_mass_column: str = "M200b",
    hr_mass_column: str = "M200b",
    calibrated_column: str = "M200b_cal",
) -> tuple[pd.DataFrame, str]:
    """
    Optionally abundance-match LR masses to the HR mass function.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog (FastPM).
    hr_catalog : pd.DataFrame
        High-resolution training catalog (UNIT).
    calibrate_mass : bool
        If True, write ``calibrated_column`` on ``lr_catalog``.
    lr_mass_column : str, optional
        Raw LR mass column.
    hr_mass_column : str, optional
        HR reference mass column.
    calibrated_column : str, optional
        Output calibrated mass column on LR.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The same ``lr_catalog`` instance and the mass column for bin assignment.
    """
    config = MassCalibrationConfig(
        enabled=calibrate_mass,
        method=MASS_CALIBRATION_METHOD_ABUNDANCE,
        calibrated_column=calibrated_column,
        matching=MatchingHyperparameters(
            lr_mass_column=lr_mass_column,
            hr_mass_column=hr_mass_column,
        ),
    )
    return apply_mass_calibration(lr_catalog, hr_catalog, config, box_size_mpc_h=0.0)
