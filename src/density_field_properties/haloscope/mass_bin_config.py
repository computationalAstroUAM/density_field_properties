"""Haloscope log-mass bin configuration for enrichment runs."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from density_field_properties.haloscope.bins import default_mass_bin_edges


@dataclass(frozen=True)
class HaloscopeMassBinConfig:
    """
    Optional override for Haloscope log10(M200b) bin edges.

    Parameters
    ----------
    log_min : Optional[float]
        Lower log-mass edge in Msun/h. Required when ``log_max`` is set unless
        ``log_min_from_fastpm`` is True.
    log_max : Optional[float]
        Upper log-mass edge. When set, Haloscope uses a single bin ``[log_min, log_max)``.
    log_min_from_fastpm : bool
        If True, set ``log_min`` from the minimum LR mass after preprocessing.
    """

    log_min: Optional[float] = None
    log_max: Optional[float] = None
    log_min_from_fastpm: bool = False

    @property
    def uses_custom_edges(self) -> bool:
        """
        Whether enrichment should bypass the default multi-bin grid.

        Returns
        -------
        bool
            True when ``log_max`` is set.
        """
        return self.log_max is not None


def default_haloscope_mass_bin_config() -> HaloscopeMassBinConfig:
    """
    Return the default multi-bin Haloscope mass grid behaviour.

    Returns
    -------
    HaloscopeMassBinConfig
        Inactive custom bin settings.
    """
    return HaloscopeMassBinConfig()


def parse_haloscope_mass_bin_config(haloscope: Mapping[str, Any]) -> HaloscopeMassBinConfig:
    """
    Parse optional ``haloscope.mass_bin`` settings from JSON.

    Parameters
    ----------
    haloscope : Mapping[str, Any]
        ``haloscope`` section of a run config.

    Returns
    -------
    HaloscopeMassBinConfig
        Parsed mass-bin override.
    """
    payload = haloscope.get("mass_bin")
    if not isinstance(payload, dict):
        return default_haloscope_mass_bin_config()

    log_min = payload.get("log_min")
    log_max = payload.get("log_max")
    log_min_from_fastpm = bool(payload.get("log_min_from_fastpm", False))

    parsed_log_min = float(log_min) if log_min is not None else None
    parsed_log_max = float(log_max) if log_max is not None else None

    if parsed_log_max is not None and not log_min_from_fastpm and parsed_log_min is None:
        raise ValueError(
            "haloscope.mass_bin requires log_min or log_min_from_fastpm when log_max is set."
        )

    return HaloscopeMassBinConfig(
        log_min=parsed_log_min,
        log_max=parsed_log_max,
        log_min_from_fastpm=log_min_from_fastpm,
    )


def resolve_mass_bin_edges(
    log_m200b_hr_max: float,
    lr_masses: np.ndarray,
    mass_bin: HaloscopeMassBinConfig,
) -> np.ndarray:
    """
    Build log-mass bin edges for Haloscope fit and FastPM enrichment.

    Parameters
    ----------
    log_m200b_hr_max : float
        log10 maximum HR mass used for the default multi-bin grid.
    lr_masses : np.ndarray
        LR masses in Msun/h after preprocessing (for auto lower edge).
    mass_bin : HaloscopeMassBinConfig
        Optional single-bin or custom edge override.

    Returns
    -------
    np.ndarray
        Bin edges with length at least two.

    Raises
    ------
    ValueError
        If custom edges are invalid or LR masses are empty.
    """
    if not mass_bin.uses_custom_edges:
        return default_mass_bin_edges(log_m200b_hr_max)

    if len(lr_masses) == 0:
        raise ValueError("Cannot resolve mass-bin edges from an empty FastPM mass sample.")

    log_min = mass_bin.log_min
    if mass_bin.log_min_from_fastpm:
        log_min = float(np.log10(np.min(lr_masses)))
    if log_min is None or mass_bin.log_max is None:
        raise ValueError("Custom mass-bin edges require log_min and log_max.")
    if log_min >= mass_bin.log_max:
        raise ValueError("mass_bin.log_min must be strictly below mass_bin.log_max.")

    return np.array([log_min, mass_bin.log_max], dtype=np.float64)
