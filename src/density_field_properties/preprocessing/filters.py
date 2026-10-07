"""Scientific sample filters for Haloscope HR/LR tables."""

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def filter_host_training_halos(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep UNIT host halos with positive mass and finite concentration.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw consistent-trees table with ``pid``, ``M200b``, and ``cv``.

    Returns
    -------
    pd.DataFrame
        Filtered host halos.
    """
    mask = (frame["pid"] == -1) & (frame["M200b"] > 0) & np.isfinite(frame["cv"])
    return frame.loc[mask].copy()


def filter_positive_mass(frame: pd.DataFrame, mass_column: str = "M200b") -> pd.DataFrame:
    """
    Keep halos with strictly positive mass.

    Parameters
    ----------
    frame : pd.DataFrame
        Halo catalog table.
    mass_column : str, optional
        Mass column name.

    Returns
    -------
    pd.DataFrame
        Filtered table with a reset index.
    """
    filtered = frame.loc[frame[mass_column] > 0].copy()
    return filtered.reset_index(drop=True)


def filter_min_m200b(
    frame: pd.DataFrame,
    min_mass_msun_h: float,
    mass_column: str = "M200b",
) -> pd.DataFrame:
    """
    Keep halos above a minimum ``M200b`` threshold.

    Parameters
    ----------
    frame : pd.DataFrame
        Halo catalog table.
    min_mass_msun_h : float
        Minimum halo mass in Msun/h.
    mass_column : str, optional
        Mass column name.

    Returns
    -------
    pd.DataFrame
        Filtered table with a reset index.
    """
    return filter_mass_range(
        frame,
        min_mass_msun_h=min_mass_msun_h,
        mass_column=mass_column,
    )


def filter_mass_range(
    frame: pd.DataFrame,
    min_mass_msun_h: Optional[float] = None,
    max_mass_msun_h: Optional[float] = None,
    mass_column: str = "M200b",
) -> pd.DataFrame:
    """
    Keep halos within optional minimum and maximum mass bounds.

    Parameters
    ----------
    frame : pd.DataFrame
        Halo catalog table.
    min_mass_msun_h : Optional[float]
        Minimum halo mass in Msun/h; ``None`` keeps all masses at the low end.
    max_mass_msun_h : Optional[float]
        Maximum halo mass in Msun/h; ``None`` keeps all masses at the high end.
    mass_column : str, optional
        Mass column name.

    Returns
    -------
    pd.DataFrame
        Filtered table with a reset index.

    Raises
    ------
    ValueError
        If ``min_mass_msun_h`` exceeds ``max_mass_msun_h`` when both are set.
    """
    if min_mass_msun_h is None and max_mass_msun_h is None:
        return frame.reset_index(drop=True)
    if (
        min_mass_msun_h is not None
        and max_mass_msun_h is not None
        and min_mass_msun_h > max_mass_msun_h
    ):
        raise ValueError("min_mass_msun_h must be <= max_mass_msun_h when both are set.")

    mask = np.ones(len(frame), dtype=bool)
    masses = frame[mass_column].to_numpy()
    if min_mass_msun_h is not None:
        mask &= masses >= min_mass_msun_h
    if max_mass_msun_h is not None:
        mask &= masses <= max_mass_msun_h
    return frame.loc[mask].copy().reset_index(drop=True)


def filter_finite_input_features(
    catalog: pd.DataFrame,
    input_features: Sequence[str],
) -> pd.DataFrame:
    """
    Drop rows with non-finite values in Haloscope input feature columns.

    Parameters
    ----------
    catalog : pd.DataFrame
        Halo table.
    input_features : Sequence[str]
        Column names required for Haloscope ``fit`` / ``predict``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with a reset index.
    """
    mask = np.ones(len(catalog), dtype=bool)
    for column in input_features:
        mask &= np.isfinite(catalog[column].to_numpy(dtype=np.float64))
    return catalog.loc[mask].reset_index(drop=True)
