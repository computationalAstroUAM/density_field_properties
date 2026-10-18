"""Optional halo mass range filters for Haloscope preprocessing."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class HaloMassFilterConfig:
    """
    Keep halos within an optional ``M200b`` mass range.

    Parameters
    ----------
    min_mass_msun_h : Optional[float]
        Minimum halo mass in Msun/h; ``None`` disables the lower bound.
    max_mass_msun_h : Optional[float]
        Maximum halo mass in Msun/h; ``None`` disables the upper bound.
    mass_column : str
        Mass column name applied to HR and LR tables before calibration.
    """

    min_mass_msun_h: Optional[float] = None
    max_mass_msun_h: Optional[float] = None
    mass_column: str = "M200b"

    def __post_init__(self) -> None:
        if self.min_mass_msun_h is not None and self.max_mass_msun_h is not None:
            if self.min_mass_msun_h > self.max_mass_msun_h:
                raise ValueError(
                    "min_mass_msun_h must be <= max_mass_msun_h when both are set."
                )

    @property
    def enabled(self) -> bool:
        """
        Whether any mass bound is active.

        Returns
        -------
        bool
            True when a minimum or maximum mass threshold is set.
        """
        return self.min_mass_msun_h is not None or self.max_mass_msun_h is not None


def default_halo_mass_filter_config() -> HaloMassFilterConfig:
    """
    Build a disabled halo mass filter (no rows removed).

    Returns
    -------
    HaloMassFilterConfig
        Filter with both bounds unset.
    """
    return HaloMassFilterConfig()


def parse_halo_mass_filter_config(sample: Mapping[str, Any]) -> HaloMassFilterConfig:
    """
    Parse halo mass filter settings from a JSON ``sample`` object.

    Parameters
    ----------
    sample : Mapping[str, Any]
        ``sample`` section of a Haloscope run config.

    Returns
    -------
    HaloMassFilterConfig
        Parsed mass filter; disabled when bounds are omitted or null.
    """
    min_mass = _optional_positive_float(sample.get("min_mass_msun_h"))
    max_mass = _optional_positive_float(sample.get("max_mass_msun_h"))
    return HaloMassFilterConfig(
        min_mass_msun_h=min_mass,
        max_mass_msun_h=max_mass,
    )


def _optional_positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError("Halo mass filter bounds must be strictly positive.")
    return parsed
