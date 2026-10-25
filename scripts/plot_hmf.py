import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from density_field_properties.pipelines.catalog_columns import (
    ROCKSTAR_LIST_CATALOG_LAYOUT,
    rockstar_catalog_column_kwargs,
)
from density_field_properties.read_data.halos.rockstar import RockstarCatalogReader

DEFAULT_BOX_SIZE_MPC_H = 1000.0

MIN_PARTICLES = 20
MP_UNIT_MSUN_H = 1.24e9
MP_FASTPM_MSUN_H = 9.97e9

BIN_EDGES_LOG10_MIN = 12.0
BIN_EDGES_LOG10_MAX = 15.5
BIN_EDGES_COUNT = 25
PLOT_MIN_LOG_MASS = 12.4
MATCHED_MIN_LOG_MASS = 12.6

ROCKSTAR_HALO_COLUMNS_POSITION_FASTPM = rockstar_catalog_column_kwargs(
    ROCKSTAR_LIST_CATALOG_LAYOUT
)


def load_halo_masses_from_catalog(
    catalog_path: Path,
    filter_by_pid: bool = True,
    params: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Load M200b halo masses from a Rockstar catalog file.

    Parameters
    ----------
    catalog_path : Path
        Path to the Rockstar list catalog.
    filter_by_pid : bool, optional
        If True, keep only main halos (pid == -1).
    params : dict[str, int] or None, optional
        Column layout overrides for ``RockstarCatalogReader``.

    Returns
    -------
    np.ndarray
        Halo masses in Msun/h.
    """
    if params is None:
        params = {}
    catalog_data = RockstarCatalogReader.read_catalog(
        str(catalog_path), filter_by_pid=filter_by_pid, **params
    )

    logging.info("Loaded %d halos from %s", len(catalog_data.halo_m200b), catalog_path)
    logging.info("\t - Min Mass %d", min(catalog_data.halo_m200b))
    logging.info("\t - Max Mass %d", max(catalog_data.halo_m200b))
    return catalog_data.halo_m200b


def halo_mass_function_from_masses(
    masses: np.ndarray,
    volume: float | int | None = None,
    return_argsort: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Build a cumulative halo mass function from a mass sample.

    Parameters
    ----------
    masses : np.ndarray
        Halo masses in Msun/h.
    volume : float or int, optional
        Simulation volume in (Mpc/h)^3. When set, number counts are divided by volume.
    return_argsort : bool, optional
        If True, return the descending sort index of ``masses``.

    Returns
    -------
    masses_sorted : np.ndarray
        Masses sorted from high to low.
    n_density : np.ndarray
        Cumulative halo number density (per volume if ``volume`` is set).
    argsort : np.ndarray or None
        Descending sort indices when ``return_argsort`` is True.
    """
    if return_argsort:
        argsort = np.argsort(masses)[::-1]
        masses_sorted = masses[argsort]
    else:
        argsort = None
        masses_sorted = np.sort(masses)[::-1]
    n_density = np.arange(len(masses_sorted), dtype=np.float64) + 1.0
    if volume:
        n_density /= float(volume)
    return masses_sorted, n_density, argsort


def halo_mass_function_from_masses_not_cumulative(
    masses: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a non-cumulative halo mass function from a mass sample.

    Parameters
    ----------
    masses : np.ndarray
        Halo masses in Msun/h.
    bin_edges : np.ndarray or None, optional
        Bin edges to use for the mass function. If None, edges span the sample range.

    Returns
    -------
    bin_centers : np.ndarray
        Bin centers.
    n_density : np.ndarray
        Number of halos in each bin.
    """
    if bin_edges is None:
        bin_edges = np.logspace(np.log10(masses.min()), np.log10(masses.max()), 100)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    n_density, _ = np.histogram(masses, bins=bin_edges)
    return bin_centers, n_density


def abundance_matching_calibration(
    masses_reference_sorted: np.ndarray,
    n_density_reference: np.ndarray,
    n_density_target: np.ndarray,
) -> np.ndarray:
    """
    Map target cumulative densities onto reference masses via log-log interpolation.

    Parameters
    ----------
    masses_reference_sorted : np.ndarray
        Reference masses sorted from high to low (same order as ``n_density_reference``).
    n_density_reference : np.ndarray
        Cumulative number density of the reference sample.
    n_density_target : np.ndarray
        Cumulative number density of the target sample (one value per halo to remap).

    Returns
    -------
    np.ndarray
        Target masses on the reference cumulative mass function, sorted high to low.
    """
    log_density_reference = np.log10(n_density_reference)
    log_density_target = np.log10(n_density_target)
    log_mass_reference = np.log10(masses_reference_sorted)
    log_mass_new = np.interp(
        log_density_target,
        log_density_reference,
        log_mass_reference,
    )
    return np.power(10.0, log_mass_new)


def load_hmf_curves_from_csv(
    csv_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load binned HMF curves written by this script's CSV export.

    Parameters
    ----------
    csv_path : Path
        Path to the comma-separated export (six data rows after the header).

    Returns
    -------
    bin_centers_unit : np.ndarray
        UNIT bin centers in Msun/h.
    n_density_unit : np.ndarray
        UNIT counts per bin.
    bin_centers_fastpm : np.ndarray
        FastPM bin centers in Msun/h.
    n_density_fastpm : np.ndarray
        FastPM counts per bin.
    bin_centers_fastpm_new : np.ndarray
        Abundance-matched FastPM bin centers in Msun/h.
    n_density_fastpm_new : np.ndarray
        Abundance-matched FastPM counts per bin.
    """
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]


def plot_binned_hmf_log_axes(
    bin_centers_unit: np.ndarray,
    n_density_unit: np.ndarray,
    bin_centers_fastpm: np.ndarray,
    n_density_fastpm: np.ndarray,
    bin_centers_fastpm_new: np.ndarray,
    n_density_fastpm_new: np.ndarray,
    min_particles: int,
    output_path: Path,
) -> None:
    """
    Plot UNIT vs FastPM binned HMFs with log10 axes and minor tick marks.

    Parameters
    ----------
    bin_centers_unit : np.ndarray
        UNIT bin centers in Msun/h.
    n_density_unit : np.ndarray
        UNIT counts per bin.
    bin_centers_fastpm : np.ndarray
        FastPM bin centers in Msun/h.
    n_density_fastpm : np.ndarray
        FastPM counts per bin.
    bin_centers_fastpm_new : np.ndarray
        FastPM abundance-matched bin centers in Msun/h.
    n_density_fastpm_new : np.ndarray
        FastPM abundance-matched counts per bin.
    min_particles : int
        Minimum particle count used for the sample (plot title).
    output_path : Path
        Destination image path.
    fastpm_label : str, optional
        Legend label for the FastPM curve.
    """
    log_mass_unit = np.log10(bin_centers_unit)
    unit_mask = (log_mass_unit >= PLOT_MIN_LOG_MASS) & (n_density_unit > 0)

    log_mass_fastpm = np.log10(bin_centers_fastpm)
    fastpm_mask = (log_mass_fastpm >= PLOT_MIN_LOG_MASS) & (n_density_fastpm > 0)
    log_mass_fastpm_new = np.log10(bin_centers_fastpm_new)
    matched_mask = (log_mass_fastpm_new > MATCHED_MIN_LOG_MASS) & (n_density_fastpm_new > 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        log_mass_fastpm_new[matched_mask],
        np.log10(n_density_fastpm_new[matched_mask]),
        "-",
        label="FastPM (abundance-matched to UNIT)",
        linewidth=1.2,
    )
    ax.plot(
        log_mass_unit[unit_mask],
        np.log10(n_density_unit[unit_mask]),
        "--",
        label="UNIT",
        linewidth=1.2,
    )
    ax.plot(
        log_mass_fastpm[fastpm_mask],
        np.log10(n_density_fastpm[fastpm_mask]),
        "-",
        label="FastPM",
        linewidth=1.2,
    )
    ax.set_title(f"Halo mass function ($N_\\mathrm{{part}} \\geq {min_particles}$)")
    ax.set_xlabel(r"$\log_{10}(M_{200\mathrm{b}}\,[\mathrm{M}_\odot/h])$")
    ax.set_ylabel(r"$\log_{10}(n(M))$")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(left=PLOT_MIN_LOG_MASS)
    ax.grid(which="major", linestyle="-", alpha=0.35)
    ax.grid(which="minor", linestyle=":", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _apply_min_particle_cut(
    masses_unit: np.ndarray,
    masses_fastpm: np.ndarray,
    min_particles: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter halo masses by a minimum particle count threshold.

    Parameters
    ----------
    masses_unit : np.ndarray
        UNIT halo masses in Msun/h.
    masses_fastpm : np.ndarray
        FastPM halo masses in Msun/h.
    min_particles : int or None
        Minimum particles per halo; no filtering when None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered UNIT and FastPM mass arrays.
    """
    if min_particles is None:
        return masses_unit, masses_fastpm
    masses_unit = masses_unit[masses_unit > min_particles * MP_UNIT_MSUN_H]
    masses_fastpm = masses_fastpm[masses_fastpm > min_particles * MP_FASTPM_MSUN_H]
    return masses_unit, masses_fastpm


def _abundance_match_fastpm_to_unit(
    masses_unit: np.ndarray,
    masses_fastpm: np.ndarray,
    volume: float,
) -> np.ndarray:
    """
    Calibrate FastPM masses to the UNIT cumulative mass function.

    Parameters
    ----------
    masses_unit : np.ndarray
        UNIT halo masses in Msun/h.
    masses_fastpm : np.ndarray
        FastPM halo masses in Msun/h.
    volume : float
        Simulation volume in (Mpc/h)^3.

    Returns
    -------
    np.ndarray
        FastPM masses in original halo order, abundance-matched to UNIT.
    """
    masses_unit_sorted, n_density_unit, _ = halo_mass_function_from_masses(masses_unit, volume)
    masses_fastpm_sorted, n_density_fastpm, masses_fastpm_argsort = halo_mass_function_from_masses(
        masses_fastpm, volume, return_argsort=True
    )
    new_masses_sorted = abundance_matching_calibration(
        masses_unit_sorted,
        n_density_unit,
        n_density_fastpm,
    )
    new_masses = np.empty_like(masses_fastpm)
    new_masses[masses_fastpm_argsort] = new_masses_sorted
    return new_masses


def main(argv: list[str]) -> int:
    """
    Plot UNIT vs abundance-matched FastPM binned halo mass functions.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    int
        Process exit code.
    """
    args = _parse_args(argv)

    if args.hmf_csv is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (
            centers_unit,
            n_unit,
            centers_fastpm,
            n_fastpm,
            centers_fastpm_new,
            n_fastpm_new,
        ) = load_hmf_curves_from_csv(args.hmf_csv)
        plot_path = (
            args.output_dir
            / f"halo_mass_function_not_cumulative_np_particles_{args.min_particles: .0f}.png"
        )
        plot_binned_hmf_log_axes(
            centers_unit,
            n_unit,
            centers_fastpm,
            n_fastpm,
            centers_fastpm_new,
            n_fastpm_new,
            args.min_particles,
            plot_path,
        )
        logging.info("Saved plot to %s", plot_path)
        return 0

    if args.halo_catalog_unit is None or args.halo_catalog_fastpm is None:
        raise SystemExit(
            "Provide --halo-catalog-unit and --halo-catalog-fastpm, or use --hmf-csv."
        )

    volume = args.box_size_mpc_h**3
    filter_by_pid_unit = not args.all_halos

    logging.info("Loading halo masses catalogs")
    masses_fastpm = load_halo_masses_from_catalog(
        args.halo_catalog_fastpm,
        filter_by_pid=False,
        params=ROCKSTAR_HALO_COLUMNS_POSITION_FASTPM,
    )
    masses_unit = load_halo_masses_from_catalog(
        args.halo_catalog_unit,
        filter_by_pid=filter_by_pid_unit,
    )

    masses_unit, masses_fastpm = _apply_min_particle_cut(
        masses_unit, masses_fastpm, args.min_particles
    )

    logging.info("Calculating cumulative halo mass functions for abundance matching")
    new_masses = _abundance_match_fastpm_to_unit(masses_unit, masses_fastpm, volume)
    logging.info(
        "Per-halo calibrated masses: min=%.3e max=%.3e",
        new_masses.min(),
        new_masses.max(),
    )

    bin_edges = np.logspace(BIN_EDGES_LOG10_MIN, BIN_EDGES_LOG10_MAX, BIN_EDGES_COUNT)

    logging.info("Calculating binned halo mass functions")
    bins_centers_unit, n_density_unit = halo_mass_function_from_masses_not_cumulative(
        masses_unit, bin_edges=bin_edges
    )
    bins_centers_fastpm, n_density_fastpm = halo_mass_function_from_masses_not_cumulative(
        masses_fastpm, bin_edges=bin_edges
    )
    bins_centers_fastpm_new, n_density_fastpm_new = halo_mass_function_from_masses_not_cumulative(
        new_masses, bin_edges=bin_edges
    )

    logging.info("Plotting halo mass function")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = (
        args.output_dir
        / f"halo_mass_function_not_cumulative_np_particles_{args.min_particles: .0f}.png"
    )
    plot_binned_hmf_log_axes(
        bins_centers_unit,
        n_density_unit,
        bins_centers_fastpm,
        n_density_fastpm,
        bins_centers_fastpm_new,
        n_density_fastpm_new,
        args.min_particles,
        plot_path,
    )
    logging.info("Saved plot to %s", plot_path)

    logging.info("Saving HMF curve data to CSV")
    hmf_columns = [
        bins_centers_unit,
        n_density_unit,
        bins_centers_fastpm,
        n_density_fastpm,
        bins_centers_fastpm_new,
        n_density_fastpm_new,
    ]
    hmf_data_path = (
        args.output_dir
        / f"halo_mass_function_not_cumulative_np_particles_{args.min_particles: .0f}.csv"
    )
    np.savetxt(
        hmf_data_path,
        hmf_columns,
        delimiter=",",
        header="Min Particles: " + str(args.min_particles),
    )
    logging.info("Saved HMF curve data to %s", hmf_data_path)

    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the halo mass function plot script.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot UNIT vs FastPM binned HMFs after cumulative-density "
            "abundance matching of FastPM masses to UNIT."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hmf-csv",
        type=Path,
        default=None,
        help="Replot from an existing CSV export (skips catalog loading).",
    )
    parser.add_argument("--halo-catalog-unit", type=Path, required=False)
    parser.add_argument("--halo-catalog-fastpm", type=Path, required=False)
    parser.add_argument(
        "--min-particles",
        type=int,
        default=MIN_PARTICLES,
        help="Minimum number of particles per halo (default: 20).",
    )
    parser.add_argument(
        "--box-size-mpc-h",
        type=float,
        default=DEFAULT_BOX_SIZE_MPC_H,
        help="Simulation box size in Mpc/h (default: 1000).",
    )
    parser.add_argument(
        "--all-halos",
        action="store_true",
        help="Use all halos, not only main halos.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(handler)
    sys.exit(main(sys.argv[1:]))
