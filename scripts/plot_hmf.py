import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from density_field_properties.pipelines.catalog_columns import (
    ROCKSTAR_LIST_CATALOG_LAYOUT,
    rockstar_catalog_column_kwargs,
)
from density_field_properties.read_data.halos.rockstar import RockstarCatalogReader

DEFAULT_BOX_SIZE_MPC_H = 1000.0

MIN_PARTICLES = 20
MP_UNIT_MSUN_H = 1.24e9 # Msun/h
MP_FASTPM_MSUN_H = 9.97e9 # Msun/h


ROCKSTAR_HALO_COLUMNS_POSITION_FASTPM = rockstar_catalog_column_kwargs(
    ROCKSTAR_LIST_CATALOG_LAYOUT
)


def load_halo_masses_from_catalog(catalog_path: Path, filter_by_pid: bool = True, params: dict[str, int] | None = None) -> np.ndarray:
    if params is None:
        params = ROCKSTAR_HALO_COLUMNS_POSITION_FASTPM
    catalog_data = RockstarCatalogReader.read_catalog(catalog_path, filter_by_pid=filter_by_pid, **params)

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


def main(argv: list[str]) -> int:
    """
    Plot UNIT and FastPM halo mass functions with density-interpolated matching.

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
    volume = args.box_size_mpc_h**3

    filter_by_pid = not args.all_halos

    logging.info("Loading halo masses catalogs")
    masses_fastpm = load_halo_masses_from_catalog(args.halo_catalog_fastpm, filter_by_pid=filter_by_pid, params=ROCKSTAR_HALO_COLUMNS_POSITION_FASTPM)
    masses_unit = load_halo_masses_from_catalog(args.halo_catalog_unit, filter_by_pid=filter_by_pid)
    

    if args.min_particles is not None:
        masses_unit = masses_unit[masses_unit > args.min_particles * MP_UNIT_MSUN_H]
        masses_fastpm = masses_fastpm[masses_fastpm > args.min_particles * MP_FASTPM_MSUN_H]

    logging.info("Calculating halo mass functions")
    masses_unit_sorted, n_density_unit, _ = halo_mass_function_from_masses(
        masses_unit, volume
    )
    masses_fastpm_sorted, n_density_fastpm, masses_fastpm_argsort = (
        halo_mass_function_from_masses(masses_fastpm, volume, return_argsort=True)
    )

    logging.info("Abundance matching calibration")
    new_masses_sorted = abundance_matching_calibration(
        masses_unit_sorted,
        n_density_unit,
        n_density_fastpm,
    )
    new_masses = np.empty_like(masses_fastpm)
    new_masses[masses_fastpm_argsort] = new_masses_sorted

    logging.info("Plotting halo mass function")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.title(
        f"Cumulative halo mass function ($N_\\mathrm{{part}} \\geq {args.min_particles}$)"
    )
    plt.plot(
        np.log10(masses_unit_sorted),
        np.log10(n_density_unit),
        label="UNIT",
    )
    plt.plot(
        np.log10(masses_fastpm_sorted),
        np.log10(n_density_fastpm),
        label="FastPM",
    )
    plt.plot(
        np.log10(new_masses_sorted),
        np.log10(n_density_fastpm),
        label="FastPM (abundance-matched to UNIT)",
    )
    plt.xlabel(r"$\log_{10}(M_{200\mathrm{b}}\,[\mathrm{M}_\odot/h])$")
    plt.ylabel(r"$\log_{10}(n(>M)\,[\mathrm{h}^{3}\,\mathrm{Mpc}^{-3}])$")
    plt.legend()
    plt.savefig(
        args.output_dir / f"halo_mass_function_np_particles_{args.min_particles: .0f}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


    logging.info("Saving HMF curve data to CSV")
    n_unit = len(masses_unit_sorted)
    n_fastpm = len(masses_fastpm_sorted)
    n_points = 10000
    hmf_columns = [
        np.log10(masses_unit_sorted)[np.linspace(0, n_unit - 1, n_points)],
        np.log10(n_density_unit)[np.linspace(0, n_unit - 1, n_points)],
        np.log10(masses_fastpm_sorted)[np.linspace(0, n_fastpm - 1, n_points)],
        np.log10(n_density_fastpm)[np.linspace(0, n_fastpm - 1, n_points)],
        np.log10(new_masses_sorted)[np.linspace(0, n_fastpm - 1, n_points)],
        np.log10(n_density_fastpm)[np.linspace(0, n_fastpm - 1, n_points)],
    ]
    hmf_padded = np.full((n_points, len(hmf_columns)), np.nan, dtype=np.float64)
    for col_idx, column in enumerate(hmf_columns):
        hmf_padded[: len(column), col_idx] = column
    hmf_data_path = args.output_dir / f"halo_mass_function_np_particles_{args.min_particles: .0f}.csv"
    np.savetxt(
        hmf_data_path,
        hmf_padded,
        delimiter=",",
        header=(
            "Min Particles: " + str(args.min_particles) + "\n"
            "log10_M200b_UNIT,log10_n_density_UNIT,"
            "log10_M200b_FastPM,log10_n_density_FastPM,"
            "log10_M200b_FastPM_corrected,log10_n_density_FastPM_corrected"
        ),
        comments="",
    )
    logging.info("Saved HMF curve data to %s", hmf_data_path)

    logging.info(
        "Per-halo calibrated masses: min=%.3e max=%.3e",
        new_masses.min(),
        new_masses.max(),
    )
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
        description="Plot UNIT vs FastPM HMFs with cumulative-density mass matching."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--halo-catalog-unit", type=Path, required=True)
    parser.add_argument("--halo-catalog-fastpm", type=Path, required=True)
    parser.add_argument("--min-particles", type=int, default=MIN_PARTICLES, help="Minimum number of particles per halo (default: 20).")
    parser.add_argument(
        "--box-size-mpc-h",
        type=float,
        default=DEFAULT_BOX_SIZE_MPC_H,
        help="Simulation box size in Mpc/h (default: 1000).",
    )
    parser.add_argument("--all-halos", action="store_true", help="Use all halos, not only main halos.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)
    sys.exit(main(sys.argv[1:]))
