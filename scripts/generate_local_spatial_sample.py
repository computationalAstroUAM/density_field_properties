#!/usr/bin/env python
"""
Build a spatially coherent local sample for UNITSIM and FastPM MN5 data.

The sample is a 200 Mpc/h cube with corners at [400, 400, 400] and
[600, 600, 600] Mpc/h inside the 1000 Mpc/h periodic box.

Run on the cluster (where /data21 is mounted), then rsync ``output/local_sample/``
to your local machine. Examples:

    PYTHONPATH=src python scripts/generate_local_spatial_sample.py

    PYTHONPATH=src python scripts/generate_local_spatial_sample.py --tasks fastpm_fof

    PYTHONPATH=src python scripts/generate_local_spatial_sample.py \\
        --tasks unitsim_dm fastpm_snap
"""

import argparse
import bz2
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from bigfile import BigFile
from default_params import BATCH_SIZE

from density_field_properties.density_field.particle_io import iter_dm_particle_batches
from density_field_properties.haloscope.sim_to_fastpm.config import (
    ROCKSTAR_LIST_COLUMNS,
    UNIT_HLIST_COLUMNS,
)

UNITSIM_HLIST_POSITION_COLUMNS = {
    "halo_x": UNIT_HLIST_COLUMNS["x"],
    "halo_y": UNIT_HLIST_COLUMNS["y"],
    "halo_z": UNIT_HLIST_COLUMNS["z"],
}

BOX_MIN_MPC_H = 400.0
BOX_MAX_MPC_H = 600.0

DEFAULT_UNITSIM_DM = Path(
    "/data21/UNITSIM/fixedAmp_InvPhase_001/DM_PARTICLES/dm_particles_0.5_128.bz2"
)
DEFAULT_UNITSIM_ROCKSTAR = Path(
    "/data21/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR/outputs/hlists/hlist_1.00000.list.bz2"
)
DEFAULT_FASTPM_BASE = Path("/data21/users/mruiz/fastpm_MN5/fastpm_tfm")
DEFAULT_OUTPUT_DIR = Path("output/local_sample")

ALL_TASKS = (
    "unitsim_dm",
    "unitsim_rockstar",
    "fastpm_nbody",
    "fastpm_pm",
    "fastpm_fof",
    "fastpm_snap",
)

TASK_SOURCE_KEYS = {
    "unitsim_dm": ("unitsim_dm",),
    "unitsim_rockstar": ("unitsim_rockstar",),
    "fastpm_nbody": ("fastpm_nbody",),
    "fastpm_pm": ("fastpm_pm",),
    "fastpm_fof": ("fastpm_fof",),
    "fastpm_snap": ("fastpm_snap",),
}


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Extract a shared 200 Mpc/h spatial sample for UNITSIM and FastPM."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unitsim-dm", type=Path, default=DEFAULT_UNITSIM_DM)
    parser.add_argument("--unitsim-rockstar", type=Path, default=DEFAULT_UNITSIM_ROCKSTAR)
    parser.add_argument("--fastpm-base", type=Path, default=DEFAULT_FASTPM_BASE)
    parser.add_argument("--box-min", type=float, default=BOX_MIN_MPC_H)
    parser.add_argument("--box-max", type=float, default=BOX_MAX_MPC_H)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS,
        default=list(ALL_TASKS),
        metavar="TASK",
        help=("Outputs to generate. Choices: %(choices)s. " "Default: all tasks."),
    )
    args = parser.parse_args(sys.argv[1:])
    selected_tasks = set(args.tasks)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    box_min = args.box_min
    box_max = args.box_max
    output_dir = args.output_dir
    unitsim_dir = output_dir / "unitsim"
    fastpm_dir = output_dir / "fastpm"
    unitsim_dir.mkdir(parents=True, exist_ok=True)
    (fastpm_dir / "rockstar_out_nbody").mkdir(parents=True, exist_ok=True)
    (fastpm_dir / "rockstar_out_pm").mkdir(parents=True, exist_ok=True)

    source_paths = {
        "unitsim_dm": args.unitsim_dm,
        "unitsim_rockstar": args.unitsim_rockstar,
        "fastpm_nbody": args.fastpm_base / "rockstar_out_nbody" / "out_8.list",
        "fastpm_pm": args.fastpm_base / "rockstar_out_pm" / "out_8.list",
        "fastpm_fof": args.fastpm_base / "output_01" / "fof_1.0000",
        "fastpm_snap": args.fastpm_base / "output_01" / "snap_1.0000",
    }
    required_source_keys = {
        source_key for task_name in selected_tasks for source_key in TASK_SOURCE_KEYS[task_name]
    }
    for label in sorted(required_source_keys):
        path = source_paths[label]
        if not path.exists():
            raise FileNotFoundError(f"Missing source for {label}: {path}")

    manifest_path = output_dir / "sample_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as manifest_handle:
            manifest = json.load(manifest_handle)
        manifest.setdefault("outputs", {})
        manifest.setdefault("counts", {})
        manifest.setdefault("sources", {})
    else:
        manifest = {
            "box_min_mpc_h": box_min,
            "box_max_mpc_h": box_max,
            "box_size_mpc_h": box_max - box_min,
            "sources": {},
            "outputs": {},
            "counts": {},
        }

    manifest["box_min_mpc_h"] = box_min
    manifest["box_max_mpc_h"] = box_max
    manifest["box_size_mpc_h"] = box_max - box_min
    manifest["tasks_run"] = sorted(selected_tasks)
    for label in sorted(required_source_keys):
        manifest["sources"][label] = str(source_paths[label])

    if "unitsim_dm" in selected_tasks:
        logging.info(
            "Filtering UNITSIM DM particles into %s", unitsim_dir / "dm_particles_0.5_128.dat"
        )
        unitsim_dm_out = unitsim_dir / "dm_particles_0.5_128.dat"
        unitsim_dm_count = 0
        dm_batch = []
        with (
            bz2.open(args.unitsim_dm, "rt") as dm_handle,
            open(unitsim_dm_out, "w", encoding="utf-8") as dm_out,
        ):
            for line in dm_handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                if (
                    x < box_min
                    or x > box_max
                    or y < box_min
                    or y > box_max
                    or z < box_min
                    or z > box_max
                ):
                    continue
                dm_batch.append([x, y, z])
                if len(dm_batch) >= BATCH_SIZE:
                    np.savetxt(dm_out, dm_batch, fmt="%.18e")
                    unitsim_dm_count += len(dm_batch)
                    dm_batch = []
            if dm_batch:
                np.savetxt(dm_out, dm_batch, fmt="%.18e")
                unitsim_dm_count += len(dm_batch)
        manifest["counts"]["unitsim_dm_particles"] = unitsim_dm_count
        manifest["outputs"]["unitsim_dm_particles"] = str(unitsim_dm_out)

    rockstar_catalog_jobs = []
    if "unitsim_rockstar" in selected_tasks:
        rockstar_catalog_jobs.append(
            (
                "unitsim_hlist",
                args.unitsim_rockstar,
                unitsim_dir / "hlist_1.00000.list",
                UNITSIM_HLIST_POSITION_COLUMNS,
            )
        )
    if "fastpm_nbody" in selected_tasks:
        rockstar_catalog_jobs.append(
            (
                "fastpm_nbody_out_8",
                source_paths["fastpm_nbody"],
                fastpm_dir / "rockstar_out_nbody" / "out_8.list",
                ROCKSTAR_LIST_COLUMNS,
            )
        )
    if "fastpm_pm" in selected_tasks:
        rockstar_catalog_jobs.append(
            (
                "fastpm_pm_out_8",
                source_paths["fastpm_pm"],
                fastpm_dir / "rockstar_out_pm" / "out_8.list",
                ROCKSTAR_LIST_COLUMNS,
            )
        )

    for job_name, source_path, target_path, column_map in rockstar_catalog_jobs:
        logging.info("Filtering Rockstar catalog %s", source_path)
        opener = bz2.open if source_path.suffix == ".bz2" else open
        mode = "rt" if source_path.suffix == ".bz2" else "r"
        halo_count = 0
        with (
            opener(source_path, mode) as catalog_in,
            open(target_path, "w", encoding="utf-8") as catalog_out,
        ):
            for line in catalog_in:
                if line.startswith("#"):
                    catalog_out.write(line if line.endswith("\n") else f"{line}\n")
                    continue
                cols = line.split()
                x = float(cols[column_map["halo_x"]])
                y = float(cols[column_map["halo_y"]])
                z = float(cols[column_map["halo_z"]])
                if (
                    x < box_min
                    or x > box_max
                    or y < box_min
                    or y > box_max
                    or z < box_min
                    or z > box_max
                ):
                    continue
                catalog_out.write(line if line.endswith("\n") else f"{line}\n")
                halo_count += 1
        manifest["counts"][job_name] = halo_count
        manifest["outputs"][job_name] = str(target_path)

    if "fastpm_fof" in selected_tasks:
        logging.info("Filtering FastPM FoF catalog from %s", source_paths["fastpm_fof"])
        src_fof_root = source_paths["fastpm_fof"]
        out_fof_root = fastpm_dir / "fof_1.0000"
        src_fof_bf = BigFile(str(src_fof_root))

        fof_group_name = None
        for block_name in src_fof_bf.blocks:
            if block_name == "Header":
                continue
            group_blocks = src_fof_bf[f"{block_name}/"].blocks
            if {"ID", "Position", "Length"}.issubset(group_blocks):
                fof_group_name = block_name
                break
        if fof_group_name is None:
            raise ValueError(
                f"No FoF catalog group with ID/Position/Length found under {src_fof_root}"
            )

        fof_group = src_fof_bf[f"{fof_group_name}/"]
        fof_column_names = sorted(fof_group.blocks)
        fof_n_halos = fof_group.open("Position").size
        fof_selected = {column_name: [] for column_name in fof_column_names}

        for batch_start in range(0, fof_n_halos, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, fof_n_halos)
            positions = np.asarray(
                fof_group.open("Position")[batch_start:batch_end], dtype=np.float64
            )
            in_box = (
                (positions[:, 0] >= box_min)
                & (positions[:, 0] <= box_max)
                & (positions[:, 1] >= box_min)
                & (positions[:, 1] <= box_max)
                & (positions[:, 2] >= box_min)
                & (positions[:, 2] <= box_max)
            )
            if not np.any(in_box):
                continue
            for column_name in fof_column_names:
                column_values = np.asarray(fof_group.open(column_name)[batch_start:batch_end])[
                    in_box
                ]
                fof_selected[column_name].append(column_values)

        if out_fof_root.exists():
            shutil.rmtree(out_fof_root)
        out_fof_root.mkdir(parents=True, exist_ok=True)
        out_fof_bf = BigFile(str(out_fof_root), create=True)

        fof_n_selected = 0
        for column_name in fof_column_names:
            src_column = fof_group.open(column_name)
            if fof_selected[column_name]:
                column_data = np.concatenate(fof_selected[column_name], axis=0)
            else:
                column_data = np.empty((0,) + src_column.dtype.shape, dtype=src_column.dtype.base)
            out_fof_bf.create_from_array(
                f"{fof_group_name}/{column_name}",
                column_data.astype(src_column.dtype.base, copy=False),
            )
            if column_name == "ID":
                fof_n_selected = int(column_data.shape[0])

        shutil.copytree(src_fof_root / "Header", out_fof_root / "Header", dirs_exist_ok=True)

        manifest["counts"]["fastpm_fof_halos"] = fof_n_selected
        manifest["outputs"]["fastpm_fof"] = str(out_fof_root)
        manifest["fastpm_fof_group"] = fof_group_name

    if "fastpm_snap" in selected_tasks:
        logging.info("Filtering FastPM snapshot from %s", source_paths["fastpm_snap"])
        src_snap_root = source_paths["fastpm_snap"]
        out_snap_root = fastpm_dir / "snap_1.0000"
        snap_positions_batches = []
        snap_block_path = str(src_snap_root / "1")
        for snap_batch, _, _ in iter_dm_particle_batches(snap_block_path, batch_size=BATCH_SIZE):
            in_box = (
                (snap_batch[:, 0] >= box_min)
                & (snap_batch[:, 0] <= box_max)
                & (snap_batch[:, 1] >= box_min)
                & (snap_batch[:, 1] <= box_max)
                & (snap_batch[:, 2] >= box_min)
                & (snap_batch[:, 2] <= box_max)
            )
            if np.any(in_box):
                snap_positions_batches.append(snap_batch[in_box])

        snap_positions = (
            np.concatenate(snap_positions_batches, axis=0)
            if snap_positions_batches
            else np.empty((0, 3), dtype=np.float64)
        )

        src_snap_bf = BigFile(str(src_snap_root))
        if out_snap_root.exists():
            shutil.rmtree(out_snap_root)
        out_snap_root.mkdir(parents=True, exist_ok=True)
        out_snap_bf = BigFile(str(out_snap_root), create=True)
        src_snap_position = src_snap_bf.open("1/Position")
        out_snap_bf.create_from_array(
            "1/Position",
            snap_positions.astype(src_snap_position.dtype.base, copy=False),
        )
        shutil.copytree(src_snap_root / "Header", out_snap_root / "Header", dirs_exist_ok=True)
        manifest["counts"]["fastpm_snap_particles"] = int(snap_positions.shape[0])
        manifest["outputs"]["fastpm_snap"] = str(out_snap_root)

    with open(manifest_path, "w", encoding="utf-8") as manifest_handle:
        json.dump(manifest, manifest_handle, indent=2)
        manifest_handle.write("\n")

    logging.info("Sample manifest written to %s", manifest_path)
    logging.info("Tasks completed: %s", sorted(selected_tasks))
    logging.info("Counts: %s", manifest["counts"])


if __name__ == "__main__":
    main()
