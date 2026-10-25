"""Rockstar and consistent-trees column index maps for catalog readers."""

ROCKSTAR_T_OVER_U_COLUMN = 37
UNIT_T_OVER_U_COLUMN = 56

UNIT_HLIST_COLUMNS = {
    "id": 1,
    "pid": 5,
    "Rvir": 11,
    "x": 17,
    "y": 18,
    "z": 19,
    "Spin": 26,
    "Rs_Klypin": 37,
    "M200b": 39,
    "ba": 46,
    "ca": 47,
    "t_over_u": UNIT_T_OVER_U_COLUMN,
}

UNIT_HLIST_TRAINING_EXTRA_COLUMN_POSITIONS = {
    "Spin": UNIT_HLIST_COLUMNS["Spin"],
    "Rs_Klypin": UNIT_HLIST_COLUMNS["Rs_Klypin"],
    "Rvir": UNIT_HLIST_COLUMNS["Rvir"],
    "ba": UNIT_HLIST_COLUMNS["ba"],
    "ca": UNIT_HLIST_COLUMNS["ca"],
}

ROCKSTAR_LIST_COLUMNS = {
    "halo_id": 0,
    "halo_pid": 40,
    "halo_x": 8,
    "halo_y": 9,
    "halo_z": 10,
    "halo_m200b": 20,
    "t_over_u": ROCKSTAR_T_OVER_U_COLUMN,
}
