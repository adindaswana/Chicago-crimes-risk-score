"""
Uji parity training-serving (WAJIB #1).

Ambil sejumlah titik acak dari `features_labels.csv`, hitung fitur lewat jalur
TRAINING (baris CSV langsung -> `assemble_feature_matrix`) dan lewat jalur
SERVING (`build_features` dari lat/lon/datetime mentah -> `assemble_feature_matrix`).
Keduanya harus identik dalam toleransi numerik - pertahanan utama terhadap bug
pembulatan grid dan `cell_id` berbasis representasi string float.
"""

import numpy as np
import pandas as pd

import features as F

N_SAMPLES = 200
RANDOM_SEED = 42


def test_serving_path_matches_training_path_on_random_sample(dataset, grid_lookup, to_datetime):
    sample = dataset.sample(n=N_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

    training_matrix = F.assemble_feature_matrix(sample)

    serving_rows = []
    for _, row in sample.iterrows():
        dt = to_datetime(row["dow"], row["hour"])
        feat = F.build_features(row["lat_r"], row["lon_r"], dt, grid_lookup)
        serving_rows.append(feat)
    serving_matrix = F.assemble_feature_matrix(pd.DataFrame(serving_rows))

    assert list(training_matrix.columns) == list(serving_matrix.columns)
    np.testing.assert_allclose(
        training_matrix.to_numpy(), serving_matrix.to_numpy(), rtol=1e-8, atol=1e-6,
    )


def test_serving_path_reconstructs_correct_cell_id(dataset, grid_lookup, to_datetime):
    sample = dataset.sample(n=50, random_state=7)
    for _, row in sample.iterrows():
        dt = to_datetime(row["dow"], row["hour"])
        feat = F.build_features(row["lat_r"], row["lon_r"], dt, grid_lookup)
        assert feat["cell_id"] == row["cell_id"]
        assert feat["dow"] == row["dow"]
        assert feat["hour"] == row["hour"]
