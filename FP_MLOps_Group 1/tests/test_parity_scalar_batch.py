"""
Uji parity skalar vs batch (WAJIB #2).

Jalur fitur endpoint batch (`api/batch.py`) mengganti N pemanggilan
`features.build_features()` dengan satu operasi merge. Uji ini membuktikan
hasilnya identik dengan memanggil jalur skalar satu per satu, untuk fitur
MAUPUN prediksi akhir.
"""

import numpy as np
import pandas as pd

import features as F
from api.batch import build_features_batch, predict_batch

N_SAMPLES = 150
RANDOM_SEED = 123


def test_batch_features_match_scalar_features(dataset, grid_lookup, to_datetime):
    sample = dataset.sample(n=N_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    dts = [to_datetime(r["dow"], r["hour"]) for _, r in sample.iterrows()]

    batch_df = build_features_batch(sample["lat_r"], sample["lon_r"], dts, grid_lookup)

    scalar_rows = [
        F.build_features(lat, lon, dt, grid_lookup)
        for lat, lon, dt in zip(sample["lat_r"], sample["lon_r"], dts)
    ]
    scalar_df = pd.DataFrame(scalar_rows)

    for col in F.FEATURE_COLUMNS:
        np.testing.assert_allclose(
            batch_df[col].to_numpy(dtype="float64"),
            scalar_df[col].to_numpy(dtype="float64"),
            rtol=1e-8, atol=1e-6, err_msg=f"kolom '{col}' berbeda antara batch dan skalar",
        )
    assert list(batch_df["cell_id"]) == list(scalar_df["cell_id"])


def test_batch_predictions_match_scalar_predictions(dataset, grid_lookup, bundle, to_datetime):
    sample = dataset.sample(n=N_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    dts = [to_datetime(r["dow"], r["hour"]) for _, r in sample.iterrows()]
    model = bundle["estimator"]

    batch_result = predict_batch(model, grid_lookup, sample["lat_r"], sample["lon_r"], dts)

    scalar_scores = []
    for lat, lon, dt in zip(sample["lat_r"], sample["lon_r"], dts):
        feat = F.build_features(lat, lon, dt, grid_lookup)
        X = F.assemble_feature_matrix(pd.DataFrame([feat]))
        raw = float(model.predict(X)[0])
        scalar_scores.append(min(max(raw, 0.0), 100.0))

    np.testing.assert_allclose(
        batch_result["risk_score"].to_numpy(), np.array(scalar_scores), rtol=1e-8, atol=1e-6,
    )
