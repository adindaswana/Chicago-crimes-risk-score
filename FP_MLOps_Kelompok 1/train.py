"""
train.py - Skrip training end-to-end Checkpoint 2.

Menjalankan seluruh pipeline: muat data -> bangun protokol A/B/C -> tangga
baseline -> kandidat model -> ablasi target encoding -> permutation importance
-> pilih champion -> latih ulang champion pada seluruh dataset -> simpan
bundle, registry, metrics, dan output JSON representatif untuk Front-End.

Notebook `notebooks/02_model_training_baseline_comparison.ipynb` menjalankan
langkah yang sama secara naratif, mengimpor modul `src/` yang identik dengan
skrip ini - tidak ada logika yang hidup hanya di satu tempat.

Jalankan: python train.py
"""

from __future__ import annotations

import json

import pandas as pd

import features as F
import src.baselines as baselines
import src.evaluation as evaluation
import src.modeling as modeling
import src.registry as registry
import src.splits as splits
from src.config import (
    DISCLAIMER, LABEL_PARAMS_PATH, METRICS_PATH, MODEL_BUNDLE_PATH, MODEL_VERSION,
    OUTPUT_JSON_PATH, RANDOM_STATE, REGISTRY_PATH, risk_level,
)


def log(msg: str = "") -> None:
    print(msg)


def build_comparison_table(protocols: dict, ladders: dict, champions: dict) -> pd.DataFrame:
    """Satu tabel perbandingan final: protokol x (baseline/model) x metrik."""
    rows = []
    for protocol_name in ("A", "B", "C"):
        split = protocols[protocol_name]
        test_df = split["test"]
        y_true = test_df[F.TARGET_COLUMN].to_numpy()

        baseline_preds = baselines.predict_baseline_ladder(ladders[protocol_name], test_df)
        for label, pred in baseline_preds.items():
            rows.append(evaluation.evaluate_row(y_true, pred, protocol_name, label, "baseline"))

        model = champions[protocol_name]
        X_test = F.assemble_feature_matrix(test_df)
        pred_model = model.predict(X_test)
        rows.append(evaluation.evaluate_row(
            y_true, pred_model, protocol_name, f"Champion: {champions['name']}", "model"
        ))
    return evaluation.add_improvement_column(pd.DataFrame(rows))


def build_representative_output(bundle: dict, df: pd.DataFrame) -> dict:
    """Prediksi Risk Score untuk sekumpulan sel dan slot waktu representatif.

    Sel dipilih tersebar merata lintas kuintil rata-rata risk_score (bukan
    hanya sel paling berisiko), slot waktu mewakili pagi/malam hari kerja dan
    akhir pekan - representatif bagi skenario Front-End (safe-route & heatmap).
    """
    grid_lookup = F.make_grid_lookup(df)
    model = bundle["estimator"]

    cell_mean = df.groupby(["cell_id", "lat_r", "lon_r"])["risk_score"].mean().reset_index()
    cell_mean["quintile"] = pd.qcut(cell_mean["risk_score"], 5, labels=False, duplicates="drop")
    sampled_cells = (cell_mean.groupby("quintile", group_keys=False)
                      .apply(lambda g: g.sample(n=min(5, len(g)), random_state=RANDOM_STATE)))

    time_slots = [
        ("weekday_morning", pd.Timestamp("2026-04-14 09:00")),   # Selasa
        ("weekday_night", pd.Timestamp("2026-04-14 23:00")),     # Selasa
        ("weekend_night", pd.Timestamp("2026-04-11 23:00")),     # Sabtu
        ("weekend_early_morning", pd.Timestamp("2026-04-12 02:00")),  # Minggu
    ]

    predictions = []
    for _, cell in sampled_cells.iterrows():
        for slot_label, dt in time_slots:
            feat = F.build_features(cell["lat_r"], cell["lon_r"], dt, grid_lookup)
            X = F.assemble_feature_matrix(pd.DataFrame([feat]))
            raw_score = float(model.predict(X)[0])
            score = float(min(max(raw_score, 0.0), 100.0))
            predictions.append({
                "cell_id": feat["cell_id"],
                "lat": feat["lat_r"],
                "lon": feat["lon_r"],
                "datetime": dt.isoformat(),
                "time_slot_label": slot_label,
                "risk_score": round(score, 2),
                "level": risk_level(score),
                "data_coverage": "historical_data" if feat["cell_total_count"] > 0 else "no_historical_data",
            })

    return {
        "model_version": bundle["model_version"],
        "last_updated": bundle["label_params"]["reference_date"][:10],
        "algorithm": bundle["algorithm"],
        "disclaimer": DISCLAIMER,
        "n_predictions": len(predictions),
        "predictions": predictions,
    }


def main() -> None:
    log("=== 1. Muat dataset dan bangun protokol evaluasi tiga-lapis ===")
    df = splits.load_dataset()
    with open(LABEL_PARAMS_PATH) as f:
        label_params = json.load(f)
    protocols = splits.build_all_protocols(df)
    log(f"Total baris: {len(df):,} | sel unik: {df['cell_id'].nunique()}")
    log(f"Persentase sel tak dikenal - protokol A: {protocols['stats']['A_unseen_cell_pct']:.2f}%"
        f" | protokol B: {protocols['stats']['B_unseen_cell_pct']:.2f}%")
    assert protocols["stats"]["A_unseen_cell_pct"] < 1.0, "Protokol A seharusnya ~0% sel tak dikenal"
    assert protocols["stats"]["B_unseen_cell_pct"] == 100.0, "Protokol B seharusnya 100% sel tak dikenal"

    log("\n=== 2. Tangga baseline empat tingkat (A, B, C berbagi train dgn B) ===")
    ladder_a = baselines.fit_baseline_ladder(protocols["A"]["train"])
    ladder_b = baselines.fit_baseline_ladder(protocols["B"]["train"])
    ladders = {"A": ladder_a, "B": ladder_b, "C": ladder_b}
    log(f"Shrinkage m terpilih - protokol A: {ladder_a['best_m']} | protokol B: {ladder_b['best_m']}")

    log("\n=== 3. Latih kandidat model pada protokol B, pilih champion ===")
    comparison_candidates, fitted_b = modeling.train_candidates(protocols["B"]["train"], protocols["B"]["test"])
    log(comparison_candidates.round(3).to_string(index=False))
    champion_name, champion_reason = modeling.select_champion(comparison_candidates)
    log(f"\nChampion terpilih: {champion_name}")
    log(f"Alasan: {champion_reason}")

    champion_a = modeling.make_model(champion_name)
    champion_a.fit(F.assemble_feature_matrix(protocols["A"]["train"]),
                    protocols["A"]["train"][F.TARGET_COLUMN].to_numpy())
    champions = {"A": champion_a, "B": fitted_b[champion_name], "C": fitted_b[champion_name], "name": champion_name}

    log("\n=== 4. Tabel perbandingan final (baseline x model, protokol A/B/C) ===")
    comparison_table = build_comparison_table(protocols, ladders, champions)
    log(comparison_table.round(3).to_string(index=False))

    log("\n=== 5. Ablasi target encoding (cell_risk_mean), protokol A vs B ===")
    ablation_table = modeling.run_target_encoding_ablation(protocols, algo_name=champion_name)
    log(ablation_table.round(3).to_string(index=False))

    log("\n=== 6. Evaluasi bersegmen (protokol B, model vs baseline terkuat) ===")
    baseline_b_strongest = baselines.predict_baseline_ladder(ladder_b, protocols["B"]["test"])[
        baselines.BASELINE_LABELS["cell_hour_shrunk"]
    ]
    segmented = evaluation.segmented_evaluation(
        protocols["B"]["test"], protocols["B"]["test"][F.TARGET_COLUMN].to_numpy(),
        champions["B"].predict(F.assemble_feature_matrix(protocols["B"]["test"])),
        baseline_b_strongest,
    )
    log(f"Spearman model: {segmented['spearman_model']:.3f} | Spearman baseline: {segmented['spearman_baseline']:.3f}")
    log("MAE per kuintil:\n" + segmented["per_quintile"].round(3).to_string())

    log("\n=== 7. Permutation importance (protokol B) ===")
    importance_table = modeling.compute_permutation_importance(
        champions["B"], F.assemble_feature_matrix(protocols["B"]["test"]),
        protocols["B"]["test"][F.TARGET_COLUMN].to_numpy(),
    )
    log(importance_table.round(3).to_string(index=False))

    log("\n=== 8. Latih ulang champion pada SELURUH dataset untuk bundle produksi ===")
    log("Holdout hanya dipakai untuk mengukur generalisasi jujur; model yang dilayani API "
        "dilatih ulang pada 100% data agar tidak sengaja buta terhadap sel yang ditahan.")
    final_model = modeling.make_model(champion_name)
    final_model.fit(F.assemble_feature_matrix(df), df[F.TARGET_COLUMN].to_numpy())

    metrics_summary = {
        "protocol_A": comparison_table[comparison_table["protocol"] == "A"].to_dict(orient="records"),
        "protocol_B": comparison_table[comparison_table["protocol"] == "B"].to_dict(orient="records"),
        "protocol_C": comparison_table[comparison_table["protocol"] == "C"].to_dict(orient="records"),
    }

    bundle = registry.build_bundle(
        model=final_model, algorithm=champion_name, metrics_summary=metrics_summary,
        dataset_df=df, label_params=label_params, model_version=MODEL_VERSION,
    )
    artifact_path = registry.save_bundle(bundle, MODEL_BUNDLE_PATH)
    log(f"\nBundle disimpan: {artifact_path}")

    registry_entries = [registry.build_registry_entry(bundle, artifact_path, status="champion")]
    registry.write_registry(registry_entries, REGISTRY_PATH)
    log(f"Registry ditulis: {REGISTRY_PATH}")

    full_metrics = {
        "champion_algorithm": champion_name,
        "champion_selection_reason": champion_reason,
        "unseen_cell_verification": protocols["stats"],
        "shrinkage_m_selected": {"A": ladder_a["best_m"], "B": ladder_b["best_m"]},
        "candidate_comparison_protocol_B": comparison_candidates.to_dict(orient="records"),
        "final_comparison_table": comparison_table.to_dict(orient="records"),
        "target_encoding_ablation": ablation_table.to_dict(orient="records"),
        "segmented_evaluation_protocol_B": {
            "per_quintile": segmented["per_quintile"].reset_index().to_dict(orient="records"),
            "per_hour_group": segmented["per_hour_group"].reset_index().to_dict(orient="records"),
            "large_error_fractions": segmented["large_error_fractions"],
            "worst_cells": segmented["worst_cells"].reset_index().to_dict(orient="records"),
            "spearman_model": segmented["spearman_model"],
            "spearman_baseline": segmented["spearman_baseline"],
        },
        "permutation_importance_protocol_B": importance_table.to_dict(orient="records"),
        "dataset_fingerprint": bundle["dataset_fingerprint"],
    }
    registry.write_metrics(full_metrics, path=METRICS_PATH)
    log("Metrics ditulis.")

    log("\n=== 9. Output JSON representatif untuk Front-End ===")
    output_payload = build_representative_output(bundle, df)
    registry.write_json(output_payload, OUTPUT_JSON_PATH)
    log(f"Output ditulis: {OUTPUT_JSON_PATH} ({output_payload['n_predictions']} prediksi)")

    log("\n=== SELESAI ===")
    log(f"Champion: {champion_name} | headline protokol B MAE model vs baseline terkuat tersedia di metrics.json")


if __name__ == "__main__":
    main()
