"""Kontrak data bersama untuk training, aplikasi, notebook, dan pengujian."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
APP_NAME = "PenguInsight"
PROJECT_DESCRIPTION = "Klasifikasi Spesies Penguin Berdasarkan Karakteristik Fisik Menggunakan Machine Learning"
PROJECT_TITLE = f"{APP_NAME}: {PROJECT_DESCRIPTION}"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
LABELS = ["Adelie", "Chinstrap", "Gentoo"]
DISPLAY = {
    "bill_length_mm": ("Panjang paruh", "mm"),
    "bill_depth_mm": ("Kedalaman paruh", "mm"),
    "flipper_length_mm": ("Panjang sirip", "mm"),
    "body_mass_g": ("Berat tubuh", "gram"),
}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "KNN": "knn.joblib",
    "Decision Tree": "decision_tree.joblib",
}
SOURCE = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Tolak data salah; izinkan missing parsial untuk imputer tersimpan.

    Tidak menebak pemisah desimal atau satuan. CSV wajib menggunakan titik
    desimal dan gram/mm. Indeks tidak di-reset supaya identitas baris terjaga.
    """
    if frame.columns.duplicated().any():
        raise ValueError("Nama kolom tidak boleh duplikat.")
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError("Kolom wajib belum lengkap: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Data kosong. Masukkan minimal satu penguin.")
    if len(frame) > 5000:
        raise ValueError("Maksimal 5.000 penguin per unggahan.")
    out = pd.DataFrame(index=frame.index)
    for name in FEATURES:
        raw = frame[name].replace(r"^\s*$", np.nan, regex=True)
        converted = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & converted.isna()).any():
            raise ValueError(f"{name} harus angka. CSV memakai titik desimal, bukan koma desimal.")
        if np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f"{name} tidak boleh tak hingga.")
        if (converted.dropna() <= 0).any():
            raise ValueError(f"{name} harus lebih besar dari nol.")
        out[name] = converted.astype(float)
    if out.isna().all(axis=1).any():
        raise ValueError("Setiap penguin harus memiliki minimal satu ukuran. Empat ukuran lengkap lebih disarankan.")
    return out


def warning_messages(frame: pd.DataFrame, training_ranges: dict) -> list[str]:
    warnings = []
    if frame.isna().any().any():
        warnings.append("Ukuran yang kosong diisi median training oleh pipeline; itu bukan pengukuran asli.")
    for name in FEATURES:
        lo, hi = training_ranges[name]
        if ((frame[name] < lo) | (frame[name] > hi)).any():
            label, unit = DISPLAY[name]
            warnings.append(f"{label} di luar rentang training ({lo:g}–{hi:g} {unit}). Prediksi perlu kehati-hatian.")
    return warnings


def load_artifacts(root: Path = ROOT):
    """Hanya muat joblib lokal yang dipercaya; jangan menerima model unggahan."""
    summary = read_json(root / "results" / "summary.json")
    path = root / "models" / "penguin_pipeline.joblib"
    if sha256(path) != summary["selected_model_sha256"]:
        raise ValueError("Model tidak cocok dengan ringkasan evaluasi. Jalankan ulang train.py.")
    pipeline = joblib.load(path)
    if list(pipeline.feature_names_in_) != FEATURES:
        raise ValueError("Kontrak fitur model tidak cocok dengan aplikasi.")
    if sorted(pipeline.classes_.tolist()) != LABELS:
        raise ValueError("Label model tidak cocok dengan aplikasi.")
    return pipeline, summary


def predict_frame(pipeline, frame: pd.DataFrame, summary: dict):
    clean = validate_inputs(frame)
    probs = pipeline.predict_proba(clean)
    out = clean.copy()
    out["prediksi"] = pipeline.predict(clean)
    out["skor_tertinggi"] = probs.max(axis=1)
    for idx, label in enumerate(pipeline.classes_):
        out[f"skor_{label}"] = probs[:, idx]
    out["jumlah_imputasi"] = clean.isna().sum(axis=1)
    return out, warning_messages(clean, summary["training_ranges"])
