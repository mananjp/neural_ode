"""Sepsis risk prediction with Neural ODE v1

Usage examples:

1) Predict risk for a specific patient from a full Dataset.csv:

    python sepsis_inference.py         --model-path ./neural_ode_sepsis.pt         --data-path ./Dataset.csv         --patient-id 3

2) Predict risk for a CSV that contains only a single patient's time series
   (same columns as Dataset.csv):

    python sepsis_inference.py         --model-path ./neural_ode_sepsis.pt         --data-path ./patient_3.csv

You must have:
- PyTorch installed
- torchdiffeq installed (pip install torchdiffeq)
- pandas, numpy

The model checkpoint is expected to be saved with the metadata format
used in the notebook (feature_cols, means, stds, etc.).
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except ImportError as e:
    raise ImportError("torchdiffeq is required. Install with `pip install torchdiffeq`.") from e


# -----------------------------
# Model definition (Neural ODE v1)
# -----------------------------


class ODEFunc(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, latent_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        return self.net(z)


class NeuralODESepsis(nn.Module):
    """Neural ODE model as used in the notebook (v1)."""

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.odefunc = ODEFunc(latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """t: [B, L], x: [B, L, D]"""
        B, L, D = x.shape
        x_last = x[:, -1, :]
        z0 = self.encoder(x_last)
        # integrate over a scaled time interval [0, 1]
        t_eval = torch.tensor([0.0, 1.0], device=x.device)
        z_traj = odeint(self.odefunc, z0, t_eval, method='rk4', options={'step_size': 0.1})
        zT = z_traj[-1]
        logits = self.classifier(zT).squeeze(-1)
        return logits


# -----------------------------
# Checkpoint loading + preprocessing
# -----------------------------


def load_neural_ode_v1_checkpoint(
    ckpt_path: str,
    device: torch.device | str = "cpu",
) -> Tuple[NeuralODESepsis, dict]:
    """Load NeuralODESepsis v1 checkpoint with metadata."""

    if isinstance(device, str) and device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    map_location = torch.device(device) if isinstance(device, str) else device
    if map_location.type == 'cuda' and not torch.cuda.is_available():
        map_location = torch.device('cpu')
        
    ckpt = torch.load(ckpt_path, map_location=map_location)

    input_dim = ckpt["input_dim"]
    latent_dim = ckpt.get("latent_dim", 32)
    feature_cols_ckpt = ckpt["feature_cols"]
    means_ckpt = ckpt["means"]
    stds_ckpt = ckpt["stds"]
    patient_col_ckpt = ckpt.get("patient_col", "Patient_ID")
    time_col_ckpt = ckpt.get("time_col", "Hour")
    label_col_ckpt = ckpt.get("label_col", "SepsisLabel")

    model = NeuralODESepsis(input_dim=input_dim, latent_dim=latent_dim).to(map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    means_series = pd.Series(means_ckpt)
    stds_series = pd.Series(stds_ckpt)

    meta = {
        "feature_cols": feature_cols_ckpt,
        "means": means_series,
        "stds": stds_series,
        "patient_col": patient_col_ckpt,
        "time_col": time_col_ckpt,
        "label_col": label_col_ckpt,
    }

    return model, meta


def build_window_from_patient_df(
    df_patient: pd.DataFrame,
    meta: dict,
    window_size: int = 24,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the last time window from a single patient's time series.

    Returns (t_tensor [1, L], x_tensor [1, L, D]).
    """

    device = torch.device(device)
    time_col = meta["time_col"]
    feature_cols = meta["feature_cols"]
    means = meta["means"]
    stds = meta["stds"]

    df_p = df_patient.sort_values(time_col).reset_index(drop=True)

    # Normalize features
    x = df_p[feature_cols].copy()
    x = (x - means[feature_cols]) / stds[feature_cols].replace(0, 1.0)
    x = x.ffill().bfill().fillna(0.0)

    times = df_p[time_col].values.astype(np.float32)
    feats = x.values.astype(np.float32)

    # Last window_size rows, pad from beginning if needed
    if len(df_p) >= window_size:
        t_window = times[-window_size:]
        x_window = feats[-window_size:]
    else:
        pad = window_size - len(df_p)
        t_window = np.pad(times, (pad, 0), mode="edge")
        x_window = np.pad(feats, ((pad, 0), (0, 0)), mode="constant")

    # Normalize time to start at 0
    t_window = t_window - t_window[0]

    t_tensor = torch.from_numpy(t_window).unsqueeze(0).to(device)
    x_tensor = torch.from_numpy(x_window).unsqueeze(0).to(device)

    return t_tensor, x_tensor


@torch.no_grad()
def predict_sepsis_risk(
    df_patient: pd.DataFrame,
    ckpt_path: str,
    window_size: int = 24,
    device: torch.device | str = "cpu",
) -> tuple[float, dict]:
    """High-level helper: load model, build window, return risk probability and meta."""

    model, meta = load_neural_ode_v1_checkpoint(ckpt_path=ckpt_path, device=device)
    t_tensor, x_tensor = build_window_from_patient_df(df_patient, meta, window_size=window_size, device=device)
    logits = model(t_tensor, x_tensor)
    prob = torch.sigmoid(logits).item()
    return prob, meta


# -----------------------------
# Patient summary for LLM or logging
# -----------------------------


def infer_key_features_for_summary(feature_cols: list[str]) -> list[str]:
    """Pick a subset of key vitals/labs for human/LLM summaries."""
    keywords = [
        "hr",
        "heart",
        "heartrate",
        "bp",
        "meanbp",
        "map",
        "temp",
        "temperature",
        "spo2",
        "o2sat",
        "o2",
        "resp",
        "rr",
        "lactate",
    ]
    selected = []
    lower_map = {c: c.lower() for c in feature_cols}
    for c in feature_cols:
        lc = lower_map[c]
        if any(k in lc for k in keywords):
            selected.append(c)
    selected = list(dict.fromkeys(selected))
    if not selected:
        selected = feature_cols[:6]
    else:
        selected = selected[:8]
    return selected


def compute_trend(series: pd.Series, eps: float = 1e-3) -> str:
    if series.empty:
        return "unknown"
    first = series.iloc[0]
    last = series.iloc[-1]
    diff = last - first
    if diff > eps:
        return "increasing"
    elif diff < -eps:
        return "decreasing"
    else:
        return "stable"


def build_patient_summary(
    df_patient: pd.DataFrame,
    risk_score: float,
    meta: dict,
    horizon_hours: int = 6,
    window_hours: int = 24,
) -> dict:
    time_col = meta["time_col"]
    feature_cols = meta["feature_cols"]

    df_p = df_patient.sort_values(time_col).reset_index(drop=True)

    # Demographics if present
    age = None
    sex = None
    for col in df_p.columns:
        lc = col.lower()
        if age is None and "age" in lc:
            age = float(df_p[col].iloc[0])
        if sex is None and ("gender" in lc or "sex" in lc):
            sex = str(df_p[col].iloc[0])

    key_features = infer_key_features_for_summary(feature_cols)

    vitals_summary = {}
    for feat in key_features:
        if feat not in df_p.columns:
            continue
        s = df_p[feat].dropna()
        if s.empty:
            continue
        vitals_summary[feat] = {
            "last": float(s.iloc[-1]),
            "min": float(s.min()),
            "max": float(s.max()),
            "trend": compute_trend(s),
        }

    patient_id = None
    patient_col = meta.get("patient_col", "Patient_ID")
    if patient_col in df_p.columns:
        try:
            patient_id = int(df_p[patient_col].iloc[0])
        except Exception:
            patient_id = str(df_p[patient_col].iloc[0])

    summary = {
        "patient_id": patient_id,
        "time_window_hours": window_hours,
        "sepsis_risk_next_hours": risk_score,
        "horizon_hours": horizon_hours,
        "demographics": {
            "age": age,
            "sex": sex,
        },
        "vitals_summary": vitals_summary,
    }

    return summary


# -----------------------------
# CLI entry point
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Sepsis risk prediction with Neural ODE v1")
    parser.add_argument("--model-path", type=str, required=True, help="Path to neural_ode_sepsis.pt checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to Dataset.csv or single-patient CSV")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Patient_ID to use from full dataset (ignored if data-path is single-patient CSV)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--window-size", type=int, default=24, help="Number of hours in input window")
    parser.add_argument("--horizon-hours", type=int, default=6, help="Prediction horizon in hours (for metadata only)")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load data
    df = pd.read_csv(args.data_path)

    # Select patient
    model, meta = load_neural_ode_v1_checkpoint(args.model_path, device=device)

    time_col = meta["time_col"]
    patient_col = meta["patient_col"]

    if args.patient_id is not None and patient_col in df.columns:
        # full dataset: filter by patient_id
        df_patient = df[df[patient_col].astype(str) == str(args.patient_id)].copy()
        if df_patient.empty:
            raise ValueError(f"No rows found for {patient_col} == {args.patient_id}")
    else:
        # assume data contains a single patient's time series already
        df_patient = df.copy()

    # Predict risk
    t_tensor, x_tensor = build_window_from_patient_df(df_patient, meta, window_size=args.window_size, device=device)
    with torch.no_grad():
        logits = model(t_tensor, x_tensor)
        prob = torch.sigmoid(logits).item()

    # Build summary
    summary = build_patient_summary(
        df_patient,
        risk_score=prob,
        meta=meta,
        horizon_hours=args.horizon_hours,
        window_hours=args.window_size,
    )

    print("Sepsis risk (next {} hours): {:.4f}".format(args.horizon_hours, prob))
    print("Patient summary JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
