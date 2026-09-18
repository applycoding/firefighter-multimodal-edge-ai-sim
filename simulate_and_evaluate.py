#!/usr/bin/env python3
"""
SYNTHETIC validation study: firefighter helmet multimodal edge-AI safety monitor.

DISCLAIMER: All data are SYNTHETIC. No real fireground, department, or human-subject
data were used. Results demonstrate methodological feasibility only.

Sampling: 1 sample / 10 seconds over T=60 mission minutes (360 timesteps).
N=12 simulated firefighter sorties. Seed=42.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_UNITS = 12
T_MINUTES = 60
DT_SEC = 10  # 1 sample / 10 s
N_STEPS = (T_MINUTES * 60) // DT_SEC  # 360
TAU_ALERT = 0.35  # latent risk threshold for ground-truth "event active"
P_ALERT = 0.8  # proposed method: alert when P(risk > tau) >= 0.8
OUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# Latent risk names
RISKS = [
    "external_heat_risk",
    "toxicity_risk",
    "physiological_strain",
    "fall_or_mayday_risk",
    "localization_uncertainty",
]

# Observation channels
OBS_COLS = [
    "thermal_proxy",
    "ambient_temp",
    "gas_CO",
    "light_vis",
    "HR",
    "skin_temp",
    "IMU_mag",
    "GPS_quality",
    "env_noise",
]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _smooth_bump(t: np.ndarray, t0: float, width: float, amp: float) -> np.ndarray:
    """Raised-cosine bump peaking near t0."""
    x = (t - t0) / max(width, 1e-6)
    bump = np.clip(0.5 * (1 + np.cos(np.pi * np.clip(x, -1, 1))), 0, 1)
    bump[np.abs(x) > 1] = 0.0
    return amp * bump


def simulate_unit(unit_id: int, scenario: dict) -> tuple[pd.DataFrame, dict]:
    """Simulate one sortie. Returns (dataframe, event_meta)."""
    t = np.arange(N_STEPS) * DT_SEC / 60.0  # minutes
    meta = {"unit_id": unit_id, "events": {}}

    # Baseline latent processes (AR(1) noise)
    heat = np.zeros(N_STEPS)
    tox = np.zeros(N_STEPS)
    phys = np.zeros(N_STEPS)
    fall = np.zeros(N_STEPS)
    loc_unc = np.zeros(N_STEPS)

    # Mild background AR(1)
    for arr, sigma in [
        (heat, 0.04),
        (tox, 0.03),
        (phys, 0.05),
        (fall, 0.02),
        (loc_unc, 0.03),
    ]:
        eps = rng.normal(0, sigma, N_STEPS)
        for i in range(1, N_STEPS):
            arr[i] = 0.92 * arr[i - 1] + eps[i]
        arr[:] = np.clip(arr, 0, None)

    # Induce events per scenario flags
    if scenario.get("heat_ramp"):
        t0 = scenario["heat_t0"]
        bump = _smooth_bump(t, t0, width=8.0, amp=0.95)
        # Flashover-precursor: sharp rise in last third of window
        flash = _smooth_bump(t, t0 + 4.0, width=3.0, amp=0.45)
        heat += bump + flash
        meta["events"]["heat_ramp"] = {"t0_min": float(t0), "width_min": 8.0}

    if scenario.get("co_spike"):
        t0 = scenario["co_t0"]
        tox += _smooth_bump(t, t0, width=6.0, amp=0.95)
        meta["events"]["co_spike"] = {"t0_min": float(t0), "width_min": 6.0}

    if scenario.get("physio_overload"):
        t0 = scenario["phys_t0"]
        phys += _smooth_bump(t, t0, width=10.0, amp=0.92)
        meta["events"]["physio_overload"] = {"t0_min": float(t0), "width_min": 10.0}

    if scenario.get("fall"):
        t0 = scenario["fall_t0"]
        # Brief high-magnitude fall spike
        fall += _smooth_bump(t, t0, width=2.0, amp=1.0)
        # Residual elevated mayday risk briefly after
        fall += _smooth_bump(t, t0 + 1.5, width=3.0, amp=0.4)
        meta["events"]["fall"] = {"t0_min": float(t0), "width_min": 2.0}

    if scenario.get("gps_denied"):
        t0 = scenario["gps_t0"]
        dur = scenario.get("gps_dur", 20.0)
        mask = (t >= t0) & (t <= t0 + dur)
        # Localization uncertainty ramps with dead-reckoning drift
        drift = np.zeros(N_STEPS)
        cum = 0.0
        for i in range(N_STEPS):
            if mask[i]:
                cum += 0.015 + abs(rng.normal(0, 0.008))
                drift[i] = min(cum, 1.0)
            else:
                cum = max(0.0, cum * 0.85)
                drift[i] = cum
        loc_unc += drift
        meta["events"]["gps_denied"] = {"t0_min": float(t0), "dur_min": float(dur)}

    # Confound: high ambient noise / radio chatter (should NOT alone trigger hazards)
    noise_confound = np.zeros(N_STEPS)
    if scenario.get("noise_confound"):
        t0 = scenario["noise_t0"]
        noise_confound = _smooth_bump(t, t0, width=12.0, amp=1.0)
        meta["events"]["noise_confound"] = {"t0_min": float(t0), "width_min": 12.0}

    heat = np.clip(heat, 0, 1)
    tox = np.clip(tox, 0, 1)
    phys = np.clip(phys, 0, 1)
    fall = np.clip(fall, 0, 1)
    loc_unc = np.clip(loc_unc, 0, 1)

    # Primary composite risk (for overall alert GT): max of core hazards
    # (localization uncertainty alone is not a mayday-class alert).
    # Soft blend keeps differentiability for RMSE correlation metrics.
    core = np.maximum.reduce([heat, tox, phys, fall])
    primary = np.clip(0.85 * core + 0.15 * (
        0.30 * heat + 0.25 * tox + 0.25 * phys + 0.20 * fall
    ), 0, 1)

    # Observations (noisy, with missingness)
    # thermal_proxy ~ heat + ambient coupling
    thermal_proxy = 25 + 55 * heat + 8 * rng.normal(0, 1, N_STEPS)
    ambient_temp = 22 + 40 * heat + 5 * rng.normal(0, 1, N_STEPS)
    gas_CO = 5 + 180 * tox + 12 * rng.normal(0, 1, N_STEPS)  # ppm proxy
    gas_CO = np.clip(gas_CO, 0, None)
    light_vis = 800 - 600 * heat * 0.5 - 200 * tox * 0.3 + 50 * rng.normal(0, 1, N_STEPS)
    light_vis = np.clip(light_vis, 10, None)
    HR = 85 + 70 * phys + 15 * heat * 0.3 + 8 * rng.normal(0, 1, N_STEPS)
    skin_temp = 33 + 5 * phys + 2 * heat + 0.8 * rng.normal(0, 1, N_STEPS)
    IMU_mag = 0.15 + 4.5 * fall + 0.4 * phys * 0.2 + 0.25 * rng.normal(0, 1, N_STEPS)
    IMU_mag = np.clip(IMU_mag, 0, None)
    GPS_quality = 0.95 - 0.85 * loc_unc + 0.05 * rng.normal(0, 1, N_STEPS)
    GPS_quality = np.clip(GPS_quality, 0, 1)
    env_noise = 55 + 35 * noise_confound + 8 * rng.normal(0, 1, N_STEPS)

    # Missingness: random dropouts ~8%, plus GPS-denied worsens GPS
    miss_rate = 0.08
    obs = {
        "thermal_proxy": thermal_proxy,
        "ambient_temp": ambient_temp,
        "gas_CO": gas_CO,
        "light_vis": light_vis,
        "HR": HR,
        "skin_temp": skin_temp,
        "IMU_mag": IMU_mag,
        "GPS_quality": GPS_quality,
        "env_noise": env_noise,
    }
    for k in obs:
        mask = rng.random(N_STEPS) < miss_rate
        arr = obs[k].astype(float)
        arr[mask] = np.nan
        obs[k] = arr

    # Extra GPS missing during denied
    if scenario.get("gps_denied"):
        t0 = scenario["gps_t0"]
        dur = scenario.get("gps_dur", 20.0)
        mask = (t >= t0) & (t <= t0 + dur)
        gq = obs["GPS_quality"].copy()
        gq[mask & (rng.random(N_STEPS) < 0.5)] = np.nan
        obs["GPS_quality"] = gq

    df = pd.DataFrame(
        {
            "unit_id": unit_id,
            "t_min": t,
            "step": np.arange(N_STEPS),
            **obs,
            "latent_heat": heat,
            "latent_tox": tox,
            "latent_phys": phys,
            "latent_fall": fall,
            "latent_loc": loc_unc,
            "latent_primary": primary,
            "gt_alert": (primary >= TAU_ALERT).astype(int),
            "gt_heat": (heat >= TAU_ALERT).astype(int),
            "gt_tox": (tox >= TAU_ALERT).astype(int),
            "gt_phys": (phys >= TAU_ALERT).astype(int),
            "gt_fall": (fall >= TAU_ALERT).astype(int),
            "noise_confound_active": (noise_confound > 0.3).astype(int),
        }
    )
    return df, meta


def build_scenarios() -> list[dict]:
    """Assign induced events across N=12 units."""
    scenarios = []
    # Heat ramp for 4 units
    heat_units = {0, 1, 2, 3}
    # CO spike for 3 units
    co_units = {1, 4, 5}
    # Physio overload for 3 units
    phys_units = {2, 6, 7}
    # Fall for 2 units
    fall_units = {3, 8}
    # GPS-denied for half (6 units)
    gps_units = {0, 2, 4, 6, 8, 10}
    # Noise confound for several units (including some without true hazards)
    noise_units = {5, 9, 10, 11}

    for u in range(N_UNITS):
        sc = {"unit_id": u}
        if u in heat_units:
            sc["heat_ramp"] = True
            sc["heat_t0"] = float(rng.uniform(15, 40))
        if u in co_units:
            sc["co_spike"] = True
            sc["co_t0"] = float(rng.uniform(10, 45))
        if u in phys_units:
            sc["physio_overload"] = True
            sc["phys_t0"] = float(rng.uniform(12, 42))
        if u in fall_units:
            sc["fall"] = True
            sc["fall_t0"] = float(rng.uniform(20, 50))
        if u in gps_units:
            sc["gps_denied"] = True
            sc["gps_t0"] = float(rng.uniform(8, 30))
            sc["gps_dur"] = float(rng.uniform(15, 25))
        if u in noise_units:
            sc["noise_confound"] = True
            sc["noise_t0"] = float(rng.uniform(5, 35))
        scenarios.append(sc)
    return scenarios


def simulate_all() -> tuple[pd.DataFrame, list[dict]]:
    scenarios = build_scenarios()
    frames = []
    metas = []
    for sc in scenarios:
        df, meta = simulate_unit(sc["unit_id"], sc)
        frames.append(df)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), metas


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def _nan_zscore(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    z = (x - mu) / max(sd, 1e-6)
    z = np.where(np.isnan(x), 0.0, z)  # missing -> no evidence
    return z


def baseline_a_thresholds(df: pd.DataFrame) -> np.ndarray:
    """Independent channel absolute/z thresholds; OR-union alert."""
    # Fit norms on pooled data (synthetic "training" priors from same pool;
    # honest: we use robust percentiles as if calibrated offline)
    scores = np.zeros(len(df))
    # Channel-specific absolute thresholds (synthetic SOP-like)
    rules = []
    # thermal_proxy high
    rules.append(np.nan_to_num(df["thermal_proxy"].values, nan=0) > 55)
    rules.append(np.nan_to_num(df["ambient_temp"].values, nan=0) > 50)
    rules.append(np.nan_to_num(df["gas_CO"].values, nan=0) > 80)
    rules.append(np.nan_to_num(df["HR"].values, nan=0) > 130)
    rules.append(np.nan_to_num(df["skin_temp"].values, nan=0) > 37.5)
    rules.append(np.nan_to_num(df["IMU_mag"].values, nan=0) > 2.0)
    # env_noise alone should NOT fire - intentionally omit
    alert = np.any(np.column_stack(rules), axis=1).astype(float)
    # Soft score = fraction of channels firing
    soft = np.mean(np.column_stack(rules).astype(float), axis=1)
    return alert, soft


def baseline_b_ewma(df: pd.DataFrame, lam: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Univariate EWMA per hazard-relevant channel; union rule."""
    channels = ["thermal_proxy", "gas_CO", "HR", "skin_temp", "IMU_mag"]
    # Reference means/sds from early quiet period proxy: global robust stats
    ewma_alerts = []
    ewma_scores = []
    for ch in channels:
        vals = df[ch].values.astype(float)
        # Impute missing with forward-fill within unit then median
        for uid in df["unit_id"].unique():
            m = df["unit_id"].values == uid
            s = pd.Series(vals[m]).ffill().bfill()
            vals[m] = s.values
        mu = np.nanmedian(vals)
        sd = np.nanstd(vals) + 1e-6
        z = (vals - mu) / sd
        # EWMA of z per unit
        ew = np.zeros_like(z)
        for uid in df["unit_id"].unique():
            m = np.where(df["unit_id"].values == uid)[0]
            prev = 0.0
            for i, idx in enumerate(m):
                prev = lam * z[idx] + (1 - lam) * prev
                ew[idx] = prev
        ewma_alerts.append(ew > 2.2)
        ewma_scores.append(ew)
    alert = np.any(np.column_stack(ewma_alerts), axis=1).astype(float)  # z>2.5 union
    soft = np.max(np.column_stack(ewma_scores), axis=1)
    # Map soft to [0,1]-ish via logistic
    soft_p = 1 / (1 + np.exp(-(soft - 2.0)))
    return alert, soft_p


def _kalman_1d_filter(
    y: np.ndarray,
    H: float,
    R: float,
    Q: float,
    x0: float = 0.0,
    P0: float = 1.0,
    A: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar Kalman filter. y may contain NaN (skip update). Returns x_filt, P_filt."""
    n = len(y)
    x = np.zeros(n)
    P = np.zeros(n)
    x_prev, P_prev = x0, P0
    for i in range(n):
        # Predict
        x_pred = A * x_prev
        P_pred = A * P_prev * A + Q
        yi = y[i]
        if np.isnan(yi):
            x[i], P[i] = x_pred, P_pred
        else:
            S = H * P_pred * H + R
            K = P_pred * H / S
            x[i] = x_pred + K * (yi - H * x_pred)
            P[i] = (1 - K * H) * P_pred
        x_prev, P_prev = x[i], P[i]
    return x, P


def _rts_smooth(
    x_f: np.ndarray, P_f: np.ndarray, A: float = 0.95, Q: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother (offline edge post-pass)."""
    n = len(x_f)
    x_s = x_f.copy()
    P_s = P_f.copy()
    for i in range(n - 2, -1, -1):
        P_pred = A * P_f[i] * A + Q
        C = P_f[i] * A / max(P_pred, 1e-12)
        x_s[i] = x_f[i] + C * (x_s[i + 1] - A * x_f[i])
        P_s[i] = P_f[i] + C * (P_s[i + 1] - P_pred) * C
    return x_s, P_s


def proposed_kalman_multimodal(
    df: pd.DataFrame,
    modalities: str = "all",
    use_shrinkage: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multimodal edge state-space / Kalman latent risk tracker with hierarchical
    shrinkage across units (empirical Bayes on process noise / prior mean).

    modalities: 'all' | 'thermal_gas' | 'physio' | 'imu'
    Returns: alert (0/1), prob (P(risk>tau)), risk_est
    """
    n = len(df)
    risk_est = np.zeros(n)
    risk_var = np.zeros(n)
    alert = np.zeros(n)
    prob = np.zeros(n)

    # Observation maps -> primary risk (synthetic calibrated linear loadings)
    # Build a fused observation of primary risk from available channels
    def fuse_obs(sub: pd.DataFrame) -> np.ndarray:
        parts = []
        weights = []
        if modalities in ("all", "thermal_gas"):
            # Normalize channels to approx [0,1] risk scale
            th = (sub["thermal_proxy"].values - 25) / 55
            co = (sub["gas_CO"].values - 5) / 180
            at = (sub["ambient_temp"].values - 22) / 40
            parts.extend([th, co, at])
            weights.extend([0.35, 0.35, 0.15])
        if modalities in ("all", "physio"):
            hr = (sub["HR"].values - 85) / 70
            sk = (sub["skin_temp"].values - 33) / 5
            parts.extend([hr, sk])
            weights.extend([0.35, 0.30] if modalities == "physio" else [0.25, 0.20])
        if modalities in ("all", "imu"):
            im = (sub["IMU_mag"].values - 0.15) / 4.5
            gq = 1.0 - sub["GPS_quality"].values  # unc proxy
            if modalities == "imu":
                parts.extend([im, gq])
                weights.extend([0.7, 0.3])
            elif modalities == "all":
                parts.extend([im])
                weights.extend([0.15])
        # env_noise intentionally excluded from risk fusion (confound)
        W = np.array(weights, dtype=float)
        W = W / W.sum()
        stacked = np.column_stack(parts)  # may contain nan
        # Softmax-weighted pool emphasizing the strongest channel evidence
        # (edge monitor should react to the dominant hazard cue).
        clipped = np.clip(stacked, 0.0, 1.25)
        # Replace nan with -inf for max, 0 weight for mean
        max_ev = np.nanmax(clipped, axis=1)
        num = np.zeros(len(sub))
        den = np.zeros(len(sub))
        for j, w in enumerate(W):
            col = clipped[:, j]
            valid = ~np.isnan(col)
            num[valid] += w * col[valid]
            den[valid] += w
        mean_ev = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        # Blend mean (stable) and max (peak-sensitive)
        y = np.where(
            np.isnan(mean_ev) & np.isnan(max_ev),
            np.nan,
            np.nan_to_num(0.45 * mean_ev, nan=0.0) + np.nan_to_num(0.55 * max_ev, nan=0.0),
        )
        # If both nan at a row, keep nan
        both_nan = np.isnan(stacked).all(axis=1)
        y = np.where(both_nan, np.nan, y)
        return y

    # Empirical Bayes: pool unit-level mean of fused obs for hierarchical prior
    unit_means = []
    unit_ids = sorted(df["unit_id"].unique())
    fused_by_unit = {}
    for uid in unit_ids:
        sub = df[df["unit_id"] == uid]
        y = fuse_obs(sub)
        fused_by_unit[uid] = y
        unit_means.append(np.nanmean(y))
    unit_means = np.array(unit_means)
    global_mu = float(np.nanmean(unit_means))
    between_var = float(np.nanvar(unit_means)) + 1e-6
    within_var = float(
        np.nanmean([np.nanvar(fused_by_unit[uid]) for uid in unit_ids])
    ) + 1e-6

    # Shrinkage factor for unit prior mean toward global
    # kappa = within / (within + n_eff * between) style EB
    for uid in unit_ids:
        m = df["unit_id"].values == uid
        y = fused_by_unit[uid]
        n_obs = np.sum(~np.isnan(y))
        shrink = within_var / (within_var + max(n_obs, 1) * between_var)
        shrink = float(np.clip(shrink, 0.05, 0.6))
        if use_shrinkage:
            x0 = shrink * global_mu + (1 - shrink) * float(np.nanmean(y))
        else:
            x0 = float(np.nanmean(y)) if n_obs else global_mu
        x0 = float(np.clip(x0, 0, 1))

        # Process / observation noise (EB-ish from residual scale)
        # Slightly larger Q for responsive edge tracking of flashover/CO onsets
        Q = 0.03 + 0.01 * shrink
        R = 0.04 + 0.06 * (modalities != "all")
        A = 0.94
        H = 1.0
        x_f, P_f = _kalman_1d_filter(y, H=H, R=R, Q=Q, x0=x0, P0=0.25, A=A)
        x_s, P_s = _rts_smooth(x_f, P_f, A=A, Q=Q)
        x_f = np.clip(x_f, 0, 1)
        x_s = np.clip(x_s, 0, 1)
        P_f = np.clip(P_f, 1e-6, None)
        P_s = np.clip(P_s, 1e-6, None)

        # Offline estimate quality uses RTS; online alerts use causal filter
        risk_est[m] = x_s
        risk_var[m] = P_s
        tau = TAU_ALERT
        # Causal P(risk > tau); inflate scale floor so extreme peaks can reach P>=0.8
        p = 1.0 - stats.norm.cdf(tau, loc=x_f, scale=np.sqrt(P_f))
        # Also raise alert if point estimate itself is clearly above tau
        # (equivalent to P>0.5 under symmetric noise) AND causal P>=P_ALERT
        # Stick to specified rule: alert iff P >= P_ALERT
        prob[m] = p
        alert[m] = (p >= P_ALERT).astype(float)

    return alert, prob, risk_est


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def detection_delay(
    gt: np.ndarray, pred: np.ndarray, unit_ids: np.ndarray, t_min: np.ndarray
) -> list[float]:
    """Per-event onset delay (minutes) when pred fires after gt rises."""
    delays = []
    for uid in np.unique(unit_ids):
        m = unit_ids == uid
        g = gt[m]
        p = pred[m]
        tt = t_min[m]
        # Find rising edges of GT
        onsets = np.where((g[1:] == 1) & (g[:-1] == 0))[0] + 1
        if g[0] == 1:
            onsets = np.concatenate([[0], onsets])
        for onset in onsets:
            # Search forward for first pred==1
            window = p[onset:]
            hits = np.where(window >= 0.5)[0]
            if len(hits):
                delays.append(float(tt[onset + hits[0]] - tt[onset]))
            else:
                delays.append(np.nan)  # missed
    return delays


def false_alert_rate(gt: np.ndarray, pred: np.ndarray, unit_ids: np.ndarray) -> float:
    """Mean fraction of non-event time with alert, averaged over units."""
    rates = []
    for uid in np.unique(unit_ids):
        m = unit_ids == uid
        g = gt[m]
        p = pred[m]
        neg = g == 0
        if neg.sum() == 0:
            continue
        rates.append(float((p[neg] >= 0.5).mean()))
    return float(np.mean(rates)) if rates else 0.0


def event_specific_delays(
    df: pd.DataFrame, pred: np.ndarray, metas: list[dict]
) -> dict:
    """Delay from induced event t0 to first alert for that unit."""
    out = {k: [] for k in ["heat_ramp", "co_spike", "physio_overload", "fall"]}
    pred = np.asarray(pred)
    for meta in metas:
        uid = meta["unit_id"]
        m = df["unit_id"].values == uid
        tt = df.loc[m, "t_min"].values
        p = pred[m]
        for ev_name, info in meta["events"].items():
            if ev_name not in out:
                continue
            t0 = info["t0_min"]
            # First alert at or after t0 within 15 min
            after = tt >= t0
            hits = np.where(after & (p >= 0.5))[0]
            if len(hits):
                out[ev_name].append(float(tt[hits[0]] - t0))
            else:
                out[ev_name].append(np.nan)
    summary = {}
    for k, vals in out.items():
        arr = np.array(vals, dtype=float)
        summary[k] = {
            "n": int(len(arr)),
            "mean_delay_min": float(np.nanmean(arr)) if len(arr) else None,
            "median_delay_min": float(np.nanmedian(arr)) if len(arr) else None,
            "miss_rate": float(np.mean(np.isnan(arr))) if len(arr) else None,
        }
    return summary


def evaluate_method(
    name: str,
    df: pd.DataFrame,
    alert: np.ndarray,
    score: np.ndarray,
    risk_est: np.ndarray | None,
    metas: list[dict],
) -> dict:
    y = df["gt_alert"].values.astype(int)
    pred = (np.asarray(alert) >= 0.5).astype(int)
    sc = np.asarray(score, dtype=float)
    sc = np.nan_to_num(sc, nan=0.0)

    metrics = {
        "method": name,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, sc)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, sc)) if y.sum() else float("nan"),
        "false_alert_rate": false_alert_rate(y, pred, df["unit_id"].values),
    }
    delays = detection_delay(y, pred, df["unit_id"].values, df["t_min"].values)
    darr = np.array(delays, dtype=float)
    metrics["mean_detection_delay_min"] = float(np.nanmean(darr)) if len(darr) else None
    metrics["median_detection_delay_min"] = (
        float(np.nanmedian(darr)) if len(darr) else None
    )
    metrics["event_miss_rate"] = float(np.mean(np.isnan(darr))) if len(darr) else None
    metrics["event_specific_delays"] = event_specific_delays(df, pred, metas)

    if risk_est is not None:
        true_r = df["latent_primary"].values
        est = np.asarray(risk_est)
        metrics["risk_pearson_r"] = float(np.corrcoef(true_r, est)[0, 1])
        metrics["risk_rmse"] = float(np.sqrt(np.mean((true_r - est) ** 2)))
    else:
        metrics["risk_pearson_r"] = None
        metrics["risk_rmse"] = None

    # Confound check: FAR during noise-only periods (noise active, gt_alert=0)
    noise_m = (df["noise_confound_active"].values == 1) & (y == 0)
    if noise_m.sum() > 0:
        metrics["far_during_noise_confound"] = float(pred[noise_m].mean())
    else:
        metrics["far_during_noise_confound"] = None

    return metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_latent_vs_estimates(df: pd.DataFrame, risk_est: np.ndarray):
    """Example units: true primary latent vs Kalman estimate."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    # Pick units with interesting events
    show = [0, 1, 3, 8]
    for ax, uid in zip(axes.ravel(), show):
        m = df["unit_id"].values == uid
        tt = df.loc[m, "t_min"].values
        ax.plot(tt, df.loc[m, "latent_primary"].values, "k-", lw=2, label="True primary risk")
        ax.plot(tt, risk_est[m], "C0-", lw=1.5, alpha=0.9, label="Kalman estimate")
        ax.axhline(TAU_ALERT, color="C3", ls="--", lw=1, label=f"tau={TAU_ALERT}")
        ax.set_ylabel("Risk")
        ax.set_title(f"Unit {uid} (synthetic)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].set_xlabel("Mission time (min)")
    axes[1, 1].set_xlabel("Mission time (min)")
    fig.suptitle(
        "SYNTHETIC: Latent primary risk vs multimodal Kalman estimates",
        fontsize=12,
    )
    fig.tight_layout()
    path = FIG_DIR / "fig1_latent_vs_estimates.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig2_roc_comparison(df: pd.DataFrame, scores: dict):
    y = df["gt_alert"].values.astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, sc in scores.items():
        sc = np.nan_to_num(np.asarray(sc), nan=0.0)
        fpr, tpr, _ = roc_curve(y, sc)
        auc = roc_auc_score(y, sc)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("SYNTHETIC: ROC comparison of alert scores")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig2_roc_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig3_delay_falsealarms(metrics_list: list[dict]):
    names = [m["method"] for m in metrics_list]
    delays = [
        m["mean_detection_delay_min"] if m["mean_detection_delay_min"] is not None else 0
        for m in metrics_list
    ]
    fars = [m["false_alert_rate"] for m in metrics_list]
    f1s = [m["f1"] for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    axes[0].barh(names, delays, color=colors)
    axes[0].set_xlabel("Mean detection delay (min)")
    axes[0].set_title("Detection delay")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(names, fars, color=colors)
    axes[1].set_xlabel("False-alert rate (non-event time)")
    axes[1].set_title("False-alert rate")
    axes[1].grid(True, axis="x", alpha=0.3)

    axes[2].barh(names, f1s, color=colors)
    axes[2].set_xlabel("F1")
    axes[2].set_title("F1 score")
    axes[2].set_xlim(0, 1)
    axes[2].grid(True, axis="x", alpha=0.3)

    fig.suptitle("SYNTHETIC: Delay, false alarms, and F1 by method", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "fig3_delay_falsealarms.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig4_modality_coverage(df: pd.DataFrame):
    """Missingness / coverage per modality."""
    cov = {}
    for c in OBS_COLS:
        cov[c] = float(1.0 - df[c].isna().mean())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(cov.keys())
    vals = [cov[k] for k in names]
    ax.bar(names, vals, color="steelblue", edgecolor="k", lw=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Non-missing fraction")
    ax.set_title("SYNTHETIC: Observation modality coverage (missingness)")
    ax.axhline(0.92, color="C3", ls="--", lw=1, label="Target ~0.92 (8% dropout)")
    plt.xticks(rotation=35, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig4_modality_coverage.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def write_summary(metrics_list: list[dict], metas: list[dict], paths: dict):
    proposed = next(m for m in metrics_list if m["method"] == "Proposed_Kalman")
    base_a = next(m for m in metrics_list if m["method"] == "BaselineA_Thresholds")
    base_b = next(m for m in metrics_list if m["method"] == "BaselineB_EWMA")

    beats_f1 = proposed["f1"] >= max(base_a["f1"], base_b["f1"])
    beats_auc = proposed["auroc"] >= max(base_a["auroc"], base_b["auroc"])
    lower_far = proposed["false_alert_rate"] <= min(
        base_a["false_alert_rate"], base_b["false_alert_rate"]
    )

    lines = []
    lines.append("# SYNTHETIC Validation Study: Firefighter Helmet Multimodal Edge-AI")
    lines.append("")
    lines.append("## Disclaimer")
    lines.append("")
    lines.append(
        "All data in this study are SYNTHETIC. No real fireground recordings, "
        "department logs, or human physiological datasets were used. Results "
        "illustrate methodological behavior under controlled induced events only "
        "and must not be interpreted as operational performance claims."
    )
    lines.append("")
    lines.append("## Design")
    lines.append("")
    lines.append(f"- N = {N_UNITS} simulated firefighter sorties (crew-members)")
    lines.append(
        f"- T = {T_MINUTES} mission minutes at 1 sample / {DT_SEC} s "
        f"({N_STEPS} timesteps per unit)"
    )
    lines.append(f"- Random seed = {SEED}")
    lines.append(
        "- Latent hazards: external_heat_risk, toxicity_risk, physiological_strain, "
        "fall_or_mayday_risk, localization_uncertainty"
    )
    lines.append(
        "- Induced events: heat ramp/flashover-precursor (4 units), CO spike (3), "
        "physio overload (3), fall (2), GPS-denied indoor DR drift (6), "
        "ambient noise/radio confound (4; should not alone trigger alerts)"
    )
    lines.append(
        f"- Ground-truth alert when latent primary risk >= {TAU_ALERT}; "
        f"proposed alerts when P(risk > tau) >= {P_ALERT}"
    )
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("1. Baseline A: Independent channel absolute thresholds (OR-union)")
    lines.append("2. Baseline B: Univariate EWMA union rule (lambda=0.2, z>2.2)")
    lines.append(
        "3. Proposed: Multimodal Kalman filter + RTS smoother with empirical-Bayes "
        "hierarchical shrinkage of unit prior means; env_noise excluded from fusion"
    )
    lines.append(
        "4. Ablations: thermal+gas only; physio-only; IMU(+GPS unc)-only"
    )
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(
        "| Method | F1 | AUROC | Precision | Recall | Mean Delay (min) | FAR |"
    )
    lines.append(
        "|--------|----|-------|-----------|--------|------------------|-----|"
    )
    for m in metrics_list:
        delay = m["mean_detection_delay_min"]
        delay_s = f"{delay:.2f}" if delay is not None else "NA"
        lines.append(
            f"| {m['method']} | {m['f1']:.3f} | {m['auroc']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {delay_s} | "
            f"{m['false_alert_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Honest Outcome Check")
    lines.append("")
    lines.append(
        f"- Proposed beats best baseline on F1: {beats_f1} "
        f"(Proposed F1={proposed['f1']:.3f} vs A={base_a['f1']:.3f}, B={base_b['f1']:.3f})"
    )
    lines.append(
        f"- Proposed beats best baseline on AUROC: {beats_auc} "
        f"(Proposed AUROC={proposed['auroc']:.3f} vs A={base_a['auroc']:.3f}, "
        f"B={base_b['auroc']:.3f})"
    )
    lines.append(
        f"- Proposed has lower or equal FAR vs best baseline: {lower_far} "
        f"(Proposed FAR={proposed['false_alert_rate']:.3f})"
    )
    if proposed.get("risk_pearson_r") is not None:
        lines.append(
            f"- Risk estimate vs true primary: Pearson r={proposed['risk_pearson_r']:.3f}, "
            f"RMSE={proposed['risk_rmse']:.3f}"
        )
    if proposed.get("far_during_noise_confound") is not None:
        lines.append(
            f"- FAR during noise-confound / non-event periods: "
            f"{proposed['far_during_noise_confound']:.3f} "
            "(env_noise not used in proposed fusion)"
        )
    lines.append("")
    lines.append("## Event-Specific Delays (Proposed)")
    lines.append("")
    for ev, info in proposed["event_specific_delays"].items():
        md = info["mean_delay_min"]
        md_s = f"{md:.2f}" if md is not None else "NA"
        lines.append(
            f"- {ev}: n={info['n']}, mean delay={md_s} min, "
            f"miss_rate={info['miss_rate']}"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for k, p in paths.items():
        lines.append(f"- {k}: {p}")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "Run: `python simulate_and_evaluate.py` from this directory with packages "
        "in requirements.txt. Prefer Kalman/RTS + empirical Bayes (no MCMC)."
    )
    lines.append("")

    text = "\n".join(lines)
    # Enforce ASCII hyphen only (no em dashes / en dashes)
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    path = RESULTS_DIR / "summary.md"
    path.write_text(text, encoding="ascii", errors="replace")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SYNTHETIC firefighter helmet multimodal edge-AI validation study")
    print("NO real fireground or department data.")
    print("=" * 70)

    df, metas = simulate_all()
    print(f"Simulated N={N_UNITS} units x {N_STEPS} steps "
          f"(dt={DT_SEC}s, T={T_MINUTES} min). Rows={len(df)}")
    print(f"GT alert prevalence: {df['gt_alert'].mean():.3f}")

    # Baseline A
    alert_a, score_a = baseline_a_thresholds(df)
    # Baseline B
    alert_b, score_b = baseline_b_ewma(df)
    # Proposed
    alert_p, score_p, risk_p = proposed_kalman_multimodal(df, modalities="all")
    # Ablations
    alert_tg, score_tg, risk_tg = proposed_kalman_multimodal(df, modalities="thermal_gas")
    alert_ph, score_ph, risk_ph = proposed_kalman_multimodal(df, modalities="physio")
    alert_im, score_im, risk_im = proposed_kalman_multimodal(df, modalities="imu")

    methods = [
        ("BaselineA_Thresholds", alert_a, score_a, None),
        ("BaselineB_EWMA", alert_b, score_b, None),
        ("Proposed_Kalman", alert_p, score_p, risk_p),
        ("Ablation_ThermalGas", alert_tg, score_tg, risk_tg),
        ("Ablation_Physio", alert_ph, score_ph, risk_ph),
        ("Ablation_IMU", alert_im, score_im, risk_im),
    ]

    metrics_list = []
    for name, alert, score, risk in methods:
        m = evaluate_method(name, df, alert, score, risk, metas)
        metrics_list.append(m)
        print(
            f"  {name}: F1={m['f1']:.3f} AUROC={m['auroc']:.3f} "
            f"FAR={m['false_alert_rate']:.3f} delay={m['mean_detection_delay_min']}"
        )

    # Figures
    paths = {}
    paths["fig1"] = str(fig1_latent_vs_estimates(df, risk_p))
    scores_for_roc = {
        "BaselineA": score_a,
        "BaselineB_EWMA": score_b,
        "Proposed_Kalman": score_p,
        "Ablation_ThermalGas": score_tg,
        "Ablation_Physio": score_ph,
        "Ablation_IMU": score_im,
    }
    paths["fig2"] = str(fig2_roc_comparison(df, scores_for_roc))
    paths["fig3"] = str(fig3_delay_falsealarms(metrics_list))
    paths["fig4"] = str(fig4_modality_coverage(df))

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "metrics.json"
    payload = {
        "disclaimer": (
            "SYNTHETIC data only. No real fireground or department data. "
            "Methodological feasibility demonstration."
        ),
        "config": {
            "seed": SEED,
            "n_units": N_UNITS,
            "t_minutes": T_MINUTES,
            "dt_sec": DT_SEC,
            "n_steps": N_STEPS,
            "tau_alert": TAU_ALERT,
            "p_alert": P_ALERT,
            "sampling": f"1 sample / {DT_SEC} seconds",
        },
        "metrics": metrics_list,
        "scenario_metas": metas,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    paths["metrics_json"] = str(metrics_path)

    summary_path = write_summary(metrics_list, metas, paths)
    paths["summary_md"] = str(summary_path)

    print("\nWrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print("\nDone.")
    return metrics_list, paths


if __name__ == "__main__":
    main()
