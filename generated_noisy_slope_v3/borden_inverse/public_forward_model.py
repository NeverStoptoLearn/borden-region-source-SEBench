"""Public Borden finite-region ADE forward model.

This file is intentionally agent-visible.  It mirrors the judge's ADE operator
for the finite-duration rectangular-region source: a rectangular source is
represented by a deterministic tensor product of sub-source points and release
offsets, and each term is evaluated with a 3D point-source ADE kernel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG = "public_problem_config.json"
DEFAULT_ANSWER = "answer.json"

HYDRO_DEFAULTS = {
    "porosity": 0.3,
    "hk_m_per_day": 10.0,
    "head_upstream_m": 220.0,
    "head_downstream_m": 219.0,
    "alpha_L_m": 10.0,
    "alpha_TH_m": 0.5,
    "alpha_TV_m": 0.01,
    "Dm_m2_per_day": 0.0,
    "lambda_per_day": 0.0,
    "retardation_factor": 1.0,
    "q_source_m3_per_day": 0.06764533176746916,
    "fallback_source_scale_factor": 1000.0,
}

SOURCE_KEYS = [
    "x_center",
    "y_center",
    "z_center",
    "half_length_x",
    "half_length_y",
    "half_length_z",
    "C0",
    "t_start",
    "duration",
]


def _read_json(value: str | Path | dict[str, Any] | None, default_path: str) -> dict[str, Any]:
    if value is None:
        return json.loads(Path(default_path).read_text(encoding="utf-8"))
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return dict(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def estimate_uniform_velocity(
    hk: float = 10.0,
    porosity: float = 0.3,
    head_upstream: float = 220.0,
    head_downstream: float = 219.0,
    Lx: float = 1200.0,
) -> float:
    hydraulic_gradient = (head_upstream - head_downstream) / Lx
    return hk * hydraulic_gradient / porosity


def normalize_region_answer(source_params: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    p = dict(source_params)
    bounds = config.get("source_search_bounds_for_agent", {})
    if "x_center" not in p and "x0" in p:
        p["x_center"] = p.get("x0")
    if "y_center" not in p and "y0" in p:
        p["y_center"] = p.get("y0")
    if "z_center" not in p and "z0" in p:
        p["z_center"] = p.get("z0")
    p.setdefault("source_type", "rectangular_region")
    p.setdefault("dimension", 3)
    p.setdefault("half_length_x", bounds.get("half_length_x_min", 5.0))
    p.setdefault("half_length_y", bounds.get("half_length_y_min", 5.0))
    p.setdefault("half_length_z", bounds.get("half_length_z_min", 0.5))
    p.setdefault("C0", 100.0)
    p.setdefault("t_start", bounds.get("t_start_min", 0.0))
    p.setdefault("duration", bounds.get("duration_min", 100.0))
    out: dict[str, Any] = {
        "source_type": str(p.get("source_type", "rectangular_region")),
        "dimension": int(p.get("dimension", 3)),
    }
    for name in SOURCE_KEYS:
        out[name] = _safe_float(p.get(name), 0.0)
    return out


def _axis_points(center: float, half_length: float, n: int) -> np.ndarray:
    center = float(center)
    half_length = float(half_length)
    n = int(n)
    if n <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - half_length, center + half_length, n)


def region_subpoints(source: dict[str, Any], config: dict[str, Any]) -> list[tuple[float, float, float]]:
    disc = config.get("region_source_discretization", {})
    nx = int(disc.get("nx", 5))
    ny = int(disc.get("ny", 5))
    nz = int(disc.get("nz", 3))
    xs = _axis_points(source["x_center"], source["half_length_x"], nx)
    ys = _axis_points(source["y_center"], source["half_length_y"], ny)
    zs = _axis_points(source["z_center"], source["half_length_z"], nz)
    return [(float(x), float(y), float(z)) for x in xs for y in ys for z in zs]


def release_offsets(duration: float, config: dict[str, Any]) -> np.ndarray:
    n_steps = int(config.get("region_source_discretization", {}).get("release_steps", 9))
    duration = max(float(duration), 1.0)
    if n_steps <= 1:
        return np.array([0.5 * duration], dtype=float)
    return np.linspace(0.0, duration, n_steps)


def point3_response(
    c0: float,
    x: Any,
    y: Any,
    z: Any,
    t: Any,
    v: float,
    porosity: float,
    alpha_l: float,
    alpha_th: float,
    alpha_tv: float,
    q_source: float,
    source_x: float,
    source_y: float,
    source_z: float,
    Dm: float = 0.0,
    lamb: float = 0.0,
    R: float = 1.0,
    source_scale_factor: float = 1000.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.maximum(np.asarray(t, dtype=float) / max(R, 1e-12), 1e-6)
    dl = max(alpha_l * abs(v) + Dm, 1e-8)
    dth = max(alpha_th * abs(v) + Dm, 1e-8)
    dtv = max(alpha_tv * abs(v) + Dm, 1e-8)
    dx = x - source_x - v * t
    dy = y - source_y
    dz = z - source_z
    expo = -((dx * dx) / (4.0 * dl * t) + (dy * dy) / (4.0 * dth * t) + (dz * dz) / (4.0 * dtv * t))
    denom = (4.0 * math.pi * t) ** 1.5 * math.sqrt(dl * dth * dtv)
    val = c0 * q_source * source_scale_factor / np.maximum(denom, 1e-30)
    val = val * np.exp(expo) * np.exp(-lamb * t)
    return np.maximum(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _point_response(c0: float, x: Any, y: Any, z: Any, t_arr: Any, x0: float, y0: float, z0: float, hydro: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    porosity = float(hydro.get("porosity", HYDRO_DEFAULTS["porosity"]))
    v = float(
        hydro.get(
            "velocity_m_per_day",
            estimate_uniform_velocity(
                hk=float(hydro.get("hk_m_per_day", HYDRO_DEFAULTS["hk_m_per_day"])),
                porosity=porosity,
                head_upstream=float(hydro.get("head_upstream_m", HYDRO_DEFAULTS["head_upstream_m"])),
                head_downstream=float(hydro.get("head_downstream_m", HYDRO_DEFAULTS["head_downstream_m"])),
                Lx=float(config.get("grid", {}).get("domain_length_x_m", 1200.0)),
            ),
        )
    )
    return point3_response(
        c0,
        x,
        y,
        z,
        t_arr,
        v,
        porosity,
        float(hydro.get("alpha_L_m", HYDRO_DEFAULTS["alpha_L_m"])),
        float(hydro.get("alpha_TH_m", HYDRO_DEFAULTS["alpha_TH_m"])),
        float(hydro.get("alpha_TV_m", HYDRO_DEFAULTS["alpha_TV_m"])),
        float(hydro.get("q_source_m3_per_day", HYDRO_DEFAULTS["q_source_m3_per_day"])),
        x0,
        y0,
        z0,
        Dm=float(hydro.get("Dm_m2_per_day", HYDRO_DEFAULTS["Dm_m2_per_day"])),
        lamb=float(hydro.get("lambda_per_day", HYDRO_DEFAULTS["lambda_per_day"])),
        R=float(hydro.get("retardation_factor", HYDRO_DEFAULTS["retardation_factor"])),
        source_scale_factor=float(hydro.get("fallback_source_scale_factor", HYDRO_DEFAULTS["fallback_source_scale_factor"])),
    )


def _records_from_inputs(wells: Any | None, times: Any | None) -> tuple[pd.DataFrame, np.ndarray, bool, tuple[int, int] | None]:
    if wells is None:
        df = pd.read_csv("public_monitoring_data.csv")
    elif isinstance(wells, (str, Path)):
        df = pd.read_csv(wells)
    elif isinstance(wells, pd.DataFrame):
        df = wells.copy()
    else:
        df = pd.DataFrame(wells)
    if times is None:
        if "time_days" not in df.columns:
            raise ValueError("times must be supplied when wells has no time_days column")
        return df, df["time_days"].to_numpy(float), False, None
    t = np.asarray(times, dtype=float).reshape(-1)
    records = []
    for _, w in df.iterrows():
        for tt in t:
            rec = dict(w)
            rec["time_days"] = float(tt)
            rec["time_years"] = float(tt) / 365.0
            records.append(rec)
    return pd.DataFrame.from_records(records), np.tile(t, len(df)), True, (len(df), len(t))


def simulate_from_answer(
    answer: dict[str, Any],
    wells: Any,
    times_days: Any,
    public_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    config = public_config if public_config is not None else _read_json(None, DEFAULT_CONFIG)
    source = normalize_region_answer(answer, config)
    hydro = config.get("hydrogeological_parameters", {})
    subpoints = region_subpoints(source, config)
    offsets = release_offsets(source["duration"], config)
    times_arr = np.asarray(list(times_days), dtype=float)
    if isinstance(wells, pd.DataFrame):
        wells_df = wells.copy()
    else:
        wells_df = pd.DataFrame(wells)

    t_start = float(source["t_start"])
    c0 = float(source["C0"])
    n_norm = max(len(subpoints) * len(offsets), 1)
    records: list[dict[str, Any]] = []
    for _, w in wells_df.iterrows():
        x = float(w["x"])
        y = float(w["y"])
        z = float(w["z"])
        conc_total = np.zeros_like(times_arr, dtype=float)
        for sx, sy, sz in subpoints:
            for off in offsets:
                tau = t_start + float(off)
                eff_t = times_arr - tau
                active = eff_t > 0.0
                if not np.any(active):
                    continue
                c = np.zeros_like(times_arr, dtype=float)
                c[active] = _point_response(c0 / n_norm, x, y, z, eff_t[active], sx, sy, sz, hydro, config)
                conc_total += c
        conc_total = np.maximum(np.nan_to_num(conc_total, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        for tt, cc in zip(times_arr, conc_total):
            records.append(
                {
                    "well_id": str(w.get("well_id", "well")),
                    "x": x,
                    "y": y,
                    "z": z,
                    "time_days": float(tt),
                    "time_years": float(tt) / 365.0,
                    "concentration_predicted_mg_L": float(cc),
                }
            )
    return pd.DataFrame.from_records(records)


def predict_concentrations(
    answer: str | Path | dict[str, Any] | None = None,
    wells: Any | None = None,
    times: Any | None = None,
    config: str | Path | dict[str, Any] | None = None,
    **answer_fields: Any,
) -> np.ndarray:
    if answer is None and any(k in answer_fields for k in SOURCE_KEYS):
        ans = {k: v for k, v in answer_fields.items() if k in SOURCE_KEYS}
    else:
        ans = _read_json(answer, DEFAULT_ANSWER)
        for k, v in answer_fields.items():
            if k in SOURCE_KEYS:
                ans[k] = v
    cfg = _read_json(config, DEFAULT_CONFIG)
    records, _t, matrix, shape = _records_from_inputs(wells, times)
    if matrix:
        wells_df = records.drop_duplicates(subset=["well_id", "x", "y", "z"], keep="first")
        out = simulate_from_answer(ans, wells_df, np.asarray(times, dtype=float), cfg)
        vals = out["concentration_predicted_mg_L"].to_numpy(dtype=float)
        return vals.reshape(shape) if shape is not None else vals
    out = simulate_from_answer(ans, records[["well_id", "x", "y", "z"]].drop_duplicates(), sorted(records["time_days"].unique()), cfg)
    merged = records[["well_id", "time_days"]].copy()
    merged["well_id"] = merged["well_id"].astype(str)
    merged["time_key"] = merged["time_days"].astype(float).round(8)
    pred = out.copy()
    pred["well_id"] = pred["well_id"].astype(str)
    pred["time_key"] = pred["time_days"].astype(float).round(8)
    joined = merged.merge(pred[["well_id", "time_key", "concentration_predicted_mg_L"]], on=["well_id", "time_key"], how="left")
    return joined["concentration_predicted_mg_L"].fillna(0.0).to_numpy(dtype=float)


def predict_from_answer(
    answer: str | Path | dict[str, Any] | None = None,
    wells: Any | None = None,
    times: Any | None = None,
    config: str | Path | dict[str, Any] | None = None,
    **answer_fields: Any,
) -> np.ndarray:
    return predict_concentrations(answer=answer, wells=wells, times=times, config=config, **answer_fields)


if __name__ == "__main__":
    data = pd.read_csv("public_monitoring_data.csv")
    pred = predict_from_answer(wells=data)
    print(pd.Series(pred).describe().to_string())
