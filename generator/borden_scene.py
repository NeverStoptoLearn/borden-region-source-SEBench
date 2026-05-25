
"""
Borden-AdePy region-source task core.

This file keeps the original Borden-style grid and hydrogeological parameters from
borden_adepy_reproduction.ipynb, but changes the inverse-problem parameterization
from a compact point source to a finite-duration rectangular region source.

Key change:
    old theta = (x0, y0, z0, C0)
    new theta = (x_center, y_center, z_center, half_length_x, half_length_y,
                 half_length_z, C0, t_start, duration)

The finite rectangular region source is evaluated by deterministic superposition:
rectangular region -> small set of sub-source points -> finite release time steps ->
sum of 3D ADE point responses. This is intentionally CPU-only and deterministic.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Any

import numpy as np
import pandas as pd

GRID_DEFAULTS = {
    "Lx": 1200.0,
    "Ly": 600.0,
    "ncol": 131,
    "nrow": 65,
    "nlay": 21,
    "z_top_flat": 222.0,
}

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

SOURCE_ZONE_INDICES = {
    "source_row_start": 20,
    "source_row_end": 30,
    "source_col_start": 20,
    "source_col_end": 35,
}

BOTTOM_PROFILE = {
    "x_ref_m": [0.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0],
    "z_bot_ref_m": [192.0, 195.0, 199.0, 204.0, 208.0, 210.0, 211.0, 211.0, 211.0],
}


def estimate_uniform_velocity(hk: float = 10.0, porosity: float = 0.3,
                              head_upstream: float = 220.0, head_downstream: float = 219.0,
                              Lx: float = 1200.0) -> float:
    hydraulic_gradient = (head_upstream - head_downstream) / Lx
    darcy_flux = hk * hydraulic_gradient
    return darcy_flux / porosity


def build_borden_grid(Lx: float = 1200.0, Ly: float = 600.0,
                      ncol: int = 131, nrow: int = 65, nlay: int = 21,
                      z_top_flat: float = 222.0) -> Dict[str, Any]:
    delr = Lx / ncol
    delc = Ly / nrow
    x_centers = (np.arange(ncol) + 0.5) * delr
    y_centers = (np.arange(nrow) + 0.5) * delc
    X2, Y2 = np.meshgrid(x_centers, y_centers)
    x_ref = np.asarray(BOTTOM_PROFILE["x_ref_m"], dtype=float)
    z_bot_ref = np.asarray(BOTTOM_PROFILE["z_bot_ref_m"], dtype=float)
    ZBOT2 = np.interp(X2, x_ref, z_bot_ref)
    X3 = np.broadcast_to(X2, (nlay, nrow, ncol)).copy()
    Y3 = np.broadcast_to(Y2, (nlay, nrow, ncol)).copy()
    layer_frac = (np.arange(nlay) + 0.5) / nlay
    Z3 = np.empty((nlay, nrow, ncol), dtype=float)
    for k, frac in enumerate(layer_frac):
        Z3[k] = z_top_flat - frac * (z_top_flat - ZBOT2)
    return {
        "X3": X3, "Y3": Y3, "Z3": Z3,
        "X2": X2, "Y2": Y2, "ZBOT2": ZBOT2,
        "x_centers": x_centers, "y_centers": y_centers,
        "delr": delr, "delc": delc,
        "z_top_flat": z_top_flat,
        "x_ref": x_ref, "z_bot_ref": z_bot_ref,
        "Lx": Lx, "Ly": Ly, "ncol": ncol, "nrow": nrow, "nlay": nlay,
    }


def source_zone_bounds(grid: Dict[str, Any]) -> Dict[str, float]:
    delr, delc = grid["delr"], grid["delc"]
    s = SOURCE_ZONE_INDICES
    return {
        "source_x_min_m": s["source_col_start"] * delr,
        "source_x_max_m": s["source_col_end"] * delr,
        "source_y_min_m": s["source_row_start"] * delc,
        "source_y_max_m": s["source_row_end"] * delc,
    }


def z_bottom_at_x(x: float) -> float:
    return float(np.interp(float(x), BOTTOM_PROFILE["x_ref_m"], BOTTOM_PROFILE["z_bot_ref_m"]))


def top_layer_center_z(x: float, nlay: int = 21, z_top_flat: float = 222.0) -> float:
    zbot = z_bottom_at_x(x)
    return float(z_top_flat - 0.5 * (z_top_flat - zbot) / nlay)


def default_region_source() -> Dict[str, float | str | int]:
    """Hidden generation source. Not used directly by the agent."""
    grid = build_borden_grid()
    b = source_zone_bounds(grid)
    xc = 0.5 * (b["source_x_min_m"] + b["source_x_max_m"]) + 8.0
    yc = 0.5 * (b["source_y_min_m"] + b["source_y_max_m"]) - 5.0
    zc = top_layer_center_z(xc) - 0.6
    return {
        "source_type": "rectangular_region",
        "dimension": 3,
        "x_center": float(xc),
        "y_center": float(yc),
        "z_center": float(zc),
        "half_length_x": 32.0,
        "half_length_y": 18.0,
        "half_length_z": 1.2,
        "C0": 420.0,
        "t_start": 730.0,
        "duration": 2555.0,
    }


def _fallback_point3(c0: float, x, y, z, t, v: float, porosity: float,
                     al: float, ah: float, av: float, q_source: float,
                     source_x: float, source_y: float, source_z: float,
                     Dm: float = 0.0, lamb: float = 0.0, R: float = 1.0,
                     source_scale_factor: float = 1000.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    t = np.maximum(t / max(R, 1e-12), 1e-6)
    DL = max(al * abs(v) + Dm, 1e-8)
    DT = max(ah * abs(v) + Dm, 1e-8)
    DV = max(av * abs(v) + Dm, 1e-8)
    dx = x - source_x - v * t
    dy = y - source_y
    dz = z - source_z
    expo = -((dx * dx) / (4 * DL * t) + (dy * dy) / (4 * DT * t) + (dz * dz) / (4 * DV * t))
    denom = (4 * math.pi * t) ** 1.5 * math.sqrt(DL * DT * DV)
    val = c0 * q_source * source_scale_factor / np.maximum(denom, 1e-30) * np.exp(expo) * np.exp(-lamb * t)
    return np.maximum(val, 0.0)


def _get_point3():
    try:
        from adepy.uniform import point3  # type: ignore
        return point3, True
    except Exception:
        return None, False


def normalize_region_answer(source_params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float | str | int]:
    """Allow a point-source answer to be interpreted as a degenerate region, but score it weakly."""
    p = dict(source_params)
    b = config.get("source_search_bounds_for_agent", {})
    if "x_center" not in p and "x0" in p:
        p["x_center"] = p.get("x0")
    if "y_center" not in p and "y0" in p:
        p["y_center"] = p.get("y0")
    if "z_center" not in p and "z0" in p:
        p["z_center"] = p.get("z0")
    p.setdefault("source_type", "rectangular_region")
    p.setdefault("dimension", 3)
    p.setdefault("half_length_x", b.get("half_length_x_min", 5.0))
    p.setdefault("half_length_y", b.get("half_length_y_min", 5.0))
    p.setdefault("half_length_z", b.get("half_length_z_min", 0.5))
    p.setdefault("C0", 100.0)
    p.setdefault("t_start", b.get("t_start_min", 0.0))
    p.setdefault("duration", b.get("duration_min", 100.0))
    out = {"source_type": str(p.get("source_type", "rectangular_region")), "dimension": int(p.get("dimension", 3))}
    for k in ["x_center", "y_center", "z_center", "half_length_x", "half_length_y", "half_length_z", "C0", "t_start", "duration"]:
        out[k] = float(p[k])
    return out


def _region_subpoints(source: Dict[str, Any], config: Dict[str, Any]):
    disc = config.get("region_source_discretization", {})
    nx = int(disc.get("nx", 5))
    ny = int(disc.get("ny", 5))
    nz = int(disc.get("nz", 1))
    xs = np.linspace(source["x_center"] - source["half_length_x"], source["x_center"] + source["half_length_x"], nx)
    ys = np.linspace(source["y_center"] - source["half_length_y"], source["y_center"] + source["half_length_y"], ny)
    zs = np.linspace(source["z_center"] - source["half_length_z"], source["z_center"] + source["half_length_z"], nz)
    pts = [(float(x), float(y), float(z)) for x in xs for y in ys for z in zs]
    return pts


def _release_offsets(duration: float, config: Dict[str, Any]):
    n_steps = int(config.get("region_source_discretization", {}).get("release_steps", 9))
    duration = max(float(duration), 1.0)
    if n_steps <= 1:
        return np.array([0.5 * duration], dtype=float)
    return np.linspace(0.0, duration, n_steps)


def _point_response(c0, x, y, z, t_arr, x0, y0, z0, hydro, config, prefer_adepy=False):
    porosity = float(hydro.get("porosity", HYDRO_DEFAULTS["porosity"]))
    v = float(hydro.get("velocity_m_per_day", estimate_uniform_velocity(
        hk=float(hydro.get("hk_m_per_day", HYDRO_DEFAULTS["hk_m_per_day"])),
        porosity=porosity,
        head_upstream=float(hydro.get("head_upstream_m", HYDRO_DEFAULTS["head_upstream_m"])),
        head_downstream=float(hydro.get("head_downstream_m", HYDRO_DEFAULTS["head_downstream_m"])),
        Lx=float(config.get("grid", {}).get("domain_length_x_m", GRID_DEFAULTS["Lx"])),
    )))
    al = float(hydro.get("alpha_L_m", HYDRO_DEFAULTS["alpha_L_m"]))
    ah = float(hydro.get("alpha_TH_m", HYDRO_DEFAULTS["alpha_TH_m"]))
    av = float(hydro.get("alpha_TV_m", HYDRO_DEFAULTS["alpha_TV_m"]))
    dm = float(hydro.get("Dm_m2_per_day", HYDRO_DEFAULTS["Dm_m2_per_day"]))
    lamb = float(hydro.get("lambda_per_day", HYDRO_DEFAULTS["lambda_per_day"]))
    R = float(hydro.get("retardation_factor", HYDRO_DEFAULTS["retardation_factor"]))
    q_source = float(hydro.get("q_source_m3_per_day", HYDRO_DEFAULTS["q_source_m3_per_day"]))
    fallback_scale = float(hydro.get("fallback_source_scale_factor", HYDRO_DEFAULTS["fallback_source_scale_factor"]))
    point3, has_adepy = _get_point3()
    if prefer_adepy and has_adepy:
        try:
            return point3(c0, x, y, z, t_arr, v, porosity, al, ah, av, q_source, x0, y0, z0, Dm=dm, lamb=lamb, R=R)
        except Exception:
            pass
    return _fallback_point3(c0, x, y, z, t_arr, v, porosity, al, ah, av, q_source, x0, y0, z0, Dm=dm, lamb=lamb, R=R, source_scale_factor=fallback_scale)


def simulate_region_monitoring_concentrations(source_params: Dict[str, Any], wells: pd.DataFrame,
                                              times_days: Iterable[float], config: Dict[str, Any] | None = None,
                                              prefer_adepy: bool = False) -> pd.DataFrame:
    if config is None:
        config = build_problem_config()
    source = normalize_region_answer(source_params, config)
    hydro = config.get("hydrogeological_parameters", {})
    subpoints = _region_subpoints(source, config)
    release_offsets = _release_offsets(source["duration"], config)
    t_start = float(source["t_start"])
    C0 = float(source["C0"])
    n_norm = max(len(subpoints) * len(release_offsets), 1)
    times_arr = np.asarray(list(times_days), dtype=float)
    records = []
    for _, w in wells.iterrows():
        x = float(w["x"]); y = float(w["y"]); z = float(w["z"])
        conc_total = np.zeros_like(times_arr, dtype=float)
        for sx, sy, sz in subpoints:
            for off in release_offsets:
                tau = t_start + float(off)
                eff_t = times_arr - tau
                active = eff_t > 0.0
                if not np.any(active):
                    continue
                c = np.zeros_like(times_arr, dtype=float)
                c[active] = _point_response(C0 / n_norm, x, y, z, eff_t[active], sx, sy, sz, hydro, config, prefer_adepy=prefer_adepy)
                conc_total += c
        conc_total = np.nan_to_num(conc_total, nan=0.0, posinf=0.0, neginf=0.0)
        conc_total = np.maximum(conc_total, 0.0)
        for t, c in zip(times_arr, conc_total):
            records.append({
                "well_id": str(w["well_id"]),
                "x": x, "y": y, "z": z,
                "time_days": float(t),
                "time_years": float(t) / 365.0,
                "concentration_mg_L": float(c),
            })
    return pd.DataFrame.from_records(records)


def add_measurement_noise(df: pd.DataFrame, noise_relative: float = 0.03, noise_absolute: float = 0.002,
                          detection_limit: float = 0.005, seed: int = 20260522) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    clean = out["concentration_mg_L"].to_numpy(dtype=float)
    sigma = noise_absolute + noise_relative * np.maximum(clean, 0.0)
    obs = clean + rng.normal(0.0, sigma)
    obs = np.maximum(obs, 0.0)
    out["concentration_clean_mg_L"] = clean
    above = obs >= detection_limit
    # Agent-visible observations are censored at the detection limit. Keep the
    # clean values private for scoring calibration and hidden evaluation.
    obs_censored = obs.copy()
    obs_censored[~above] = 0.0
    out["concentration_observed_mg_L"] = obs_censored
    out["above_detection_limit"] = obs >= detection_limit
    out["detection_limit_mg_L"] = detection_limit
    return out


def build_problem_config() -> Dict[str, Any]:
    grid = build_borden_grid()
    hydro = dict(HYDRO_DEFAULTS)
    hydro["velocity_m_per_day"] = estimate_uniform_velocity(
        hk=hydro["hk_m_per_day"], porosity=hydro["porosity"],
        head_upstream=hydro["head_upstream_m"], head_downstream=hydro["head_downstream_m"],
        Lx=GRID_DEFAULTS["Lx"])
    source_bounds = source_zone_bounds(grid)
    z_min = top_layer_center_z(source_bounds["source_x_max_m"]) - 5.0
    z_max = GRID_DEFAULTS["z_top_flat"]
    return {
        "task_name": "Borden-style 3D finite-duration rectangular-region source inversion",
        "random_seed": 20260522,
        "source_parameterization": "finite_duration_rectangular_region",
        "coordinate_system": {
            "x_description": "main groundwater flow direction",
            "y_description": "transverse horizontal direction",
            "z_description": "elevation in meters; top water-table elevation is about 222 m",
            "length_unit": "m",
            "time_unit": "day",
            "concentration_unit": "mg/L",
        },
        "grid": {
            "domain_length_x_m": GRID_DEFAULTS["Lx"],
            "domain_length_y_m": GRID_DEFAULTS["Ly"],
            "ncol": GRID_DEFAULTS["ncol"],
            "nrow": GRID_DEFAULTS["nrow"],
            "nlay": GRID_DEFAULTS["nlay"],
            "delr_m": grid["delr"],
            "delc_m": grid["delc"],
            "z_top_m": GRID_DEFAULTS["z_top_flat"],
            "bottom_profile_x_m": BOTTOM_PROFILE["x_ref_m"],
            "bottom_profile_z_m": BOTTOM_PROFILE["z_bot_ref_m"],
        },
        "hydrogeological_parameters": hydro,
        "borden_source_zone_from_original_scene": {
            **SOURCE_ZONE_INDICES,
            **source_bounds,
            "description": "Original Borden landfill source-zone prior from the notebook; the task uses a finite-duration rectangular-region source inside/near this prior zone.",
        },
        "source_search_bounds_for_agent": {
            "x_center_min": max(0.0, source_bounds["source_x_min_m"] - 40.0),
            "x_center_max": min(GRID_DEFAULTS["Lx"], source_bounds["source_x_max_m"] + 70.0),
            "y_center_min": max(0.0, source_bounds["source_y_min_m"] - 50.0),
            "y_center_max": min(GRID_DEFAULTS["Ly"], source_bounds["source_y_max_m"] + 50.0),
            "z_center_min": float(z_min),
            "z_center_max": float(z_max),
            "half_length_x_min": 5.0,
            "half_length_x_max": 80.0,
            "half_length_y_min": 5.0,
            "half_length_y_max": 60.0,
            "half_length_z_min": 0.5,
            "half_length_z_max": 5.0,
            "C0_min": 20.0,
            "C0_max": 800.0,
            "t_start_min": 0.0,
            "t_start_max": 3000.0,
            "duration_min": 100.0,
            "duration_max": 8000.0,
        },
        "region_source_discretization": {
            "nx": 5,
            "ny": 5,
            "nz": 1,
            "release_steps": 9,
            "note": "Judge approximates rectangular-region finite release by superposing point-source ADE responses over sub-source points and release times.",
        },
        "known_files_for_agent": {
            "well_file": "public_wells.csv",
            "monitoring_file": "public_monitoring_data.csv",
            "grid_file": "borden_grid.npz",
            "source_prior_file": "public_source_prior.json",
        },
        "data_column_description": {
            "well_id": "monitoring well ID",
            "x": "well x coordinate in m",
            "y": "well y coordinate in m",
            "z": "well elevation in m",
            "time_days": "observation time in days",
            "time_years": "observation time in years",
            "concentration_clean_mg_L": "noise-free generated concentration, if present",
            "concentration_observed_mg_L": "observed concentration with measurement noise",
            "above_detection_limit": "whether observed concentration is above detection limit",
        },
        "important_note": "The judge evaluates finite-duration rectangular-region source predictions on withheld monitoring wells/times. Easy format/prior checks sum to <=15 points; hidden monitoring prediction dominates the score.",
    }


def default_public_wells() -> pd.DataFrame:
    rows = [
        ("W01", 300.0, 230.0),
        ("W02", 380.0, 230.0),
        ("W03", 460.0, 240.0),
        ("W04", 560.0, 250.0),
        ("W05", 520.0, 205.0),
        ("W06", 520.0, 295.0),
        ("W07", 680.0, 260.0),
        ("W08", 760.0, 210.0),
    ]
    return pd.DataFrame([{"well_id": w, "x": x, "y": y, "z": top_layer_center_z(x)-0.3} for w, x, y in rows])


def default_hidden_wells() -> pd.DataFrame:
    rows = [
        ("H01", 420.0, 215.0),
        ("H02", 500.0, 270.0),
        ("H03", 610.0, 235.0),
        ("H04", 760.0, 275.0),
    ]
    return pd.DataFrame([{"well_id": w, "x": x, "y": y, "z": top_layer_center_z(x)-0.3} for w, x, y in rows])


def public_times() -> np.ndarray:
    # Irregular sparse sampling spanning early rise, peak neighborhood, and a
    # small late tail. This makes public-only curve fitting less exact while
    # still giving enough signal for a two-hour agent run to improve.
    return np.array([
        3.00, 3.28, 3.63, 4.05, 4.47, 4.93, 5.36, 5.88,
        6.41, 7.02, 7.74, 8.55, 9.35, 10.40, 11.70, 13.25,
    ], dtype=float) * 365.0


def hidden_well_times() -> np.ndarray:
    return np.linspace(6.0 * 365.0, 20.0 * 365.0, 48)


def future_times() -> np.ndarray:
    return np.linspace(8.5 * 365.0, 20.0 * 365.0, 42)


def write_grid_npz(path: str | Path) -> None:
    path = Path(path)
    grid = build_borden_grid()
    np.savez_compressed(
        path,
        x_centers=grid["x_centers"],
        y_centers=grid["y_centers"],
        X2=grid["X2"],
        Y2=grid["Y2"],
        ZBOT2=grid["ZBOT2"],
        Z3=grid["Z3"],
        delr=np.array([grid["delr"]]),
        delc=np.array([grid["delc"]]),
        z_top_flat=np.array([grid["z_top_flat"]]),
        x_ref=grid["x_ref"],
        z_bot_ref=grid["z_bot_ref"],
    )
