"""Local public validator for a submitted Borden `forward_model.py`.

The validator compares the submitted prediction function against the public
reference ADE operator on synthetic finite-region source cases.  It does not use
hidden wells, hidden observations, or true source parameters.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import public_forward_model


ROOT = Path(__file__).resolve().parent


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def probe_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    hydro = config.get("hydrogeological_parameters", {})

    def model() -> dict[str, Any]:
        return {
            "equation_type": "advection_dispersion_reaction",
            "governing_equation": "R*dC/dt = div(D grad C) - v dot grad C - lambda*C + source",
            "velocity_m_per_day": float(hydro.get("velocity_m_per_day", 0.02777777777777778)),
            "alpha_L_m": float(hydro.get("alpha_L_m", 10.0)),
            "alpha_TH_m": float(hydro.get("alpha_TH_m", 0.5)),
            "alpha_TV_m": float(hydro.get("alpha_TV_m", 0.01)),
            "porosity": float(hydro.get("porosity", 0.3)),
            "retardation_factor": float(hydro.get("retardation_factor", 1.0)),
            "lambda_per_day": float(hydro.get("lambda_per_day", 0.0)),
            "implementation_file": "forward_model.py",
            "implementation_function": "predict_from_answer",
            "numerical_approach": "public finite-region point3 superposition ADE reference",
        }

    cases = [
        ("centerline_finite_release", 250.0, 230.0, 220.5, 25.0, 15.0, 1.0, 300.0, 200.0, 2500.0),
        ("broad_early_release", 260.0, 225.0, 220.9, 35.0, 20.0, 1.1, 500.0, 0.0, 3200.0),
        ("thin_delayed_release", 230.0, 235.0, 219.0, 20.0, 10.0, 0.8, 200.0, 500.0, 1800.0),
        ("wide_low_concentration_tail", 215.0, 255.0, 218.6, 70.0, 45.0, 2.5, 35.0, 50.0, 6500.0),
        ("late_off_center_release", 310.0, 205.0, 219.8, 18.0, 12.0, 1.4, 260.0, 1800.0, 1400.0),
    ]
    out = []
    for name, x, y, z, hx, hy, hz, c0, t0, duration in cases:
        out.append(
            {
                "name": name,
                "answer": {
                    "source_type": "rectangular_region",
                    "dimension": 3,
                    "x_center": x,
                    "y_center": y,
                    "z_center": z,
                    "half_length_x": hx,
                    "half_length_y": hy,
                    "half_length_z": hz,
                    "C0": c0,
                    "t_start": t0,
                    "duration": duration,
                    "transport_model": model(),
                    "method": "public local ADE probe",
                },
            }
        )
    return out


def expanded_records(wells_df: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    records = []
    for _, w in wells_df.iterrows():
        for t in times:
            rec = dict(w)
            rec["well_id"] = str(rec.get("well_id", "well"))
            rec["time_days"] = float(t)
            rec["time_years"] = float(t) / 365.0
            records.append(rec)
    return pd.DataFrame.from_records(records)


def import_submission():
    path = ROOT / "forward_model.py"
    if not path.exists():
        raise FileNotFoundError("forward_model.py is missing; copy/adapt public_forward_model.py first")
    spec = importlib.util.spec_from_file_location("submitted_forward_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import forward_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ["predict_from_answer", "predict_concentrations", "simulate_from_answer", "predict"]:
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    raise RuntimeError("forward_model.py has no recognized prediction function")


def as_values(obj: Any, expanded_df: pd.DataFrame, wells_df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    if hasattr(obj, "columns"):
        df = obj.copy()
        pred_col = None
        for name in ["concentration_predicted_mg_L", "predicted_concentration_mg_L", "concentration_mg_L", "predicted", "pred", "C"]:
            if name in df.columns:
                pred_col = name
                break
        if pred_col is None:
            raise ValueError(f"prediction DataFrame has no concentration column: {list(df.columns)}")
        if "well_id" in df.columns and "time_days" in df.columns:
            left = expanded_df[["well_id", "time_days"]].copy()
            left["well_id"] = left["well_id"].astype(str)
            left["time_key"] = left["time_days"].astype(float).round(8)
            right = df[["well_id", "time_days", pred_col]].copy()
            right["well_id"] = right["well_id"].astype(str)
            right["time_key"] = right["time_days"].astype(float).round(8)
            merged = left.merge(right[["well_id", "time_key", pred_col]], on=["well_id", "time_key"], how="left")
            return merged[pred_col].to_numpy(dtype=float)
        return df[pred_col].to_numpy(dtype=float)
    arr = np.asarray(obj, dtype=float)
    n_expected = len(expanded_df)
    if arr.size == n_expected:
        return arr.reshape(-1)
    if arr.shape == (len(wells_df), len(times)):
        return arr.reshape(-1)
    if arr.shape == (len(times), len(wells_df)):
        return arr.T.reshape(-1)
    raise ValueError(f"prediction shape {arr.shape} does not match {n_expected} records")


def call_submission(fn, answer: dict[str, Any], wells_df: pd.DataFrame, times: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    expanded_df = expanded_records(wells_df, times)
    attempts = [
        ((), {"answer": answer, "wells": expanded_df, "times": None, "config": config}),
        ((), {"answer": answer, "wells": expanded_df, "times": None}),
        ((answer, expanded_df, None, config), {}),
        ((answer, expanded_df, None), {}),
        ((answer, wells_df, times, config), {}),
        ((answer, wells_df, times), {}),
    ]
    errors = []
    for args, kwargs in attempts:
        try:
            return as_values(fn(*args, **kwargs), expanded_df, wells_df, times)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError("; ".join(errors[-3:]))


def score_case(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ref = np.maximum(np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    pred = np.maximum(np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    rel_rmse = float(np.sqrt(np.mean((pred - ref) ** 2)) / max(np.mean(np.abs(ref)), 1e-8))
    log_rmse = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(ref)) ** 2)))
    ref_p90 = float(np.percentile(ref, 90))
    pred_p90 = float(np.percentile(pred, 90))
    eps = max(ref_p90 * 1e-4, 1e-9)
    scale_log_error = abs(math.log((pred_p90 + eps) / (ref_p90 + eps)))
    return {
        "relative_rmse": rel_rmse,
        "log_rmse": log_rmse,
        "scale_log_error": scale_log_error,
    }


def main() -> int:
    config = load_json(ROOT / "public_problem_config.json")
    wells = pd.read_csv(ROOT / "public_wells.csv").head(6)
    times = np.asarray([1095.0, 1631.55, 2146.2, 2825.1, 3412.75, 4270.5, 4836.25], dtype=float)
    fn_name, fn = import_submission()

    failures = []
    for case in probe_cases(config):
        answer = case["answer"]
        ref_df = public_forward_model.simulate_from_answer(answer, wells, times, config)
        reference = ref_df["concentration_predicted_mg_L"].to_numpy(dtype=float)
        predicted = call_submission(fn, answer, wells, times, config)
        metrics = score_case(reference, predicted)
        ok = metrics["relative_rmse"] <= 0.12 and metrics["scale_log_error"] <= 0.10
        print(
            f"{case['name']}: {'OK' if ok else 'FAIL'} "
            f"relative_rmse={metrics['relative_rmse']:.4g} "
            f"log_rmse={metrics['log_rmse']:.4g} "
            f"scale_log_error={metrics['scale_log_error']:.4g}"
        )
        if not ok:
            failures.append(case["name"])

    if failures:
        print("Forward model does not yet match the public ADE reference on:", ", ".join(failures))
        return 1
    print(f"forward_model.py validated against public ADE reference using {fn_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
