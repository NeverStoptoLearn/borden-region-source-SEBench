from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd



def public_prediction_band(metrics: dict[str, Any]) -> str:
    hidden = metrics.get("hidden_well", {}) or {}
    future = metrics.get("future_public", {}) or {}
    hidden_rrmse = safe_float(hidden.get("rRMSE"), float("inf"))
    future_rrmse = safe_float(future.get("rRMSE"), float("inf"))
    if hidden_rrmse < 0.05 and future_rrmse < 0.12:
        return "strong"
    if hidden_rrmse < 0.10 and future_rrmse < 0.16:
        return "good"
    if hidden_rrmse < 0.18 and future_rrmse < 0.22:
        return "acceptable"
    if hidden_rrmse < 0.20 and future_rrmse < 0.25:
        return "moderate"
    if hidden_rrmse < 0.85 and future_rrmse < 0.90:
        return "weak"
    return "poor"


def safe_summary(detail: dict[str, Any]) -> str:
    try:
        feedback = project_review_feedback(detail)
        return (
            f"REVIEW_STATUS={feedback.get('review_status', 'needs_revision')}; "
            f"PROCESS_STAGE={feedback.get('process_stage', 'unstarted')}; "
            f"MODEL_STATUS={feedback.get('model_status', 'not_reviewed')}; "
            f"PUBLIC_FIT={feedback.get('public_fit', 'not_reviewed')}; "
            f"VALIDATION_STATUS={feedback.get('validation_status', 'not_reviewed')}; "
            f"NEXT_REVIEW={feedback.get('next_review', 'Review data, model, inversion evidence, and report completeness.')}"
        )
    except Exception:
        return "REVIEW_STATUS=needs_revision; PROCESS_STAGE=unknown; NEXT_REVIEW=Review submission package completeness."
TASK_NAME = "borden_inverse"
ADE_MODEL_GATE_SCORE = 8.0
ADE_CORRECTNESS_MAX_SCORE = 10.0
ADE_CORRECTNESS_GATE_SCORE = 7.5
ADE_CORRECTNESS_FAIL_CAP = 15.0
PROCESS_SCORE_MAX = 100.0


# ============================================================
# Basic IO
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


# ============================================================
# Continuous metric scoring
# ============================================================


def round3(x):
    try:
        return round(float(x), 3)
    except Exception:
        return 0.0


def log_power_score(error: float, good: float, bad: float, points: float, gamma: float = 2.5) -> float:
    # Truncated logarithmic-power score for lower-is-better metrics.
    # error <= good: full points; error >= bad: zero.
    # good < error < bad: points * z**gamma on a logarithmic scale.
    try:
        error = float(error)
        good = float(good)
        bad = float(bad)
        points = float(points)
        gamma = float(gamma)
    except Exception:
        return 0.0

    if not math.isfinite(error) or not math.isfinite(good) or not math.isfinite(bad):
        return 0.0
    if good <= 0.0 or bad <= good or points <= 0.0:
        return 0.0
    if error <= good:
        return points
    if error >= bad:
        return 0.0

    z = (math.log(bad) - math.log(error)) / (math.log(bad) - math.log(good))
    z = max(0.0, min(1.0, z))
    return points * (z ** gamma)


def smooth_linear_score(error: float, good: float, bad: float, points: float, gamma: float = 1.3) -> float:
    # Wide, gentle slope for poor-to-moderate answers. This makes hidden metric
    # feedback useful before answers are accurate enough for the precision score.
    try:
        error = float(error)
        good = float(good)
        bad = float(bad)
        points = float(points)
        gamma = float(gamma)
    except Exception:
        return 0.0
    if not math.isfinite(error) or bad <= good or points <= 0.0:
        return 0.0
    if error <= good:
        return points
    if error >= bad:
        return 0.0
    z = (bad - error) / (bad - good)
    return points * (max(0.0, min(1.0, z)) ** gamma)


def metric_cap_reason(metrics: dict) -> str:
    hidden = metrics.get("hidden_well", {})
    future = metrics.get("future_public", {})

    hidden_rrmse = safe_float(hidden.get("rRMSE", float("inf")))
    future_rrmse = safe_float(future.get("rRMSE", float("inf")))
    if hidden_rrmse >= 1.60 or future_rrmse >= 1.60:
        return "cap15_public_only_or_very_poor_hidden"
    if hidden_rrmse >= 0.20 or future_rrmse >= 0.25:
        return "cap30_poor_but_improving_hidden"
    if hidden_rrmse >= 0.12 or future_rrmse >= 0.16:
        return "cap45_moderate_hidden"
    return "none"


def prediction_quality_cap(detail: dict[str, Any]) -> tuple[float, str]:
    """Hard cap from withheld/future predictive credibility.

    A submission with no useful hidden or future predictive power should not get
    a high score from schema, public fit, static text, or source-prior proximity.
    """
    metrics = detail.get("metrics", {}) or {}
    metric_scores = detail.get("metric_score_detail", {}) or {}
    hidden = metrics.get("hidden_well", {}) or {}
    future = metrics.get("future_public", {}) or {}

    hidden_rrmse = safe_float(hidden.get("rRMSE"), float("inf"))
    future_rrmse = safe_float(future.get("rRMSE"), float("inf"))
    hidden_log = safe_float(hidden.get("log_rmse"), float("inf"))
    future_log = safe_float(future.get("log_rmse"), float("inf"))
    hidden_score = safe_float(metric_scores.get("hidden_rrmse_score"), 0.0) + safe_float(
        metric_scores.get("hidden_log_score"), 0.0
    )
    future_score = safe_float(metric_scores.get("future_rrmse_score"), 0.0) + safe_float(
        metric_scores.get("future_log_score"), 0.0
    )

    if hidden_score <= 0.0 and future_score <= 0.0:
        return 14.999, "cap15_no_hidden_or_future_prediction_credit"
    if hidden_rrmse >= 1.60 or future_rrmse >= 1.60:
        return 14.999, "cap15_very_poor_hidden_or_future"
    if hidden_rrmse >= 0.85 or future_rrmse >= 0.90:
        return 14.999, "cap15_prediction_poor"
    if hidden_rrmse >= 0.20 or future_rrmse >= 0.25:
        return 29.999, "cap30_prediction_weak"
    if hidden_rrmse >= 0.12 or future_rrmse >= 0.16:
        return 49.999, "cap50_prediction_moderate"
    if hidden_rrmse >= 0.08 or future_rrmse >= 0.12:
        return 69.999, "cap70_prediction_acceptable"
    if hidden_log >= 0.110 or future_log >= 0.120:
        return 29.999, "cap30_log_prediction_weak"
    return PROCESS_SCORE_MAX, "none"


def ade_quality_cap(detail: dict[str, Any]) -> tuple[float, str]:
    """Cap from submitted ADE forward-model correctness.

    The ADE operator is public, so this cap is primarily a reproducibility and
    anti-surrogate guard. Ranking pressure should come from hidden/future
    predictions, not from reverse-engineering the forward operator.
    """
    if detail.get("model_gate_status") != "passed":
        return 14.999, "cap15_forward_model_not_usable"

    ade_score = safe_float(detail.get("ade_correctness_score"), 0.0)
    if ade_score < 5.0:
        return 24.999, "cap25_ade_unreliable"
    if detail.get("ade_correctness_gate_status") != "passed":
        return 39.999, "cap40_ade_probe_failed"
    if ade_score < 9.0:
        return 79.999, "cap80_ade_minor_mismatch"
    return PROCESS_SCORE_MAX, "none"


def normalized_score(score: Any, max_score: float) -> float:
    score = safe_float(score, 0.0)
    max_score = max(safe_float(max_score, 0.0), 1e-8)
    return max(0.0, min(1.0, score / max_score))


def dependency_quality(model_score: Any, ade_score: Any) -> float:
    """Soft upstream quality used to convert isolated process credit to native credit."""
    model_q = normalized_score(model_score, 9.0)
    ade_q = normalized_score(ade_score, ADE_CORRECTNESS_MAX_SCORE)
    return math.sqrt(max(0.0, model_q * ade_q))


def mandatory_time_progress_cap() -> float | None:
    """
    Mandatory time-envelope cap for iterative runs.

    Prefer a fixed run budget if the harness exposes one. Fall back to elapsed
    time for judge-server submissions. Direct/manual evaluations without either
    environment variable remain uncapped so local debugging is not distorted.
    """
    seconds = safe_float(
        os.environ.get(
            "SEBENCH_RUN_BUDGET_SECONDS",
            os.environ.get("SEBENCH_RUN_ELAPSED_SECONDS"),
        ),
        float("nan"),
    )
    if not math.isfinite(seconds) or seconds < 0.0:
        return None
    grace = 60.0
    if seconds <= 30.0 * 60.0 + grace:
        return 14.999
    if seconds <= 2.0 * 3600.0 + grace:
        return 29.999
    if seconds <= 48.0 * 3600.0 + grace:
        return 69.999
    return PROCESS_SCORE_MAX


def compute_process_scores(detail: dict[str, Any]) -> dict[str, Any]:
    """
    DAG-style process score.

    Downstream source/prediction tasks receive isolated credit from the judge
    reference model even when the submitted forward model is imperfect. A soft
    native multiplier then rewards the same downstream work only to the extent
    that the submitted forward model can carry its own upstream dependencies.
    """
    metric_scores = detail.get("metric_score_detail", {}) or {}

    schema = 2.0 * (
        0.60 * normalized_score(detail.get("format_score", 0.0), 1.0)
        + 0.40 * normalized_score(detail.get("bounds_score", 0.0), 1.0)
    )
    model_static = 2.0 * normalized_score(detail.get("model_relevance_score", 0.0), 9.0)
    ade = 8.0 * normalized_score(
        detail.get("ade_correctness_score", 0.0), ADE_CORRECTNESS_MAX_SCORE
    )
    public_fit = 3.0 * normalized_score(
        detail.get("public_sanity_score_before_ade_gate", detail.get("public_sanity_score", 0.0)),
        2.0,
    )
    hidden_fit = 45.0 * (
        0.75 * normalized_score(metric_scores.get("hidden_rrmse_score", 0.0), 38.0)
        + 0.25 * normalized_score(metric_scores.get("hidden_log_score", 0.0), 12.0)
    )
    future_fit = 30.0 * (
        0.75 * normalized_score(metric_scores.get("future_rrmse_score", 0.0), 30.0)
        + 0.25 * normalized_score(metric_scores.get("future_log_score", 0.0), 12.0)
    )
    source_physics = 8.0 * normalized_score(detail.get("region_physics_score", 0.0), 15.0)
    workflow = 2.0 * (
        0.55 * normalized_score(detail.get("method_workflow_score", 0.0), 1.0)
        + 0.45 * normalized_score(detail.get("transport_equation_score", 0.0), 2.0)
    )

    downstream_isolated = hidden_fit + future_fit + source_physics
    upstream = dependency_quality(
        detail.get("model_relevance_score", 0.0),
        detail.get("ade_correctness_score", 0.0),
    )
    native_multiplier = 0.10 + 0.90 * upstream
    downstream_native = downstream_isolated * native_multiplier

    local_score = schema + model_static + ade + workflow
    carry_score = downstream_isolated
    native_score = downstream_native
    raw_process = local_score + public_fit + downstream_isolated

    downstream_ratio = normalized_score(downstream_isolated, 83.0)
    end_to_end_ratio = upstream * downstream_ratio
    soft_end_to_end_cap = 20.0 + 80.0 * end_to_end_ratio
    score_after_soft_cap = min(raw_process, soft_end_to_end_cap)

    structural_cap = None
    structural_reason = None
    model_detail = detail.get("model_relevance_detail", {}) or {}
    model_failures = set(model_detail.get("failures", []) or [])
    if detail.get("model_gate_status") != "passed":
        structural_cap = 25.0 if "forward_model.py_missing" in model_failures else 30.0
        structural_reason = (
            "cap25_missing_forward_model"
            if "forward_model.py_missing" in model_failures
            else "cap30_incomplete_forward_model"
        )
        score_after_soft_cap = min(score_after_soft_cap, structural_cap)

    time_cap = mandatory_time_progress_cap()
    time_cap_applied = False
    if time_cap is not None and score_after_soft_cap > time_cap:
        score_after_soft_cap = time_cap
        time_cap_applied = True

    components = {
        "schema_bounds": round3(schema),
        "model_static": round3(model_static),
        "ade_correctness": round3(ade),
        "public_fit": round3(public_fit),
        "hidden_fit": round3(hidden_fit),
        "future_fit": round3(future_fit),
        "source_physics": round3(source_physics),
        "workflow": round3(workflow),
    }
    return {
        "components": components,
        "local_score": round3(local_score),
        "carry_score": round3(carry_score),
        "native_score": round3(native_score),
        "raw_process_score": round3(raw_process),
        "upstream_dependency_quality": round3(upstream),
        "native_downstream_multiplier": round3(native_multiplier),
        "downstream_isolated": round3(downstream_isolated),
        "downstream_native": round3(downstream_native),
        "end_to_end_ratio": round3(end_to_end_ratio),
        "soft_end_to_end_cap": round3(soft_end_to_end_cap),
        "structural_cap": structural_cap,
        "structural_cap_reason": structural_reason,
        "time_cap": round3(time_cap) if time_cap is not None else None,
        "time_cap_applied": time_cap_applied,
        "score_after_caps": round3(score_after_soft_cap),
    }

def clipped_linear_score(error: float, good: float, bad: float, points: float) -> float:
    """
    Lower-is-better linear score with clipping.

    error <= good: full points
    error >= bad:  zero points
    good < error < bad: linearly interpolated
    """
    try:
        error = float(error)
    except Exception:
        return 0.0

    if not math.isfinite(error):
        return 0.0
    if error <= good:
        return float(points)
    if error >= bad:
        return 0.0
    return float(points) * (bad - error) / (bad - good)


def score_metric_linear(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """
    Continuous metric score with two layers:
    - coarse score: gentle slope for poor answers so agents can learn direction;
    - precision score: strict high-value reward for true hidden/future fit.
    """
    hidden = metrics.get("hidden_well", {}) or {}
    future = metrics.get("future_public", {}) or {}

    hidden_rrmse = safe_float(hidden.get("rRMSE"), float("inf"))
    future_rrmse = safe_float(future.get("rRMSE"), float("inf"))
    hidden_log = safe_float(hidden.get("log_rmse"), float("inf"))
    future_log = safe_float(future.get("log_rmse"), float("inf"))

    hidden_rrmse_coarse = smooth_linear_score(hidden_rrmse, good=0.85, bad=1.60, points=5, gamma=1.6)
    future_rrmse_coarse = smooth_linear_score(future_rrmse, good=0.90, bad=1.60, points=5, gamma=1.6)
    hidden_log_coarse = smooth_linear_score(hidden_log, good=0.035, bad=0.110, points=2, gamma=1.4)
    future_log_coarse = smooth_linear_score(future_log, good=0.040, bad=0.120, points=2, gamma=1.4)

    hidden_rrmse_precise = log_power_score(hidden_rrmse, good=0.04, bad=0.35, points=33, gamma=2.6)
    future_rrmse_precise = log_power_score(future_rrmse, good=0.05, bad=0.40, points=25, gamma=2.6)
    hidden_log_precise = log_power_score(hidden_log, good=0.010, bad=0.045, points=10, gamma=2.2)
    future_log_precise = log_power_score(future_log, good=0.012, bad=0.055, points=10, gamma=2.2)

    total = (
        hidden_rrmse_coarse + future_rrmse_coarse + hidden_log_coarse + future_log_coarse
        + hidden_rrmse_precise + future_rrmse_precise + hidden_log_precise + future_log_precise
    )
    detail = {
        "hidden_rrmse_coarse": hidden_rrmse_coarse,
        "future_rrmse_coarse": future_rrmse_coarse,
        "hidden_log_coarse": hidden_log_coarse,
        "future_log_coarse": future_log_coarse,
        "hidden_rrmse_precise": hidden_rrmse_precise,
        "future_rrmse_precise": future_rrmse_precise,
        "hidden_log_precise": hidden_log_precise,
        "future_log_precise": future_log_precise,
        "hidden_rrmse_score": hidden_rrmse_coarse + hidden_rrmse_precise,
        "future_rrmse_score": future_rrmse_coarse + future_rrmse_precise,
        "hidden_log_score": hidden_log_coarse + hidden_log_precise,
        "future_log_score": future_log_coarse + future_log_precise,
        "metric_score_linear": total,
    }
    return total, detail


def normalize_time_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time_days" not in out.columns:
        if "time" in out.columns:
            out["time_days"] = out["time"].astype(float)
        else:
            raise ValueError("time_days column not found")
    out["time_key"] = out["time_days"].astype(float).round(8)
    return out


def find_obs_concentration_column(df: pd.DataFrame) -> str:
    candidates = [
        "concentration_clean_mg_L",
        "concentration_observed_mg_L",
        "concentration_mg_L",
        "concentration",
        "obs",
        "observed",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No observation concentration column found. Columns={list(df.columns)}")


def find_pred_concentration_column(df: pd.DataFrame) -> str:
    candidates = [
        "concentration_predicted_mg_L",
        "concentration_mg_L",
        "concentration",
        "pred",
        "predicted",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No prediction concentration column found. Columns={list(df.columns)}")


def compute_censored_public_metrics(obs: pd.DataFrame, pred: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    obs2 = normalize_time_key(obs)
    pred2 = normalize_time_key(pred)
    pred_col = find_pred_concentration_column(pred2)
    merged = obs2.merge(pred2, on=["well_id", "time_key"], suffixes=("_obs", "_pred"), how="inner")
    if merged.empty:
        return {
            "censored_log_rmse": float("inf"),
            "above_log_rmse": float("inf"),
            "below_excess_rate": float("inf"),
            "n": 0,
        }, merged

    pred_col_m = pred_col if pred_col in merged.columns else f"{pred_col}_pred"
    if "concentration_observed_mg_L" in merged.columns:
        obs_col = "concentration_observed_mg_L"
    elif "concentration_observed_mg_L_obs" in merged.columns:
        obs_col = "concentration_observed_mg_L_obs"
    else:
        obs_col = find_obs_concentration_column(merged)
    if "above_detection_limit" in merged.columns:
        above_col = "above_detection_limit"
    elif "above_detection_limit_obs" in merged.columns:
        above_col = "above_detection_limit_obs"
    else:
        above_col = None
    if "detection_limit_mg_L" in merged.columns:
        det = merged["detection_limit_mg_L"].to_numpy(dtype=float)
    elif "detection_limit_mg_L_obs" in merged.columns:
        det = merged["detection_limit_mg_L_obs"].to_numpy(dtype=float)
    else:
        det = np.full(len(merged), 0.005, dtype=float)

    y = np.maximum(merged[obs_col].to_numpy(dtype=float), 0.0)
    p = np.maximum(np.nan_to_num(merged[pred_col_m].to_numpy(dtype=float), nan=0.0), 0.0)
    above = merged[above_col].astype(bool).to_numpy() if above_col else y > det

    parts = []
    if np.any(above):
        parts.append(np.log1p(p[above]) - np.log1p(y[above]))
        above_log = float(np.sqrt(np.mean(parts[-1] * parts[-1])))
    else:
        above_log = 0.0
    if np.any(~above):
        excess = np.maximum(p[~above] - det[~above], 0.0)
        below = np.log1p(excess / np.maximum(det[~above], 1e-8))
        parts.append(below)
        below_excess_rate = float(np.mean(excess > 0.0))
    else:
        below_excess_rate = 0.0
    all_resid = np.concatenate(parts) if parts else np.zeros(1, dtype=float)
    return {
        "censored_log_rmse": float(np.sqrt(np.mean(all_resid * all_resid))),
        "above_log_rmse": above_log,
        "below_excess_rate": below_excess_rate,
        "n": int(len(merged)),
    }, merged


def compute_rmse_metrics(obs: pd.DataFrame, pred: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    obs2 = normalize_time_key(obs)
    pred2 = normalize_time_key(pred)

    obs_col = find_obs_concentration_column(obs2)
    pred_col = find_pred_concentration_column(pred2)

    if "well_id" not in obs2.columns or "well_id" not in pred2.columns:
        raise ValueError("well_id column required in obs and pred")

    merged = obs2.merge(pred2, on=["well_id", "time_key"], suffixes=("_obs", "_pred"), how="inner")
    if merged.empty:
        return {
            "rmse": float("inf"),
            "rRMSE": float("inf"),
            "log_rmse": float("inf"),
            "log_rRMSE": float("inf"),
            "n": 0,
        }, merged

    obs_col_m = obs_col if obs_col in merged.columns else f"{obs_col}_obs"
    pred_col_m = pred_col if pred_col in merged.columns else f"{pred_col}_pred"

    c_obs = merged[obs_col_m].to_numpy(dtype=float)
    c_pred = merged[pred_col_m].to_numpy(dtype=float)
    c_pred = np.nan_to_num(c_pred, nan=0.0, posinf=0.0, neginf=0.0)
    c_pred = np.maximum(c_pred, 0.0)

    diff = c_pred - c_obs
    rmse = float(np.sqrt(np.mean(diff * diff)))
    rrmse = rmse / max(float(np.mean(np.abs(c_obs))), 1e-8)

    log_obs = np.log1p(np.maximum(c_obs, 0.0))
    log_pred = np.log1p(np.maximum(c_pred, 0.0))
    log_rmse = float(np.sqrt(np.mean((log_pred - log_obs) ** 2)))
    log_rrmse = log_rmse / max(float(np.mean(np.abs(log_obs))), 1e-8)

    return {
        "rmse": rmse,
        "rRMSE": rrmse,
        "log_rmse": log_rmse,
        "log_rRMSE": log_rrmse,
        "n": int(len(merged)),
    }, merged


def detect_group_column(df: pd.DataFrame) -> str | None:
    for c in ["eval_split", "eval_group", "group", "subset", "split", "category"]:
        if c in df.columns:
            return c
    return None


def compute_grouped_hidden_metrics(hidden_obs: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    all_metrics, _ = compute_rmse_metrics(hidden_obs, pred)
    metrics["all"] = all_metrics

    group_col = detect_group_column(hidden_obs)
    if group_col is not None:
        groups = {str(g): hidden_obs[hidden_obs[group_col].astype(str) == str(g)].copy() for g in hidden_obs[group_col].dropna().unique()}
        for name, sub in groups.items():
            if len(sub) > 0:
                metrics[name] = compute_rmse_metrics(sub, pred)[0]

    # Fallback aliases if hidden data lacks explicit split labels.
    if "hidden_well" not in metrics:
        metrics["hidden_well"] = metrics.get("all", all_metrics)
    if "future_public" not in metrics:
        metrics["future_public"] = metrics.get("all", all_metrics)

    return metrics


# ============================================================
# Non-metric scoring
# ============================================================

def score_format(answer: dict[str, Any], exists: bool) -> int:
    if not exists:
        return 0

    required = [
        "source_type", "dimension", "x_center", "y_center", "z_center",
        "half_length_x", "half_length_y", "half_length_z", "C0", "t_start", "duration",
    ]
    score = 0
    if all(k in answer for k in required):
        score += 2
    try:
        if int(answer.get("dimension")) == 3 and str(answer.get("source_type", "")) == "rectangular_region":
            score += 1
    except Exception:
        pass
    return min(score, 3)


def get_bounds_dict(config: dict[str, Any]) -> dict[str, Any]:
    # Support both source_prior-style and source_search_bounds-style configs.
    for key in ["source_parameter_bounds", "source_search_bounds_for_agent", "source_bounds"]:
        if isinstance(config.get(key), dict):
            return config[key]
    return config


def range_from_bounds(bounds: dict[str, Any], name: str, fallback: tuple[float, float]) -> tuple[float, float]:
    # Supports either name_range=[lo,hi] or name_min/name_max.
    rkey = f"{name}_range"
    if isinstance(bounds.get(rkey), (list, tuple)) and len(bounds[rkey]) == 2:
        return float(bounds[rkey][0]), float(bounds[rkey][1])
    lo = bounds.get(f"{name}_min", fallback[0])
    hi = bounds.get(f"{name}_max", fallback[1])
    return float(lo), float(hi)


SOURCE_KEYS = [
    "x_center", "y_center", "z_center",
    "half_length_x", "half_length_y", "half_length_z",
    "C0", "t_start", "duration",
]


def source_values(answer: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    vals: dict[str, float] = {}
    failures: list[str] = []
    for name in SOURCE_KEYS:
        vals[name] = safe_float(answer.get(name), float("nan"))
        if not math.isfinite(vals[name]):
            failures.append(f"{name}_missing_or_nonfinite")
    return vals, failures


def source_extents(vals: dict[str, float]) -> dict[str, float]:
    return {
        "x_min": vals["x_center"] - vals["half_length_x"],
        "x_max": vals["x_center"] + vals["half_length_x"],
        "y_min": vals["y_center"] - vals["half_length_y"],
        "y_max": vals["y_center"] + vals["half_length_y"],
        "z_min": vals["z_center"] - vals["half_length_z"],
        "z_max": vals["z_center"] + vals["half_length_z"],
        "release_end": vals["t_start"] + vals["duration"],
    }


def bottom_z_at_x(config: dict[str, Any], x: float) -> float:
    grid = config.get("grid", {}) or {}
    xs = grid.get("bottom_profile_x_m")
    zs = grid.get("bottom_profile_z_m")
    if isinstance(xs, list) and isinstance(zs, list) and len(xs) == len(zs) and xs:
        return float(np.interp(float(x), np.asarray(xs, dtype=float), np.asarray(zs, dtype=float)))
    return float(grid.get("z_bottom_m", 190.0))


def validate_physical_constraints(answer: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    vals, failures = source_values(answer)
    detail: dict[str, Any] = {
        "status": "passed",
        "failures": failures,
        "warnings": [],
        "checks": {},
        "extents": {},
    }
    if failures:
        detail["status"] = "failed"
        return detail

    ext = source_extents(vals)
    detail["extents"] = {k: round3(v) for k, v in ext.items()}

    if vals["half_length_x"] <= 0 or vals["half_length_y"] <= 0 or vals["half_length_z"] <= 0:
        failures.append("nonpositive_region_half_length")
    if vals["C0"] <= 0:
        failures.append("nonpositive_source_concentration")
    if vals["duration"] <= 0 or vals["t_start"] < 0:
        failures.append("invalid_release_time_window")

    grid = config.get("grid", {}) or {}
    domain_x = safe_float(grid.get("domain_length_x_m"), 1200.0)
    domain_y = safe_float(grid.get("domain_length_y_m"), 600.0)
    domain_ok = 0.0 <= ext["x_min"] <= ext["x_max"] <= domain_x and 0.0 <= ext["y_min"] <= ext["y_max"] <= domain_y
    detail["checks"]["domain_extent"] = bool(domain_ok)
    if not domain_ok:
        failures.append("source_region_outside_model_domain")

    zone = config.get("borden_source_zone_from_original_scene", {}) or {}
    delr = safe_float(grid.get("delr_m"), 0.0)
    delc = safe_float(grid.get("delc_m"), 0.0)
    zone_tol = max(delr, delc, 1.0)
    zone_ok = True
    if all(k in zone for k in ["source_x_min_m", "source_x_max_m", "source_y_min_m", "source_y_max_m"]):
        zone_ok = (
            ext["x_min"] >= safe_float(zone["source_x_min_m"]) - zone_tol
            and ext["x_max"] <= safe_float(zone["source_x_max_m"]) + zone_tol
            and ext["y_min"] >= safe_float(zone["source_y_min_m"]) - zone_tol
            and ext["y_max"] <= safe_float(zone["source_y_max_m"]) + zone_tol
        )
    detail["checks"]["source_zone_extent"] = bool(zone_ok)
    if not zone_ok:
        failures.append("source_region_outside_original_source_zone")

    z_top = safe_float(grid.get("z_top_m"), 222.0)
    z_tol = 0.25
    x_samples = [ext["x_min"], vals["x_center"], ext["x_max"]]
    bottom_max = max(bottom_z_at_x(config, x) for x in x_samples)
    aquifer_ok = ext["z_min"] >= bottom_max + z_tol and ext["z_max"] <= z_top + z_tol
    detail["checks"]["aquifer_vertical_extent"] = bool(aquifer_ok)
    detail["checks"]["local_bottom_z_max"] = round3(bottom_max)
    detail["checks"]["z_top_m"] = round3(z_top)
    if not aquifer_ok:
        failures.append("source_region_outside_saturated_vertical_extent")

    detail["status"] = "failed" if failures else "passed"
    detail["failures"] = failures
    return detail


def score_bounds(answer: dict[str, Any], config: dict[str, Any], prior_path: Path | None = None) -> int:
    bounds = get_bounds_dict(config)
    if prior_path and prior_path.exists():
        try:
            prior = load_json(prior_path)
            if isinstance(prior, dict):
                bounds = {**bounds, **prior}
        except Exception:
            pass

    checks = [
        ("x_center", (-1e9, 1e9)),
        ("y_center", (-1e9, 1e9)),
        ("z_center", (-1e9, 1e9)),
        ("half_length_x", (0.0, 1e9)),
        ("half_length_y", (0.0, 1e9)),
        ("half_length_z", (0.0, 1e9)),
        ("C0", (0.0, 1e12)),
        ("t_start", (0.0, 1e12)),
        ("duration", (0.0, 1e12)),
    ]

    ok = True
    for name, fb in checks:
        v = safe_float(answer.get(name), float("nan"))
        lo, hi = range_from_bounds(bounds, name, fb)
        if not (math.isfinite(v) and lo <= v <= hi):
            ok = False
            break

    return 2 if ok else 0


def score_method_and_workflow(answer: dict[str, Any], submission_dir: Path) -> int:
    score = 0
    method = str(answer.get("method", "")).strip()
    if len(method) >= 20:
        score += 2
    elif method:
        score += 1

    py_files = [p for p in submission_dir.glob("*.py") if p.name != "baseline_solver.py"]
    if py_files:
        score += 2

    if (submission_dir / "report.md").exists() or (submission_dir / "results.json").exists():
        score += 1

    return min(score, 5)


def score_transport_equation(answer: dict[str, Any], config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Reward an explicit ADE solute transport construction, but keep it small.
    This is learning feedback for agents; predictive hidden/future metrics still
    control the final caps.
    """
    model = answer.get("transport_model", {})
    if not isinstance(model, dict):
        return 0.0, {"reason": "transport_model_missing_or_not_object"}

    hydro = config.get("hydrogeological_parameters", {}) or {}
    score = 0.0
    detail: dict[str, Any] = {}

    equation_text = " ".join([
        str(model.get("equation_type", "")),
        str(model.get("governing_equation", "")),
        str(model.get("numerical_approach", "")),
    ]).lower()
    terms = {
        "time": ("dc/dt" in equation_text) or ("time" in equation_text),
        "advection": ("advection" in equation_text) or ("v dot grad" in equation_text) or ("velocity" in equation_text),
        "dispersion": ("dispersion" in equation_text) or ("div(d grad" in equation_text) or ("alpha_l" in equation_text),
        "reaction": ("lambda" in equation_text) or ("decay" in equation_text) or ("reaction" in equation_text),
        "source": "source" in equation_text,
    }
    term_count = sum(bool(v) for v in terms.values())
    score += min(3.0, 0.6 * term_count)
    detail["recognized_terms"] = terms

    numeric_checks = [
        ("velocity_m_per_day", hydro.get("velocity_m_per_day")),
        ("alpha_L_m", hydro.get("alpha_L_m")),
        ("alpha_TH_m", hydro.get("alpha_TH_m")),
        ("alpha_TV_m", hydro.get("alpha_TV_m")),
        ("porosity", hydro.get("porosity")),
        ("retardation_factor", hydro.get("retardation_factor")),
        ("lambda_per_day", hydro.get("lambda_per_day")),
    ]
    matched = 0
    hydro_feedback: dict[str, Any] = {}
    for name, expected in numeric_checks:
        got = safe_float(model.get(name), float("nan"))
        exp = safe_float(expected, float("nan"))
        hydro_feedback[name] = {
            "value": got if math.isfinite(got) else None,
            "expected": exp if math.isfinite(exp) else None,
            "status": "missing_or_nonfinite",
        }
        if math.isfinite(got) and math.isfinite(exp):
            tol = max(abs(exp) * 0.25, 1e-8)
            if abs(got - exp) <= tol:
                matched += 1
                hydro_feedback[name]["status"] = "near_public_reference"
            else:
                hydro_feedback[name]["status"] = "far_from_public_reference"
            hydro_feedback[name]["relative_error"] = abs(got - exp) / max(abs(exp), 1e-8)
    score += min(3.0, 3.0 * matched / max(len(numeric_checks), 1))
    detail["matched_public_hydro_parameters"] = matched
    detail["hydro_parameter_feedback"] = hydro_feedback

    approach = str(model.get("numerical_approach", "")).lower()
    detail["numerical_approach_status"] = "missing_or_too_short"
    if len(approach) >= 30:
        score += 1.0
        detail["numerical_approach_status"] = "described"
    if any(k in approach for k in ["optimi", "least", "differential", "anneal", "grid", "multi-start", "multistart"]):
        score += 1.0
        detail["numerical_approach_status"] = "describes_optimization"
    detail["transport_equation_score_raw"] = score
    return min(score, 8.0), detail


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def score_model_relevance(answer: dict[str, Any], submission_dir: Path) -> tuple[float, dict[str, Any]]:
    """
    First-pass static gate for a submitted ADE forward model, not only a
    plausible answer.json. A separate dynamic public-probe gate executes the
    model only after these static checks pass.
    """
    detail: dict[str, Any] = {
        "required_file": "forward_model.py",
        "components": {},
        "failures": [],
    }
    model_path = submission_dir / "forward_model.py"
    if not model_path.exists():
        detail["failures"].append("forward_model.py_missing")
        return 0.0, detail

    try:
        text = model_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        detail["failures"].append(f"forward_model.py_unreadable:{exc!r}")
        return 0.0, detail

    lower = text.lower()
    score = 0.0

    forbidden_terms = [
        "hidden_monitoring_data", "hidden_wells", "hidden_true_region_source",
        "hidden_forward_model", "hidden_eval_config", "/opt/borden_scoring",
        "scoring/", "score.json", "sebench-submit",
    ]
    forbidden_hits = [term for term in forbidden_terms if term in lower]
    detail["forbidden_hits"] = forbidden_hits
    if forbidden_hits:
        detail["failures"].append("forbidden_hidden_or_feedback_reference")
        return 0.0, detail

    function_ok = _contains_any(lower, [
        "def predict_from_answer", "def predict_concentrations",
        "def simulate_from_answer", "def forward",
    ])
    if function_ok:
        score += 1.5
    else:
        detail["failures"].append("no_forward_prediction_function")
    detail["components"]["prediction_function"] = 1.5 if function_ok else 0.0

    required_params = [
        "x_center", "y_center", "z_center",
        "half_length_x", "half_length_y", "half_length_z",
        "c0", "t_start", "duration",
    ]
    param_hits = sum(1 for param in required_params if param in lower)
    param_score = 2.0 * param_hits / len(required_params)
    score += param_score
    detail["components"]["answer_parameter_coupling"] = round3(param_score)
    detail["answer_parameter_hits"] = param_hits
    if param_hits < len(required_params):
        detail["failures"].append("not_all_region_source_parameters_used")

    ade_groups = {
        "time_derivative_or_transient": ["time_days", "tau", "t_start", "duration", "release"],
        "advection": ["advection", "velocity", "vx", "v *", "v_m_per_day"],
        "dispersion": ["dispersion", "alpha_l", "alpha_th", "alpha_tv", "d_l", "dx", "dy", "dz"],
        "reaction_retardation": ["lambda", "decay", "reaction", "retardation", " r "],
        "source_superposition": ["meshgrid", "linspace", "source_points", "subpoints", "quadrature", "for "],
    }
    ade_hits = {name: _contains_any(lower, needles) for name, needles in ade_groups.items()}
    ade_score = 2.5 * sum(ade_hits.values()) / len(ade_groups)
    score += ade_score
    detail["components"]["ade_term_coverage"] = round3(ade_score)
    detail["ade_term_hits"] = ade_hits
    if not all(ade_hits.values()):
        detail["failures"].append("incomplete_ade_term_coverage")

    math_groups = {
        "green_function_shape": ["np.exp", "math.exp", "erfc", "sqrt", "pi"],
        "finite_time_mask": ["> 0", "maximum(", "clip(", "where(", "active"],
        "nonnegative_finite_output": ["nan_to_num", "maximum", "isfinite", "nonnegative"],
        "tabular_prediction_output": ["dataframe", "concentration_predicted", "time_days", "well_id"],
    }
    math_hits = {name: _contains_any(lower, needles) for name, needles in math_groups.items()}
    math_score = 2.0 * sum(math_hits.values()) / len(math_groups)
    score += math_score
    detail["components"]["numerical_forward_model"] = round3(math_score)
    detail["numerical_hits"] = math_hits
    if not all(math_hits.values()):
        detail["failures"].append("incomplete_numerical_forward_model")

    model = answer.get("transport_model", {}) if isinstance(answer, dict) else {}
    reference_text = " ".join([
        str(model.get("implementation_file", "")) if isinstance(model, dict) else "",
        str(model.get("implementation_function", "")) if isinstance(model, dict) else "",
        str(model.get("numerical_approach", "")) if isinstance(model, dict) else "",
        str(answer.get("method", "")) if isinstance(answer, dict) else "",
    ]).lower()
    answer_references_model = "forward_model.py" in reference_text or "predict_from_answer" in reference_text
    if answer_references_model:
        score += 1.0
    else:
        detail["failures"].append("answer_does_not_reference_forward_model")
    detail["components"]["answer_model_traceability"] = 1.0 if answer_references_model else 0.0

    score = min(float(score), 9.0)
    detail["score"] = round3(score)
    detail["passed_gate"] = score >= ADE_MODEL_GATE_SCORE and not forbidden_hits
    if not detail["passed_gate"]:
        detail["failures"].append("ade_model_gate_failed")
    return score, detail


def _clip_to_public_bounds(name: str, value: float, config: dict[str, Any], fallback: tuple[float, float]) -> float:
    bounds = get_bounds_dict(config)
    lo, hi = range_from_bounds(bounds, name, fallback)
    if not math.isfinite(value):
        value = 0.5 * (lo + hi)
    return float(min(max(value, lo), hi))


def _probe_transport_model(config: dict[str, Any], base_model: dict[str, Any] | None = None) -> dict[str, Any]:
    hydro = config.get("hydrogeological_parameters", {}) or {}
    model = dict(base_model or {})
    model.update({
        "equation_type": "advection_dispersion_reaction",
        "governing_equation": "R*dC/dt = div(D grad C) - v dot grad C - lambda*C + source",
        "velocity_m_per_day": safe_float(hydro.get("velocity_m_per_day"), 0.02777777777777778),
        "alpha_L_m": safe_float(hydro.get("alpha_L_m"), 10.0),
        "alpha_TH_m": safe_float(hydro.get("alpha_TH_m"), 0.5),
        "alpha_TV_m": safe_float(hydro.get("alpha_TV_m"), 0.01),
        "porosity": safe_float(hydro.get("porosity"), 0.3),
        "retardation_factor": safe_float(hydro.get("retardation_factor"), 1.0),
        "lambda_per_day": safe_float(hydro.get("lambda_per_day"), 0.0),
        "implementation_file": "forward_model.py",
        "implementation_function": "predict_from_answer",
        "numerical_approach": "ADE correctness probe supplied by judge using public configuration only",
    })
    return model


def _complete_probe_answer(params: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "x_center": 250.0,
        "y_center": 230.0,
        "z_center": 220.5,
        "half_length_x": 25.0,
        "half_length_y": 15.0,
        "half_length_z": 1.0,
        "C0": 300.0,
        "t_start": 200.0,
        "duration": 2500.0,
    }
    fallbacks = {
        "x_center": (0.0, 1200.0),
        "y_center": (0.0, 600.0),
        "z_center": (216.0, 222.0),
        "half_length_x": (5.0, 80.0),
        "half_length_y": (5.0, 60.0),
        "half_length_z": (0.5, 5.0),
        "C0": (20.0, 650.0),
        "t_start": (0.0, 3000.0),
        "duration": (100.0, 8000.0),
    }
    out: dict[str, Any] = {
        "source_type": "rectangular_region",
        "dimension": 3,
    }
    for name in SOURCE_KEYS:
        out[name] = _clip_to_public_bounds(
            name,
            safe_float(params.get(name), defaults[name]),
            config,
            fallbacks[name],
        )
    model = params.get("transport_model", {}) if isinstance(params, dict) else {}
    out["transport_model"] = _probe_transport_model(config, model if isinstance(model, dict) else None)
    out["method"] = "ADE correctness public probe"
    return out


def make_ade_probe_cases(answer: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    vals, failures = source_values(answer)
    if not failures:
        submitted = dict(answer)
        submitted["transport_model"] = _probe_transport_model(
            config,
            answer.get("transport_model", {}) if isinstance(answer.get("transport_model"), dict) else None,
        )
        cases.append({
            "name": "submitted_answer_public_probe",
            "answer": _complete_probe_answer(submitted, config),
        })

    synthetic = [
        (
            "centerline_finite_release",
            {
                "x_center": 250.0,
                "y_center": 230.0,
                "z_center": 220.5,
                "half_length_x": 25.0,
                "half_length_y": 15.0,
                "half_length_z": 1.0,
                "C0": 300.0,
                "t_start": 200.0,
                "duration": 2500.0,
            },
        ),
        (
            "broad_early_release",
            {
                "x_center": 260.0,
                "y_center": 225.0,
                "z_center": 220.9,
                "half_length_x": 35.0,
                "half_length_y": 20.0,
                "half_length_z": 1.1,
                "C0": 500.0,
                "t_start": 0.0,
                "duration": 3200.0,
            },
        ),
        (
            "thin_delayed_release",
            {
                "x_center": 230.0,
                "y_center": 235.0,
                "z_center": 219.0,
                "half_length_x": 20.0,
                "half_length_y": 10.0,
                "half_length_z": 0.8,
                "C0": 200.0,
                "t_start": 500.0,
                "duration": 1800.0,
            },
        ),
        (
            "compact_high_concentration_pulse",
            {
                "x_center": 285.0,
                "y_center": 218.0,
                "z_center": 221.0,
                "half_length_x": 8.0,
                "half_length_y": 6.0,
                "half_length_z": 0.6,
                "C0": 620.0,
                "t_start": 900.0,
                "duration": 700.0,
            },
        ),
        (
            "wide_low_concentration_tail",
            {
                "x_center": 215.0,
                "y_center": 255.0,
                "z_center": 218.6,
                "half_length_x": 70.0,
                "half_length_y": 45.0,
                "half_length_z": 2.5,
                "C0": 35.0,
                "t_start": 50.0,
                "duration": 6500.0,
            },
        ),
        (
            "late_off_center_release",
            {
                "x_center": 310.0,
                "y_center": 205.0,
                "z_center": 219.8,
                "half_length_x": 18.0,
                "half_length_y": 12.0,
                "half_length_z": 1.4,
                "C0": 260.0,
                "t_start": 1800.0,
                "duration": 1400.0,
            },
        ),
        (
            "shallow_vertical_thin_source",
            {
                "x_center": 245.0,
                "y_center": 260.0,
                "z_center": 221.4,
                "half_length_x": 40.0,
                "half_length_y": 8.0,
                "half_length_z": 0.55,
                "C0": 180.0,
                "t_start": 300.0,
                "duration": 3600.0,
            },
        ),
    ]
    for name, params in synthetic:
        cases.append({"name": name, "answer": _complete_probe_answer(params, config)})
    return cases


def make_ade_probe_wells_times(case_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    wells_path = case_dir / "public_wells.csv"
    if wells_path.exists():
        wells = pd.read_csv(wells_path)
        if "well_id" in wells.columns:
            preferred = ["W01", "W02", "W03", "W04", "W05", "W06"]
            subset = wells[wells["well_id"].astype(str).isin(preferred)].copy()
            if len(subset) >= 4:
                wells = subset
        wells = wells.head(6).copy()
    else:
        wells = pd.DataFrame.from_records([
            {"well_id": "W01", "x": 300.0, "y": 230.0, "z": 221.15},
            {"well_id": "W02", "x": 380.0, "y": 230.0, "z": 221.25},
            {"well_id": "W03", "x": 460.0, "y": 240.0, "z": 221.30},
            {"well_id": "W04", "x": 560.0, "y": 250.0, "z": 221.35},
            {"well_id": "W05", "x": 520.0, "y": 205.0, "z": 221.33},
            {"well_id": "W06", "x": 520.0, "y": 295.0, "z": 221.33},
        ])

    obs_path = case_dir / "public_monitoring_data.csv"
    default_times = np.asarray([1095.0, 1631.55, 2146.2, 2825.1, 3412.75, 4270.5, 4836.25], dtype=float)
    if obs_path.exists():
        try:
            obs = pd.read_csv(obs_path)
            public_times = np.asarray(sorted(obs["time_days"].astype(float).unique()), dtype=float)
            if len(public_times) >= len(default_times):
                idx = np.linspace(0, len(public_times) - 1, len(default_times)).round().astype(int)
                default_times = public_times[idx]
        except Exception:
            pass
    return wells, default_times


def _prediction_probe_runner_script() -> str:
    return r'''
import contextlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def _expanded_records(wells_df, times):
    records = []
    for _, w in wells_df.iterrows():
        for t in times:
            rec = {k: w[k] for k in wells_df.columns}
            rec["well_id"] = str(rec.get("well_id", "well"))
            rec["time_days"] = float(t)
            rec["time_years"] = float(t) / 365.0
            records.append(rec)
    return pd.DataFrame.from_records(records)

def _pred_column(df):
    for c in [
        "concentration_predicted_mg_L",
        "predicted_concentration_mg_L",
        "concentration_mg_L",
        "concentration",
        "predicted",
        "pred",
        "c",
        "C",
    ]:
        if c in df.columns:
            return c
    return None

def _as_values(obj, expanded_df, wells_df, times):
    n_expected = len(expanded_df)
    if isinstance(obj, dict):
        for key in [
            "concentration_predicted_mg_L",
            "predicted_concentration_mg_L",
            "concentration_mg_L",
            "concentration",
            "predictions",
            "predicted",
            "pred",
            "values",
        ]:
            if key in obj:
                return _as_values(obj[key], expanded_df, wells_df, times)
    if hasattr(obj, "columns"):
        df = obj.copy()
        col = _pred_column(df)
        if col is None:
            raise ValueError("prediction DataFrame has no recognized concentration column")
        if "well_id" in df.columns and "time_days" in df.columns:
            df = df.copy()
            df["well_id"] = df["well_id"].astype(str)
            df["time_key"] = df["time_days"].astype(float).round(8)
            left = expanded_df[["well_id", "time_days"]].copy()
            left["well_id"] = left["well_id"].astype(str)
            left["time_key"] = left["time_days"].astype(float).round(8)
            merged = left.merge(df[["well_id", "time_key", col]], on=["well_id", "time_key"], how="left")
            vals = merged[col].to_numpy(dtype=float)
            if np.isfinite(vals).sum() == n_expected:
                return vals
        vals = df[col].to_numpy(dtype=float)
        if vals.size == n_expected:
            return vals.reshape(-1)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 0:
        raise ValueError("prediction output is scalar")
    if arr.size == n_expected:
        return arr.reshape(-1)
    if arr.shape == (len(wells_df), len(times)):
        return arr.reshape(-1)
    if arr.shape == (len(times), len(wells_df)):
        return arr.T.reshape(-1)
    raise ValueError(f"prediction output shape {arr.shape} cannot be aligned to {n_expected} probe records")

def _call_candidates(fn, probe_answer, expanded_df, wells_df, times, config):
    call_styles = [
        ("kwargs_answer_expanded", (), {"answer": probe_answer, "wells": expanded_df, "times": None}),
        ("kwargs_answer_expanded_times", (), {"answer": probe_answer, "wells": expanded_df, "times": times}),
        ("answer_expanded_none", (probe_answer, expanded_df, None), {}),
        ("answer_expanded_times", (probe_answer, expanded_df, times), {}),
        ("answer_wells_times_config", (probe_answer, wells_df, times, config), {}),
        ("expanded_answer_kw", (expanded_df,), {"answer": probe_answer}),
        ("expanded_times_answer_kw", (expanded_df, times), {"answer": probe_answer}),
        ("expanded_only_after_answer_json", (expanded_df,), {}),
        ("expanded_times_after_answer_json", (expanded_df, times), {}),
        ("no_args_after_answer_json", (), {}),
    ]
    errors = []
    for label, args, kwargs in call_styles:
        try:
            out = fn(*args, **kwargs)
            vals = _as_values(out, expanded_df, wells_df, times)
            return label, vals
        except Exception as exc:
            errors.append(f"{label}:{type(exc).__name__}:{str(exc)[:160]}")
    raise RuntimeError("; ".join(errors[-5:]))

payload = json.load(sys.stdin)
orig_stdout = sys.stdout
try:
    sys.path.insert(0, str(Path.cwd()))
    spec = importlib.util.spec_from_file_location("agent_forward_model", Path.cwd() / "forward_model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load forward_model.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)

    preferred = []
    for case in payload["cases"]:
        impl = case.get("answer", {}).get("transport_model", {}).get("implementation_function")
        if isinstance(impl, str) and impl:
            preferred.append(impl)
    preferred.extend(["predict_from_answer", "predict_concentrations", "simulate_from_answer", "forward", "predict"])
    functions = []
    seen = set()
    for name in preferred:
        if name in seen:
            continue
        seen.add(name)
        fn = getattr(module, name, None)
        if callable(fn):
            functions.append((name, fn))
    if not functions:
        raise RuntimeError("no callable prediction function found")

    wells_df = pd.DataFrame.from_records(payload["wells"])
    times = np.asarray(payload["times"], dtype=float)
    expanded_df = _expanded_records(wells_df, times)
    config = payload.get("config", {})
    results = []
    for case in payload["cases"]:
        probe_answer = case["answer"]
        with open("answer.json", "w", encoding="utf-8") as f:
            json.dump(probe_answer, f)
        last_error = None
        for fn_name, fn in functions:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    call_style, vals = _call_candidates(fn, probe_answer, expanded_df, wells_df, times, config)
                results.append({
                    "name": case["name"],
                    "function": fn_name,
                    "call_style": call_style,
                    "values": [float(x) if math.isfinite(float(x)) else None for x in vals],
                })
                last_error = None
                break
            except Exception as exc:
                last_error = f"{fn_name}:{type(exc).__name__}:{str(exc)[:300]}"
        if last_error is not None:
            raise RuntimeError(f"{case['name']} failed: {last_error}")
    sys.stdout = orig_stdout
    print(json.dumps({"ok": True, "results": results}, allow_nan=False))
except Exception as exc:
    sys.stdout = orig_stdout
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:1200]}"}))
'''


def run_submitted_forward_model_probes(
    submission_dir: Path,
    case_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    model_path = submission_dir / "forward_model.py"
    if not model_path.exists():
        return {"ok": False, "error": "forward_model.py_missing"}

    with tempfile.TemporaryDirectory(prefix="borden_ade_probe_") as tmp:
        tmp_path = Path(tmp)
        public_names = [
            "public_problem_config.json",
            "public_wells.csv",
            "public_monitoring_data.csv",
            "public_source_prior.json",
            "borden_grid.npz",
            "answer_template.json",
        ]
        for name in public_names:
            src = case_dir / name
            if src.exists():
                shutil.copy2(src, tmp_path / name)

        for py_file in submission_dir.glob("*.py"):
            if py_file.name.startswith("."):
                continue
            shutil.copy2(py_file, tmp_path / py_file.name)

        if not (tmp_path / "forward_model.py").exists():
            shutil.copy2(model_path, tmp_path / "forward_model.py")

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        cmd = [sys.executable, "-c", _prediction_probe_runner_script()]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp_path,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=25,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "forward_model_probe_timeout"}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    try:
        result = json.loads(stdout.splitlines()[-1] if stdout else "{}")
    except Exception:
        return {
            "ok": False,
            "error": "forward_model_probe_non_json_output",
            "stdout_tail": stdout[-800:],
            "stderr_tail": stderr[-800:],
        }
    if stderr:
        result["stderr_tail"] = stderr[-800:]
    return result


def _ade_metric_band(value: float, cuts: list[float]) -> str:
    return metric_feedback_band(value, cuts)


def _score_ade_probe_arrays(reference: np.ndarray, predicted_raw: np.ndarray) -> tuple[float, dict[str, Any]]:
    pred_finite = np.isfinite(predicted_raw)
    finite_fraction = float(np.mean(pred_finite)) if predicted_raw.size else 0.0
    pred = np.nan_to_num(predicted_raw, nan=0.0, posinf=0.0, neginf=0.0)
    negative_fraction = float(np.mean(pred < -1e-9)) if pred.size else 1.0
    pred = np.maximum(pred, 0.0)
    ref = np.maximum(np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    mean_ref = max(float(np.mean(np.abs(ref))), 1e-8)
    rel_rmse = float(np.sqrt(np.mean((pred - ref) ** 2)) / mean_ref) if len(ref) else float("inf")
    log_rmse = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(ref)) ** 2))) if len(ref) else float("inf")
    ref_p90 = float(np.percentile(ref, 90)) if len(ref) else 0.0
    pred_p90 = float(np.percentile(pred, 90)) if len(pred) else 0.0
    eps = max(ref_p90 * 1e-4, 1e-9)
    scale_log_error = abs(math.log((pred_p90 + eps) / (ref_p90 + eps))) if ref_p90 > 0.0 else float("inf")
    active_threshold = max(float(np.percentile(ref, 75)) * 0.05, 1e-6) if len(ref) else 1e-6
    active_ref = ref > active_threshold
    active_pred = pred > active_threshold
    active_pattern_error = float(np.mean(active_ref != active_pred)) if len(ref) else 1.0

    finite_score = 1.0 * finite_fraction * max(0.0, 1.0 - negative_fraction)
    log_score = log_power_score(log_rmse, good=0.05, bad=0.35, points=2.0, gamma=1.4)
    rel_score = log_power_score(rel_rmse, good=0.12, bad=1.0, points=4.0, gamma=1.3)
    scale_score = clipped_linear_score(scale_log_error, good=0.10, bad=0.75, points=2.0)
    active_score = clipped_linear_score(active_pattern_error, good=0.02, bad=0.25, points=1.0)

    score = finite_score + log_score + rel_score + scale_score + active_score
    detail = {
        "score": round3(score),
        "components": {
            "finite_nonnegative": round3(finite_score),
            "log_shape": round3(log_score),
            "relative_rmse": round3(rel_score),
            "scale": round3(scale_score),
            "arrival_activity": round3(active_score),
        },
        "metrics": {
            "relative_rmse": round3(rel_rmse),
            "log_rmse": round3(log_rmse),
            "scale_log_error": round3(scale_log_error),
            "active_pattern_error": round3(active_pattern_error),
            "finite_fraction": round3(finite_fraction),
            "negative_fraction": round3(negative_fraction),
        },
    }
    return min(score, ADE_CORRECTNESS_MAX_SCORE), detail


def score_ade_correctness_gate(
    answer: dict[str, Any],
    submission_dir: Path,
    case_dir: Path,
    config: dict[str, Any],
    scoring_dir: Path,
) -> tuple[float, dict[str, Any]]:
    """
    Dynamic ADE gate: run the submitted public forward model on public synthetic
    probes and compare it with the judge ADE response. Hidden observations and
    true source parameters are not passed to the submitted code.
    """
    detail: dict[str, Any] = {
        "status": "failed",
        "score": 0.0,
        "max_score": ADE_CORRECTNESS_MAX_SCORE,
        "gate_threshold": ADE_CORRECTNESS_GATE_SCORE,
        "probe_set": "public_synthetic_and_submitted_answer",
        "failures": [],
        "components": {},
        "probe_feedback": [],
    }
    if not (submission_dir / "forward_model.py").exists():
        detail["failures"].append("forward_model.py_missing")
        return 0.0, detail

    try:
        sys.path.insert(0, str(scoring_dir))
        from hidden_forward_model import simulate_from_answer  # type: ignore

        wells, times = make_ade_probe_wells_times(case_dir)
        cases = make_ade_probe_cases(answer, config)
        if not cases:
            detail["failures"].append("no_valid_probe_cases")
            return 0.0, detail

        reference_by_case: dict[str, np.ndarray] = {}
        for case in cases:
            ref_df = simulate_from_answer(case["answer"], wells, np.asarray(times, dtype=float), config)
            pred_col = find_pred_concentration_column(ref_df)
            reference_by_case[case["name"]] = ref_df[pred_col].to_numpy(dtype=float)
    except Exception as exc:
        detail["failures"].append(f"reference_probe_generation_failed:{type(exc).__name__}:{str(exc)[:200]}")
        return 0.0, detail

    payload = {
        "cases": cases,
        "wells": wells.to_dict(orient="records"),
        "times": [float(x) for x in np.asarray(times, dtype=float)],
        "config": config,
    }
    agent_result = run_submitted_forward_model_probes(submission_dir, case_dir, payload)
    detail["runner"] = {
        "ok": bool(agent_result.get("ok")),
        "stderr_tail": agent_result.get("stderr_tail"),
    }
    if not agent_result.get("ok"):
        detail["failures"].append(str(agent_result.get("error", "forward_model_probe_failed")))
        return 0.0, detail

    case_scores = []
    component_totals: dict[str, list[float]] = {}
    metric_totals: dict[str, list[float]] = {}
    for result in agent_result.get("results", []):
        name = str(result.get("name"))
        if name not in reference_by_case:
            continue
        raw_values = np.asarray([
            safe_float(x, float("nan")) for x in result.get("values", [])
        ], dtype=float)
        ref = reference_by_case[name]
        if raw_values.size != ref.size:
            detail["failures"].append(f"{name}:prediction_length_mismatch")
            continue
        score, case_detail = _score_ade_probe_arrays(ref, raw_values)
        case_scores.append(score)
        for comp_name, comp_value in case_detail.get("components", {}).items():
            component_totals.setdefault(comp_name, []).append(float(comp_value))
        for metric_name, metric_value in case_detail.get("metrics", {}).items():
            metric_totals.setdefault(metric_name, []).append(float(metric_value))
        m = case_detail["metrics"]
        detail["probe_feedback"].append({
            "name": name,
            "score": round3(score),
            "function": result.get("function"),
            "call_style": result.get("call_style"),
            "relative_rmse_band": _ade_metric_band(safe_float(m.get("relative_rmse"), float("inf")), [0.35, 0.75, 1.5, 3.0]),
            "log_rmse_band": _ade_metric_band(safe_float(m.get("log_rmse"), float("inf")), [0.12, 0.25, 0.55, 1.05]),
            "scale_band": _ade_metric_band(safe_float(m.get("scale_log_error"), float("inf")), [0.25, 0.6, 1.2, 2.0]),
            "activity_band": _ade_metric_band(safe_float(m.get("active_pattern_error"), float("inf")), [0.05, 0.15, 0.35, 0.60]),
        })

    if not case_scores:
        detail["failures"].append("no_probe_predictions_scored")
        return 0.0, detail

    score = float(np.mean(case_scores))
    detail["score"] = round3(score)
    detail["components"] = {
        name: round3(float(np.mean(values)))
        for name, values in sorted(component_totals.items())
    }
    detail["aggregate_metrics"] = {
        name: round3(float(np.mean(values)))
        for name, values in sorted(metric_totals.items())
    }
    component_caps = {
        "finite_nonnegative": 1.0,
        "log_shape": 2.0,
        "relative_rmse": 4.0,
        "scale": 2.0,
        "arrival_activity": 1.0,
    }
    weak_components = sorted(
        detail["components"],
        key=lambda name: safe_float(detail["components"].get(name), 0.0) / max(component_caps.get(name, 1.0), 1e-8),
    )
    detail["weakest_component"] = weak_components[0] if weak_components else "unknown"
    if score >= ADE_CORRECTNESS_GATE_SCORE:
        detail["status"] = "passed"
        detail["passed_gate"] = True
    else:
        detail["status"] = "failed"
        detail["passed_gate"] = False
        detail["failures"].append("ade_correctness_gate_failed")
    return min(score, ADE_CORRECTNESS_MAX_SCORE), detail


def score_public_sanity(answer: dict[str, Any], case_dir: Path, config: dict[str, Any], scoring_dir: Path) -> tuple[int, dict[str, Any]]:
    """
    Public observations are noisy and censored. This sanity score rewards
    plausible public predictions without letting public-only fitting dominate.
    """
    public_obs_path = case_dir / "public_monitoring_data.csv"
    public_wells_path = case_dir / "public_wells.csv"
    if not public_obs_path.exists() or not public_wells_path.exists():
        return 0, {}

    try:
        sys.path.insert(0, str(scoring_dir))
        from hidden_forward_model import simulate_from_answer  # type: ignore

        public_obs = pd.read_csv(public_obs_path)
        public_wells = pd.read_csv(public_wells_path)
        times = sorted(public_obs["time_days"].astype(float).unique())
        pred = simulate_from_answer(answer, public_wells, np.asarray(times, dtype=float), config)
        m, _ = compute_censored_public_metrics(public_obs, pred)
        score = clipped_linear_score(m.get("censored_log_rmse", float("inf")), good=0.035, bad=0.55, points=10)
        return float(score), m
    except Exception:
        return 0, {}


def mass_proxy(vals: dict[str, float]) -> float:
    volume = (
        2.0 * vals["half_length_x"]
        * 2.0 * vals["half_length_y"]
        * 2.0 * vals["half_length_z"]
    )
    return max(vals["C0"], 0.0) * max(volume, 0.0) * max(vals["duration"], 0.0)


def log_ratio_error(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0 or not math.isfinite(a) or not math.isfinite(b):
        return float("inf")
    return abs(math.log(a / b))


def score_region_physics(
    answer: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
    scoring_dir: Path,
    physical_detail: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Region/physics score, total 15 points. This checks geometry and source
    consistency in addition to prediction quality, so an unphysical surrogate
    that happens to fit monitoring curves cannot collect the physics credit.
    """
    detail: dict[str, Any] = {
        "gated": False,
        "physical_constraints": physical_detail,
        "components": {},
    }
    hidden_rrmse = safe_float(metrics.get("hidden_well", {}).get("rRMSE"), float("inf"))
    future_rrmse = safe_float(metrics.get("future_public", {}).get("rRMSE"), float("inf"))
    if hidden_rrmse >= 0.85 or future_rrmse >= 0.90:
        detail["prediction_support"] = "weak"
        detail["prediction_support_reason"] = "hidden_or_future_prediction_too_weak"
        detail["gated"] = True
        detail["gate_reason"] = "prediction_too_weak_for_region_physics"
        return 0.0, detail
    else:
        detail["prediction_support"] = "adequate"

    vals, failures = source_values(answer)
    if failures:
        detail["failures"] = failures
        return 0.0, detail

    score = 0.0
    if physical_detail.get("status") == "passed":
        score += 5.0
        detail["components"]["physical_extent"] = 5.0
    else:
        detail["components"]["physical_extent"] = 0.0

    truth_path = scoring_dir / "hidden_true_region_source.json"
    if truth_path.exists():
        try:
            truth = load_json(truth_path)
            true_vals, true_failures = source_values(truth)
            if true_failures:
                raise ValueError(",".join(true_failures))

            xy_err = math.hypot(vals["x_center"] - true_vals["x_center"], vals["y_center"] - true_vals["y_center"])
            z_err = abs(vals["z_center"] - true_vals["z_center"])
            shape_err = float(np.sqrt(np.mean([
                log_ratio_error(vals["half_length_x"], true_vals["half_length_x"]) ** 2,
                log_ratio_error(vals["half_length_y"], true_vals["half_length_y"]) ** 2,
                log_ratio_error(vals["half_length_z"], true_vals["half_length_z"]) ** 2,
            ])))
            timing_err = math.sqrt(
                ((vals["t_start"] - true_vals["t_start"]) / 730.0) ** 2
                + ((vals["duration"] - true_vals["duration"]) / 2555.0) ** 2
            )
            mass_err = log_ratio_error(mass_proxy(vals), mass_proxy(true_vals))

            center_score = clipped_linear_score(xy_err, good=8.0, bad=80.0, points=2.0)
            z_score = clipped_linear_score(z_err, good=0.25, bad=2.5, points=1.5)
            shape_score = clipped_linear_score(shape_err, good=0.20, bad=1.00, points=3.0)
            timing_score = clipped_linear_score(timing_err, good=0.15, bad=1.00, points=2.0)
            mass_score = clipped_linear_score(mass_err, good=0.25, bad=1.50, points=1.5)
            score += center_score + z_score + shape_score + timing_score + mass_score
            detail["components"].update({
                "center_score": round3(center_score),
                "z_score": round3(z_score),
                "shape_score": round3(shape_score),
                "timing_score": round3(timing_score),
                "mass_score": round3(mass_score),
            })
            detail["errors"] = {
                "xy_center_error_m": round3(xy_err),
                "z_center_error_m": round3(z_err),
                "shape_log_rms": round3(shape_err),
                "timing_scaled_rms": round3(timing_err),
                "mass_log_error": round3(mass_err),
            }
        except Exception as exc:
            detail["truth_comparison_error"] = repr(exc)
    else:
        # Fallback for development bundles without hidden truth.
        positives = [
            vals["half_length_x"] > 0,
            vals["half_length_y"] > 0,
            vals["half_length_z"] > 0,
            vals["C0"] > 0,
            vals["duration"] > 0,
            vals["t_start"] >= 0,
        ]
        fallback = 10.0 * sum(positives) / len(positives)
        score += fallback
        detail["components"]["fallback_positive_parameter_score"] = round3(fallback)

    if hidden_rrmse >= 0.20 or future_rrmse >= 0.25:
        detail["prediction_support_reason"] = "prediction_weak_caps_region_physics"
        score = min(score, 5.0)

    return min(float(score), 15.0), detail


def physical_cap(physical_detail: dict[str, Any], region_physics_score: float) -> tuple[float | None, str | None]:
    failures = set(physical_detail.get("failures", []) or [])
    if failures & {
        "source_region_outside_model_domain",
        "source_region_outside_saturated_vertical_extent",
        "nonpositive_region_half_length",
        "nonpositive_source_concentration",
        "invalid_release_time_window",
    }:
        return 30.0, "cap30_physical_extent_violation"
    if "source_region_outside_original_source_zone" in failures:
        return 45.0, "cap45_source_prior_extent_violation"
    if region_physics_score < 12.0:
        return 15.0, "cap15_weak_region_physics_consistency"
    if region_physics_score < 14.0:
        return 30.0, "cap30_imperfect_region_physics_consistency"
    return None, None


def metric_feedback_band(error: float, cuts: list[float]) -> str:
    error = safe_float(error, float("inf"))
    if error <= cuts[0]:
        return "excellent"
    if error <= cuts[1]:
        return "good"
    if error <= cuts[2]:
        return "moderate"
    if error <= cuts[3]:
        return "weak"
    return "poor"


def progress_band(score: float, max_score: float) -> str:
    score = safe_float(score, 0.0)
    max_score = max(safe_float(max_score, 0.0), 1e-8)
    ratio = max(0.0, min(1.0, score / max_score))
    if ratio >= 0.90:
        return "strong"
    if ratio >= 0.65:
        return "good"
    if ratio >= 0.35:
        return "partial"
    if ratio > 0.0:
        return "weak"
    return "missing"


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _json_load_optional(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _keyword_group_hits(text: str, groups: list[list[str]]) -> int:
    lower = text.lower()
    return sum(1 for group in groups if any(k in lower for k in group))


def _row_count_csv(path: Path) -> int:
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _review_band(ratio: float) -> str:
    ratio = max(0.0, min(1.0, safe_float(ratio, 0.0)))
    if ratio >= 0.85:
        return "strong"
    if ratio >= 0.65:
        return "adequate"
    if ratio >= 0.35:
        return "partial"
    if ratio > 0.0:
        return "weak"
    return "missing"


def _stage_result(score: float, max_score: float, status: str | None = None, notes: list[str] | None = None) -> dict[str, Any]:
    score = max(0.0, min(float(max_score), float(score)))
    ratio = 0.0 if max_score <= 0 else score / float(max_score)
    return {
        "score": round3(score),
        "max_score": round3(max_score),
        "status": status or _review_band(ratio),
        "notes": notes or [],
    }


def _public_residual_review(public_metrics: dict[str, Any], public_score: float) -> str:
    if not public_metrics:
        return "Public residual review is unavailable because no public prediction comparison was produced."
    censored = safe_float(public_metrics.get("censored_log_rmse"), float("inf"))
    above = safe_float(public_metrics.get("above_log_rmse"), float("inf"))
    excess = safe_float(public_metrics.get("below_excess_rate"), float("inf"))
    notes: list[str] = []
    if public_score >= 1.6:
        notes.append("Main public breakthrough curves are broadly aligned")
    elif public_score >= 0.7:
        notes.append("Public breakthrough fit is partially reasonable")
    else:
        notes.append("Public breakthrough fit remains weak")
    if above > 0.20:
        notes.append("detected concentration magnitudes still need calibration")
    if censored > 0.25:
        notes.append("arrival or tail timing residuals remain material")
    if excess > 0.12:
        notes.append("nondetect handling remains incomplete")
    return "; ".join(notes) + "."


def _validation_review(validation_status: str) -> str:
    return {
        "defensible": "Withheld predictive review is defensible under the submitted model assumptions.",
        "acceptable": "Withheld predictive review is acceptable but still has material uncertainty.",
        "provisional": "Withheld predictive review is provisional; keep alternative source hypotheses open.",
        "not_defensible": "Withheld predictive review is not defensible; do not treat the current source estimate as a validated conclusion.",
        "not_reviewed": "Withheld predictive review was not performed because the submission is not yet structurally or physically ready.",
        "not_run": "Withheld predictive review was not run because the submission is structurally incomplete.",
    }.get(validation_status, "Withheld predictive review is not defensible; do not treat the current source estimate as a validated conclusion.")


def _validation_status(withheld_score: float, prediction_cap_reason: str) -> str:
    if prediction_cap_reason.startswith("cap15"):
        return "not_defensible"
    if withheld_score >= 16.0:
        return "defensible"
    if withheld_score >= 12.0:
        return "acceptable"
    if withheld_score >= 6.0:
        return "provisional"
    return "not_defensible"


def _process_stage(stage_scores: dict[str, Any]) -> str:
    names = []
    if safe_float(stage_scores.get("data_understanding", {}).get("score"), 0.0) >= 5.0:
        names.append("data")
    if safe_float(stage_scores.get("forward_model", {}).get("score"), 0.0) >= 8.0:
        names.append("forward")
    if safe_float(stage_scores.get("inversion_framework", {}).get("score"), 0.0) >= 5.0:
        names.append("baseline")
    if safe_float(stage_scores.get("optimization_evidence", {}).get("score"), 0.0) >= 7.0:
        names.append("diagnostics")
    if safe_float(stage_scores.get("report_quality", {}).get("score"), 0.0) >= 5.0:
        names.append("final_package")
    return "+".join(names) if names else "unstarted"


def score_realistic_project_stages(
    detail: dict[str, Any],
    submission_dir: Path,
    case_dir: Path,
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Internal project-style score aligned with real source-inversion work."""
    stage_scores: dict[str, Any] = {}

    public_data_ok = 0.0
    public_notes: list[str] = []
    for fname, required_cols in [
        ("public_problem_config.json", []),
        ("public_wells.csv", ["well_id", "x", "y", "z"]),
        ("public_monitoring_data.csv", ["well_id", "time_days"]),
    ]:
        path = case_dir / fname
        if path.exists():
            public_data_ok += 1.0
            if required_cols:
                try:
                    df = pd.read_csv(path)
                    if all(c in df.columns for c in required_cols):
                        public_data_ok += 0.35
                except Exception:
                    public_notes.append(f"{fname} could not be read")
        else:
            public_notes.append(f"{fname} missing")
    audit_text = _read_optional_text(submission_dir / "data_audit.md")
    audit_json = _json_load_optional(submission_dir / "data_audit.json")
    audit_groups = [
        ["coordinate", "坐标", "x", "y", "z"],
        ["unit", "单位", "mg/l", "day", "meter", "metre"],
        ["well", "monitoring", "井"],
        ["detection", "censor", "nondetect", "检出", "检测限"],
        ["breakthrough", "peak", "arrival", "峰值", "到达"],
        ["time", "align", "时序", "对齐"],
    ]
    data_score = min(4.0, public_data_ok)
    if audit_text:
        data_score += min(7.0, 1.0 + _keyword_group_hits(audit_text, audit_groups))
    else:
        public_notes.append("data_audit.md missing")
    if isinstance(audit_json, dict):
        data_score += min(4.0, 1.0 + len(set(audit_json.keys()) & {"wells", "times", "detections", "units", "summary"}))
    else:
        public_notes.append("data_audit.json missing or unreadable")
    stage_scores["data_understanding"] = _stage_result(data_score, 15.0, notes=public_notes[:4])

    forward_notes: list[str] = []
    validation_rows = _row_count_csv(submission_dir / "forward_validation.csv")
    forward_validation_text = _read_optional_text(submission_dir / "forward_validation.md")
    forward_score = 0.0
    if detail.get("model_gate_status") == "passed":
        forward_score += 4.0
    else:
        forward_notes.append("forward_model.py interface or ADE coverage needs review")
    forward_score += 10.0 * normalized_score(detail.get("ade_correctness_score", 0.0), ADE_CORRECTNESS_MAX_SCORE)
    forward_score += 3.0 * normalized_score(detail.get("transport_equation_score", 0.0), 2.0)
    if validation_rows > 0:
        forward_score += min(2.0, validation_rows / 5.0)
    else:
        forward_notes.append("forward_validation.csv missing")
    if forward_validation_text and _keyword_group_hits(forward_validation_text, [["ade", "advection", "dispersion"], ["release", "duration"], ["well", "time"]]) >= 2:
        forward_score += 1.0
    elif not forward_validation_text:
        forward_notes.append("forward_validation.md missing")
    stage_scores["forward_model"] = _stage_result(forward_score, 20.0, notes=forward_notes[:4])

    inversion_notes: list[str] = []
    inv_score = 4.0 * (
        0.5 * normalized_score(detail.get("format_score", 0.0), 1.0)
        + 0.5 * normalized_score(detail.get("bounds_score", 0.0), 1.0)
    )
    if (submission_dir / "inverse_solver.py").exists():
        inv_score += 3.0
    else:
        inversion_notes.append("inverse_solver.py missing")
    inv_cfg = _json_load_optional(submission_dir / "inversion_config.json")
    obj_report = _read_optional_text(submission_dir / "objective_report.md")
    if isinstance(inv_cfg, dict):
        inv_score += min(4.0, 1.0 + len(set(inv_cfg.keys()) & {"parameters", "bounds", "objective", "weights", "optimizer"}))
    else:
        inversion_notes.append("inversion_config.json missing or unreadable")
    obj_groups = [["objective", "loss", "误差"], ["bound", "prior", "边界"], ["censor", "nondetect", "检测限"], ["log", "relative", "rmse"], ["baseline", "initial"]]
    if obj_report:
        inv_score += min(4.0, _keyword_group_hits(obj_report, obj_groups))
    else:
        inversion_notes.append("objective_report.md missing")
    stage_scores["inversion_framework"] = _stage_result(inv_score, 15.0, notes=inversion_notes[:4])

    opt_notes: list[str] = []
    opt_score = 4.0 * normalized_score(detail.get("public_sanity_score_before_ade_gate", 0.0), 2.0)
    trace_rows = _row_count_csv(submission_dir / "optimization_trace.csv")
    if trace_rows > 0:
        opt_score += min(4.0, 1.0 + trace_rows / 10.0)
    else:
        opt_notes.append("optimization_trace.csv missing")
    candidates = _json_load_optional(submission_dir / "candidate_sources.json")
    if isinstance(candidates, list):
        opt_score += min(4.0, len(candidates) / 2.0)
    elif isinstance(candidates, dict) and isinstance(candidates.get("candidates"), list):
        opt_score += min(4.0, len(candidates["candidates"]) / 2.0)
    else:
        opt_notes.append("candidate_sources.json missing or not a candidate list")
    residual_text = _read_optional_text(submission_dir / "residual_analysis.md")
    fit_rows = _row_count_csv(submission_dir / "fit_diagnostics.csv")
    residual_groups = [["residual", "残差"], ["well", "w01", "w02", "w03"], ["tail", "arrival", "peak"], ["nondetect", "censor", "检测限"], ["geometry", "source", "uncertainty"]]
    residual_credit = _keyword_group_hits(residual_text, residual_groups) if residual_text else 0
    if fit_rows > 0:
        residual_credit += 1
    opt_score += min(5.0, residual_credit)
    if not residual_text:
        opt_notes.append("residual_analysis.md missing")
    method_text = " ".join([
        _read_optional_text(submission_dir / "inverse_solver.py"),
        str(answer.get("method", "")),
    ]).lower()
    if any(k in method_text for k in ["differential", "evolution", "multi-start", "multistart", "nelder", "least_squares", "local search"]):
        opt_score += 3.0
    else:
        opt_notes.append("optimizer evidence is weak")
    stage_scores["optimization_evidence"] = _stage_result(opt_score, 20.0, notes=opt_notes[:4])

    report_notes: list[str] = []
    report_score = 0.0
    if detail.get("format_score", 0.0) > 0 and detail.get("bounds_score", 0.0) > 0:
        report_score += 2.0
    if (submission_dir / "public_predictions.csv").exists():
        report_score += 1.0
    else:
        report_notes.append("public_predictions.csv missing")
    report_text = _read_optional_text(submission_dir / "report.md")
    report_groups = [["source", "region", "源"], ["public", "fit", "residual"], ["uncertainty", "nonunique", "confidence"], ["reproduce", "command", "workflow"], ["figure", "plot", "table"]]
    if report_text:
        report_score += min(4.0, 0.5 + _keyword_group_hits(report_text, report_groups))
    else:
        report_notes.append("report.md missing")
    uncertainty = _json_load_optional(submission_dir / "uncertainty_report.json")
    if isinstance(uncertainty, dict):
        report_score += min(2.0, 0.5 + len(set(uncertainty.keys()) & {"source_ensemble", "credible_region", "parameter_ranges", "limitations"}) * 0.5)
    else:
        report_notes.append("uncertainty_report.json missing or unreadable")
    if report_text and any(k in report_text.lower() for k in ["python", "sebench-submit", "inverse_solver", "command"]):
        report_score += 1.0
    stage_scores["report_quality"] = _stage_result(report_score, 10.0, notes=report_notes[:4])

    metric_scores = detail.get("metric_score_detail", {}) or {}
    hidden_quality = (
        0.75 * normalized_score(metric_scores.get("hidden_rrmse_score", 0.0), 38.0)
        + 0.25 * normalized_score(metric_scores.get("hidden_log_score", 0.0), 12.0)
    )
    future_quality = (
        0.75 * normalized_score(metric_scores.get("future_rrmse_score", 0.0), 30.0)
        + 0.25 * normalized_score(metric_scores.get("future_log_score", 0.0), 12.0)
    )
    log_quality = 0.5 * normalized_score(metric_scores.get("hidden_log_score", 0.0), 12.0) + 0.5 * normalized_score(metric_scores.get("future_log_score", 0.0), 12.0)
    physics_quality = normalized_score(detail.get("region_physics_score", 0.0), 15.0)
    withheld_score = 8.0 * hidden_quality + 6.0 * future_quality + 3.0 * log_quality + 3.0 * physics_quality
    stage_scores["withheld_prediction"] = _stage_result(withheld_score, 20.0)

    raw_total = sum(safe_float(v.get("score"), 0.0) for v in stage_scores.values())
    caps: list[float] = []
    cap_reasons: list[str] = []
    if detail.get("format_score", 0.0) <= 0 or detail.get("bounds_score", 0.0) <= 0:
        caps.append(20.0)
        cap_reasons.append("answer_format_or_bounds_invalid")
    if detail.get("model_gate_status") != "passed":
        caps.append(25.0)
        cap_reasons.append("forward_model_not_usable")
    prediction_cap_value, prediction_cap_reason = prediction_quality_cap(detail)
    if prediction_cap_reason.startswith("cap15"):
        caps.append(65.0)
        cap_reasons.append("withheld_prediction_not_defensible")
    elif prediction_cap_reason.startswith("cap30") or prediction_cap_reason.startswith("cap50"):
        caps.append(80.0)
        cap_reasons.append("withheld_prediction_weak")
    if safe_float(stage_scores["optimization_evidence"]["score"], 0.0) < 5.0:
        caps.append(70.0)
        cap_reasons.append("optimization_evidence_insufficient")
    if safe_float(stage_scores["report_quality"]["score"], 0.0) < 3.0:
        caps.append(85.0)
        cap_reasons.append("report_or_uncertainty_incomplete")

    final_total = min(raw_total, min(caps) if caps else PROCESS_SCORE_MAX, PROCESS_SCORE_MAX)
    validation_status = _validation_status(withheld_score, prediction_cap_reason)
    return {
        "stage_scores": stage_scores,
        "raw_total_score": round3(raw_total),
        "total_score": round3(final_total),
        "caps": caps,
        "cap_reasons": cap_reasons,
        "process_stage": _process_stage(stage_scores),
        "validation_status": validation_status,
        "public_residual_review": _public_residual_review(detail.get("public_metrics", {}), detail.get("public_sanity_score_before_ade_gate", 0.0)),
        "validation_review": _validation_review(validation_status),
    }


def project_review_state(detail: dict[str, Any]) -> dict[str, Any]:
    review = detail.get("project_artifact_review", {}) or {}
    process_scores = detail.get("process_scores", {}) or {}
    process_components = process_scores.get("components", {}) or {}
    stage_scores = review.get("stage_scores", {}) or {}
    data_status = stage_scores.get("data_understanding", {}).get("status", "missing")
    inversion_status = stage_scores.get("inversion_framework", {}).get("status", "missing")
    optimization_status = stage_scores.get("optimization_evidence", {}).get("status", "missing")
    report_status = stage_scores.get("report_quality", {}).get("status", "missing")

    model_status_raw = str(detail.get("model_gate_status", "unknown") or "unknown")
    ade_status_raw = str(detail.get("ade_correctness_gate_status", "unknown") or "unknown")
    ade_score = safe_float(detail.get("ade_correctness_score"), 0.0)
    if model_status_raw != "passed":
        forward_status = "missing" if "forward_model.py_missing" in str(detail.get("model_relevance_detail", {})) else "weak"
        model_status = "missing" if forward_status == "missing" else "failed"
    elif ade_score < 5.0:
        forward_status = "weak"
        model_status = "needs_rebuild"
    elif ade_status_raw != "passed":
        forward_status = "partial"
        model_status = "needs_review"
    elif ade_score < 9.0:
        forward_status = "adequate"
        model_status = "acceptable"
    else:
        forward_status = "strong"
        model_status = "strong"

    public_score = safe_float(detail.get("public_sanity_score_before_ade_gate"), 0.0)
    if model_status in {"missing", "failed", "needs_rebuild"}:
        public_fit = "not_reviewed"
    elif public_score >= 1.6:
        public_fit = "adequate"
    elif public_score >= 0.7:
        public_fit = "partial"
    elif public_score > 0.0:
        public_fit = "weak"
    else:
        public_fit = "missing"

    prediction_cap_reason = str(detail.get("prediction_quality_cap_reason", "") or "")
    if model_status in {"missing", "failed", "needs_rebuild"}:
        validation_status = "not_reviewed"
    elif prediction_cap_reason.startswith("cap15"):
        validation_status = "not_defensible"
    elif prediction_cap_reason.startswith("cap30"):
        validation_status = "provisional"
    elif prediction_cap_reason.startswith("cap50"):
        validation_status = "provisional"
    elif prediction_cap_reason.startswith("cap70"):
        validation_status = "acceptable"
    elif prediction_cap_reason == "none":
        validation_status = "defensible"
    else:
        validation_status = "not_defensible"

    if detail.get("errors"):
        review_status = "invalid"
    elif model_status in {"missing", "failed", "needs_rebuild"}:
        review_status = "needs_revision"
    elif validation_status == "defensible" and report_status in {"adequate", "strong"}:
        review_status = "defensible"
    elif validation_status in {"acceptable", "defensible"}:
        review_status = "acceptable"
    elif validation_status == "provisional":
        review_status = "provisional"
    else:
        review_status = "needs_revision"

    hidden_fit = safe_float(process_components.get("hidden_fit"), 0.0)
    future_fit = safe_float(process_components.get("future_fit"), 0.0)
    source_physics = safe_float(process_components.get("source_physics"), 0.0)
    if detail.get("format_score", 0.0) <= 0 or detail.get("bounds_score", 0.0) <= 0:
        process_stage = "setup"
    elif model_status in {"missing", "failed"}:
        process_stage = "data+setup"
    elif model_status in {"needs_rebuild", "needs_review"}:
        process_stage = "data+forward"
    elif public_fit in {"missing", "weak", "partial"}:
        process_stage = "data+forward+public_calibration"
    elif hidden_fit < 8.0 and future_fit < 8.0:
        process_stage = "data+forward+baseline_inversion"
    elif source_physics < 4.0:
        process_stage = "data+forward+inversion+diagnostics"
    else:
        process_stage = "final_package"

    next_review = "Improve data audit, public residual diagnosis, and source-geometry uncertainty."
    if detail.get("format_score", 0.0) <= 0 or detail.get("bounds_score", 0.0) <= 0:
        next_review = "Fix the source schema and keep all source parameters within the public search bounds before model review."
    elif model_status in {"missing", "failed"}:
        next_review = "Submit a runnable forward_model.py coupled to answer.json before interpreting inversion results."
    elif model_status in {"needs_rebuild", "needs_review"}:
        next_review = "Rebuild and validate the finite-region ADE forward model before further source-parameter tuning."
    elif data_status in {"missing", "weak"}:
        next_review = "Complete data audit: coordinates, units, detection limits, and breakthrough timing."
    elif public_fit in {"missing", "weak", "partial"}:
        next_review = "Calibrate against public breakthrough curves and document residuals before using withheld predictive review."
    elif inversion_status in {"missing", "weak", "partial"}:
        next_review = "Document inversion parameterization, bounds, objective function, and baseline solver behavior."
    elif optimization_status in {"missing", "weak", "partial"}:
        next_review = "Add optimization traces, candidate comparison, and public residual diagnostics."
    elif report_status in {"missing", "weak", "partial"}:
        next_review = "Strengthen final report, uncertainty discussion, and reproducibility notes."
    elif hidden_fit < future_fit and validation_status in {"not_defensible", "provisional"}:
        next_review = "Use public residual patterns to revisit source location and rectangular geometry uncertainty."
    elif validation_status in {"not_defensible", "provisional"}:
        next_review = "Review release timing, duration, and plume-tail behavior using public time-series residuals."
    elif validation_status in {"not_defensible", "provisional"}:
        next_review = "Use public residuals and uncertainty analysis to revise source geometry and release history."

    return {
        "review_status": review_status,
        "process_stage": process_stage,
        "model_status": model_status,
        "public_fit": public_fit,
        "validation_status": validation_status,
        "next_review": next_review,
        "data_understanding": data_status,
        "forward_model": forward_status,
        "inversion_framework": inversion_status,
        "optimization_evidence": optimization_status,
        "report_quality": report_status,
        "public_residual_review": _public_residual_review(detail.get("public_metrics", {}), public_score),
        "validation_review": _validation_review(validation_status),
    }


REVIEW_STATUS_TEXT = {
    "invalid": "This submission cannot be reviewed yet because required files, schema, or runnable model components are missing or invalid.",
    "needs_revision": "This submission is not yet a defensible source-inversion result and needs revision before it can support a project conclusion.",
    "provisional": "This submission provides a provisional source hypothesis, but the model evidence is still insufficient for a final conclusion.",
    "acceptable": "This submission is broadly acceptable for interim technical review, but uncertainty and validation evidence still need attention.",
    "defensible": "This submission is technically defensible as a source-inversion result, subject to the documented assumptions and uncertainty.",
}

PROCESS_STAGE_TEXT = {
    "setup": "The submission is still in the setup stage; fix the source schema, required files, and parameter bounds before model review.",
    "data+setup": "The public data are available, but the submission has not yet formed a runnable modeling workflow.",
    "data+forward": "The current work is mainly at the forward-modeling stage; the finite-region ADE model must be made reliable before inversion.",
    "data+forward+public_calibration": "The forward model is usable, but public monitoring curves still need calibration before the inverse result is meaningful.",
    "data+forward+baseline_inversion": "The workflow has reached baseline inversion; the next issue is whether the source hypothesis generalizes beyond the public fit.",
    "data+forward+inversion+diagnostics": "The workflow has an inversion result, but it still needs residual diagnosis, candidate-source comparison, and uncertainty analysis.",
    "final_package": "The submission has reached the final-package stage; review reproducibility, uncertainty, and project defensibility.",
    "unstarted": "The workflow has not yet provided enough evidence to identify a source-inversion stage.",
}

MODEL_STATUS_TEXT = {
    "missing": "No usable forward_model.py was found; submit a runnable forward model before interpreting any inversion result.",
    "failed": "The submitted forward model exists but fails interface, execution, or coupling checks and cannot support inversion yet.",
    "needs_rebuild": "The forward model does not yet reproduce the finite-region ADE behavior well enough; rebuild the transport model before tuning source parameters.",
    "needs_review": "The forward model is runnable, but its ADE response still needs review before it can be used as a reliable inversion engine.",
    "acceptable": "The forward model is acceptable for inversion, but validation records and edge-case checks should still be documented.",
    "strong": "The forward ADE model is reliable; further progress should focus on inversion evidence, residual diagnosis, and uncertainty.",
}

PUBLIC_FIT_TEXT = {
    "not_reviewed": "Public-fit review is not meaningful yet because the forward model is not sufficiently reliable.",
    "missing": "No meaningful public monitoring fit was produced; generate predictions for the public wells and compare them with the observed time series.",
    "weak": "The public monitoring fit is weak; first align the main breakthrough curves and handle nondetect observations consistently.",
    "partial": "The public monitoring fit is partially reasonable, but peak timing, plume tail, or nondetect residuals still need diagnosis.",
    "adequate": "The public monitoring fit is adequate for baseline inversion; the next focus should be source non-uniqueness and predictive robustness.",
}

VALIDATION_STATUS_TEXT = {
    "not_reviewed": "Withheld predictive review was not performed because the submission is not yet structurally or physically ready.",
    "not_defensible": "The source hypothesis is not yet defensible under withheld predictive review; treat it as an unvalidated working hypothesis.",
    "provisional": "The source hypothesis has limited predictive support, but alternative source geometries and release histories should remain open.",
    "acceptable": "The source hypothesis has acceptable predictive support, but uncertainty and sensitivity should still be documented.",
    "defensible": "The source hypothesis is defensible under withheld predictive review, given the submitted model assumptions and uncertainty.",
}

DATA_UNDERSTANDING_TEXT = {
    "missing": "No data-audit evidence was submitted; document coordinates, units, wells, time alignment, detection limits, and breakthrough summaries.",
    "weak": "The data audit is incomplete; coordinates, units, detection limits, well alignment, or breakthrough timing are not sufficiently documented.",
    "partial": "The data audit covers some essentials, but important checks such as nondetect handling, well-time alignment, or breakthrough summaries remain incomplete.",
    "adequate": "The data audit is adequate and supports the forward-modeling and inversion workflow.",
    "strong": "The data audit is strong and clearly documents coordinates, units, monitoring geometry, detection limits, breakthrough timing, and data-quality checks.",
}

FORWARD_MODEL_TEXT = {
    "missing": "No usable forward-model evidence was submitted.",
    "weak": "The forward-model evidence is weak and does not yet support reliable contaminant transport simulation.",
    "partial": "The forward model is partially supported, but release timing, source superposition, or ADE behavior still needs validation.",
    "adequate": "The forward model is adequately supported for inversion use.",
    "strong": "The forward model is strongly supported and can be treated as a reliable inversion component.",
}

INVERSION_FRAMEWORK_TEXT = {
    "missing": "No clear inversion framework was submitted; provide parameterization, bounds, objective function, and baseline behavior.",
    "weak": "The inversion framework is weak; parameterization, search bounds, objective function, or censoring treatment need clearer implementation.",
    "partial": "The inversion framework is partially developed, but search strategy, error metrics, or baseline comparison still need work.",
    "adequate": "The inversion framework is adequate for systematic source-parameter search.",
    "strong": "The inversion framework is strong and clearly documents parameterization, bounds, objective function, and baseline comparison.",
}

OPTIMIZATION_EVIDENCE_TEXT = {
    "missing": "No optimization evidence was submitted; provide traces, candidate sources, and residual diagnostics.",
    "weak": "Optimization evidence is weak; the current answer appears insufficiently supported by candidate comparison or residual diagnosis.",
    "partial": "Optimization evidence is partial; add clearer optimization traces, multi-start comparison, and public residual analysis.",
    "adequate": "Optimization evidence is adequate and supports the selected source hypothesis.",
    "strong": "Optimization evidence is strong, with clear search traces, candidate comparison, residual diagnosis, and uncertainty discussion.",
}

REPORT_QUALITY_TEXT = {
    "missing": "No adequate final report or uncertainty discussion was submitted.",
    "weak": "The report is weak; it does not yet explain the source hypothesis, residuals, uncertainty, limitations, and reproducibility.",
    "partial": "The report is partially useful, but uncertainty, non-uniqueness, residual interpretation, or reproducibility details remain incomplete.",
    "adequate": "The report is adequate for technical review.",
    "strong": "The report is strong and clearly explains the source hypothesis, evidence chain, residuals, uncertainty, limitations, and reproducibility.",
}


def project_review_feedback(detail: dict[str, Any]) -> dict[str, Any]:
    state = project_review_state(detail)
    return {
        "review_status": REVIEW_STATUS_TEXT.get(state["review_status"], REVIEW_STATUS_TEXT["needs_revision"]),
        "process_stage": PROCESS_STAGE_TEXT.get(state["process_stage"], PROCESS_STAGE_TEXT["unstarted"]),
        "model_status": MODEL_STATUS_TEXT.get(state["model_status"], MODEL_STATUS_TEXT["failed"]),
        "public_fit": PUBLIC_FIT_TEXT.get(state["public_fit"], PUBLIC_FIT_TEXT["missing"]),
        "validation_status": VALIDATION_STATUS_TEXT.get(state["validation_status"], VALIDATION_STATUS_TEXT["not_defensible"]),
        "next_review": state["next_review"],
        "data_understanding": DATA_UNDERSTANDING_TEXT.get(state["data_understanding"], DATA_UNDERSTANDING_TEXT["missing"]),
        "forward_model": FORWARD_MODEL_TEXT.get(state["forward_model"], FORWARD_MODEL_TEXT["missing"]),
        "inversion_framework": INVERSION_FRAMEWORK_TEXT.get(state["inversion_framework"], INVERSION_FRAMEWORK_TEXT["missing"]),
        "optimization_evidence": OPTIMIZATION_EVIDENCE_TEXT.get(state["optimization_evidence"], OPTIMIZATION_EVIDENCE_TEXT["missing"]),
        "report_quality": REPORT_QUALITY_TEXT.get(state["report_quality"], REPORT_QUALITY_TEXT["missing"]),
        "public_residual_review": state["public_residual_review"],
        "validation_review": state["validation_review"],
    }


def teacher_feedback(detail: dict[str, Any]) -> dict[str, Any]:
    """
    Structured learning feedback for agents.

    The feedback intentionally exposes bands, caps, and component scores rather
    than hidden observations, hidden true parameters, or pointwise residuals.
    This is enough to guide search while preserving the private evaluation set.
    """
    metrics = detail.get("metrics", {}) or {}
    hidden = metrics.get("hidden_well", {}) or {}
    future = metrics.get("future_public", {}) or {}
    metric_scores = detail.get("metric_score_detail", {}) or {}
    region_detail = detail.get("region_physics_detail", {}) or {}
    region_gate_reason = str(region_detail.get("gate_reason", "") or "")
    ade_detail = detail.get("ade_correctness_detail", {}) or {}
    ade_gate_status = str(detail.get("ade_correctness_gate_status", "unknown") or "unknown")

    hidden_rrmse = safe_float(hidden.get("rRMSE"), float("inf"))
    future_rrmse = safe_float(future.get("rRMSE"), float("inf"))
    hidden_log = safe_float(hidden.get("log_rmse"), float("inf"))
    future_log = safe_float(future.get("log_rmse"), float("inf"))

    component_specs = {
        "format": (detail.get("format_score", 0.0), 1.0),
        "bounds": (detail.get("bounds_score", 0.0), 1.0),
        "public_sanity": (detail.get("public_sanity_score", 0.0), 2.0),
        "method_workflow": (detail.get("method_workflow_score", 0.0), 1.0),
        "transport_equation": (detail.get("transport_equation_score", 0.0), 2.0),
        "model_relevance": (detail.get("model_relevance_score", 0.0), 9.0),
        "ade_correctness": (detail.get("ade_correctness_score", 0.0), ADE_CORRECTNESS_MAX_SCORE),
        "hidden_well_prediction": (metric_scores.get("hidden_rrmse_score", 0.0), 38.0),
        "future_time_extrapolation": (metric_scores.get("future_rrmse_score", 0.0), 30.0),
        "hidden_log_fit": (metric_scores.get("hidden_log_score", 0.0), 12.0),
        "future_log_fit": (metric_scores.get("future_log_score", 0.0), 12.0),
        "region_physics": (detail.get("region_physics_score", 0.0), 15.0),
    }

    component_scores: dict[str, float] = {}
    component_progress: dict[str, str] = {}
    component_ratios: dict[str, float] = {}
    for name, (score, max_score) in component_specs.items():
        s = max(0.0, safe_float(score, 0.0))
        m = max(safe_float(max_score, 0.0), 1e-8)
        component_scores[name] = round3(s)
        component_progress[name] = progress_band(s, m)
        component_ratios[name] = max(0.0, min(1.0, s / m))

    if detail.get("model_gate_status") != "passed":
        weakest = "model_relevance"
        hint = (
            "Forward-model interface or ADE coverage is the weakest upstream step. "
            "Use public_forward_model.py as the reference operator and run "
            "local_validate_forward_model.py before tuning answer.json."
        )
    elif ade_gate_status != "passed":
        weakest = "ade_correctness"
        ade_weak = str(ade_detail.get("weakest_component", "ADE response") or "ADE response")
        hint = (
            "The submitted forward_model.py is runnable but weak on public ADE probes; "
            f"improve {ade_weak}. The public reference exposes source normalization, "
            "release offsets, point3 advection/dispersion, and rectangular-region "
            "superposition; validate locally before parameter tuning."
        )
    elif detail.get("physical_cap_reason"):
        weakest = "region_physics"
        region_score = safe_float(detail.get("region_physics_score", 0.0), 0.0)
        if region_score >= 12.0:
            hint = (
                "Source physics is close, but physical consistency is still the weakest process component."
            )
        else:
            hint = "Improve physical source-region consistency while preserving monitoring prediction quality."
    else:
        priority = [
            "future_time_extrapolation",
            "hidden_well_prediction",
            "future_log_fit",
            "hidden_log_fit",
            "region_physics",
            "public_sanity",
            "transport_equation",
            "ade_correctness",
            "method_workflow",
            "bounds",
            "format",
        ]
        weakest = min(priority, key=lambda name: component_ratios.get(name, 1.0))
        if weakest == "future_time_extrapolation":
            hint = "Future extrapolation is weakest; tune release timing, duration, source mass, and plume tail behavior."
        elif weakest == "hidden_well_prediction":
            hint = "Hidden well prediction is weakest; tune source location and rectangular geometry before timing."
        elif weakest == "future_log_fit":
            hint = "Future log fit is weak; improve low-concentration tail timing and avoid broad overprediction."
        elif weakest == "hidden_log_fit":
            hint = "Hidden log fit is weak; improve arrival timing and order-of-magnitude concentration fit."
        elif weakest == "region_physics":
            hint = "Prediction is plausible but source geometry or physical consistency is weak."
        elif weakest == "public_sanity":
            hint = "Public sanity fit is weak; verify predictions against public censored monitoring data first."
        elif weakest == "transport_equation":
            hint = "Improve the ADE description and public hydrogeologic parameter reporting in answer.json."
        elif weakest == "ade_correctness":
            hint = "Forward-model ADE probes are still weak; copy or adapt public_forward_model.py and rerun local_validate_forward_model.py."
        else:
            hint = "Continue local refinement around the best candidate while preserving valid format and bounds."

    model_feedback = (detail.get("model_relevance_detail", {}) or {}).get("failures", []) or []
    ade_feedback = (detail.get("ade_correctness_detail", {}) or {}).get("failures", []) or []
    physical_feedback = (detail.get("physical_constraint_detail", {}) or {}).get("failures", []) or []
    if detail.get("model_gate_status") != "passed" and model_feedback:
        hint = hint + " Model issues: " + ", ".join(str(x) for x in model_feedback[:3]) + "."
    elif ade_gate_status != "passed" and ade_feedback:
        hint = hint + " ADE probe issues: " + ", ".join(str(x) for x in ade_feedback[:3]) + "."
    elif detail.get("physical_cap_reason") and physical_feedback:
        hint = hint + " Physical issues: " + ", ".join(str(x) for x in physical_feedback[:3]) + "."

    final_score = round3(detail.get("total_score", 0.0))
    visible_learning_score = round3(detail.get("raw_total_before_cap", 0.0))
    process_scores = detail.get("process_scores", {}) or {}
    return {
        "final_score": final_score,
        "strict_ade_final_score": final_score,
        "strict_score": final_score,
        "visible_learning_score": visible_learning_score,
        "learning_score": visible_learning_score,
        "raw_total_score": visible_learning_score,
        "total_score": final_score,
        "score_semantics": {
            "score": "strict process score after soft dependency and optional time caps",
            "visible_learning_score": (
                "continuous raw DAG/process score before soft end-to-end and time caps; "
                "later tasks receive isolated credit even when upstream ADE is weak"
            ),
        },
        "cap_delta": round3(visible_learning_score - final_score),
        "process_scores": process_scores,
        "isolated_score": process_scores.get("carry_score"),
        "native_score": process_scores.get("native_score"),
        "upstream_dependency_quality": process_scores.get("upstream_dependency_quality"),
        "native_downstream_multiplier": process_scores.get("native_downstream_multiplier"),
        "time_cap": process_scores.get("time_cap"),
        "time_cap_applied": process_scores.get("time_cap_applied"),
        "prediction_band": detail.get("prediction_band", "unknown"),
        "cap_reason": detail.get("cap_reason", "unknown"),
        "metric_cap_reason": detail.get("metric_cap_reason"),
        "prediction_quality_cap": detail.get("prediction_quality_cap"),
        "prediction_quality_cap_reason": detail.get("prediction_quality_cap_reason"),
        "ade_quality_cap": detail.get("ade_quality_cap"),
        "ade_quality_cap_reason": detail.get("ade_quality_cap_reason"),
        "physical_cap_reason": detail.get("physical_cap_reason"),
        "model_cap_reason": detail.get("model_cap_reason"),
        "region_gate_reason": region_gate_reason or None,
        "model_gate_status": detail.get("model_gate_status", "unknown"),
        "ade_correctness_gate_status": ade_gate_status,
        "ade_correctness_score": round3(detail.get("ade_correctness_score", 0.0)),
        "ade_probe_band": progress_band(detail.get("ade_correctness_score", 0.0), ADE_CORRECTNESS_MAX_SCORE),
        "ade_probe_component": ade_detail.get("weakest_component"),
        "post_ade_components_zeroed": bool(detail.get("post_ade_components_zeroed", False)),
        "post_ade_zeroed_reason": detail.get("post_ade_zeroed_reason"),
        "post_ade_zeroed_components": detail.get("post_ade_zeroed_components", []),
        "public_sanity_score_before_ade_gate": round3(detail.get("public_sanity_score_before_ade_gate", 0.0)),
        "physical_constraint_status": detail.get("physical_constraint_status", "unknown"),
        "metric_bands": {
            "hidden_well_rRMSE": metric_feedback_band(hidden_rrmse, [0.05, 0.10, 0.20, 0.85]),
            "future_public_rRMSE": metric_feedback_band(future_rrmse, [0.08, 0.16, 0.25, 0.90]),
            "hidden_well_log_rmse": metric_feedback_band(hidden_log, [0.012, 0.025, 0.045, 0.110]),
            "future_public_log_rmse": metric_feedback_band(future_log, [0.015, 0.030, 0.055, 0.120]),
        },
        "component_scores": component_scores,
        "component_progress": component_progress,
        "weakest_component": weakest,
        "next_hint": hint,
        "model_feedback": [str(x) for x in model_feedback],
        "ade_feedback": [str(x) for x in ade_feedback],
        "ade_probe_feedback": ade_detail.get("probe_feedback", []),
        "physical_feedback": [str(x) for x in physical_feedback],
    }


def inspect_forbidden_access(submission_dir: Path) -> list[str]:
    warnings: list[str] = []
    forbidden = [
        "hidden_monitoring_data", "hidden_wells", "hidden_true_region_source",
        "hidden_forward_model", "hidden_eval_config", "/opt/borden_scoring",
        "private_generation_record_not_for_agent", "scoring/", "MODFLOW", "MT3DMS", "flopy.run_model",
    ]
    for p in submission_dir.rglob("*.py"):
        rel = p.relative_to(submission_dir).as_posix()
        if rel.startswith("scoring/"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for kw in forbidden:
            if kw in text:
                warnings.append(f"Forbidden/suspicious keyword `{kw}` in {rel}")
    return warnings


# ============================================================
# Main evaluation
# ============================================================

def evaluate(submission_dir: Path, case_dir: Path, scoring_dir: Path, output: Path) -> dict[str, Any]:
    answer_path = submission_dir / "answer.json"
    config_path = case_dir / "public_problem_config.json"
    prior_path = case_dir / "public_source_prior.json"

    detail: dict[str, Any] = {
        "total_score": 0,
        "raw_total_before_cap": 0,
        "format_score": 0,
        "bounds_score": 0,
        "public_sanity_score": 0,
        "public_sanity_score_before_ade_gate": 0,
        "post_ade_components_zeroed": False,
        "post_ade_zeroed_components": [],
        "post_ade_zeroed_reason": None,
        "method_workflow_score": 0,
        "transport_equation_score": 0,
        "transport_equation_detail": {},
        "model_relevance_score": 0,
        "model_relevance_detail": {},
        "model_gate_status": "unknown",
        "ade_correctness_score": 0.0,
        "ade_correctness_detail": {},
        "ade_correctness_gate_status": "unknown",
        "metric_score_linear": 0,
        "region_physics_score": 0,
        "region_physics_detail": {},
        "physical_constraint_status": "unknown",
        "physical_constraint_detail": {},
        "metric_score_detail": {},
        "metrics": {},
        "public_metrics": {},
        "warnings": [],
        "errors": [],
    }

    if not answer_path.exists():
        detail["errors"].append("answer.json not found")
        save_json(detail, output)
        print(f"CASE {TASK_NAME} WA score=0")
        print("TOTAL_SCORE 0")
        return detail

    try:
        answer = load_json(answer_path)
    except Exception as exc:
        detail["errors"].append(f"Failed to load answer.json: {exc}")
        save_json(detail, output)
        print(f"CASE {TASK_NAME} WA score=0")
        print("TOTAL_SCORE 0")
        return detail

    try:
        config = load_json(config_path)
    except Exception:
        config = {}

    answer_summary = {
        "source_type": answer.get("source_type"),
        "x_center": answer.get("x_center"),
        "y_center": answer.get("y_center"),
        "z_center": answer.get("z_center"),
        "half_length_x": answer.get("half_length_x"),
        "half_length_y": answer.get("half_length_y"),
        "half_length_z": answer.get("half_length_z"),
        "C0": answer.get("C0"),
        "t_start": answer.get("t_start"),
        "duration": answer.get("duration"),
        "transport_model": answer.get("transport_model"),
    }

    physical_detail = validate_physical_constraints(answer, config)
    detail["physical_constraint_detail"] = physical_detail
    detail["physical_constraint_status"] = physical_detail.get("status", "unknown")

    detail["format_score"] = score_format(answer, True)
    detail["bounds_score"] = score_bounds(answer, config, prior_path)
    detail["method_workflow_score"] = score_method_and_workflow(answer, submission_dir)
    transport_score, transport_detail = score_transport_equation(answer, config)
    detail["transport_equation_score"] = transport_score
    detail["transport_equation_detail"] = transport_detail
    model_relevance_score, model_relevance_detail = score_model_relevance(answer, submission_dir)
    detail["model_relevance_score"] = model_relevance_score
    detail["model_relevance_detail"] = model_relevance_detail
    detail["model_gate_status"] = "passed" if model_relevance_score >= ADE_MODEL_GATE_SCORE else "failed"
    detail["warnings"].extend(inspect_forbidden_access(submission_dir))

    if detail["model_gate_status"] == "passed":
        ade_correctness_score, ade_correctness_detail = score_ade_correctness_gate(
            answer, submission_dir, case_dir, config, scoring_dir
        )
        detail["ade_correctness_score"] = ade_correctness_score
        detail["ade_correctness_detail"] = ade_correctness_detail
        detail["ade_correctness_gate_status"] = (
            "passed" if ade_correctness_score >= ADE_CORRECTNESS_GATE_SCORE else "failed"
        )
    else:
        detail["ade_correctness_detail"] = {
            "status": "not_run",
            "gate_reason": "static_model_gate_failed",
            "failures": ["static_model_gate_failed"],
        }
        detail["ade_correctness_gate_status"] = "not_run"

    # Public sanity score is small and intentionally non-dominant.
    public_sanity_score, public_metrics = score_public_sanity(answer, case_dir, config, scoring_dir)
    detail["public_sanity_score"] = public_sanity_score
    detail["public_sanity_score_before_ade_gate"] = public_sanity_score
    detail["public_metrics"] = public_metrics

    # Score the submitted source answer with the judge reference model whenever
    # the answer itself is structurally usable. This is the "isolated" process
    # check: a weak submitted forward_model.py should hurt native/end-to-end
    # credit, but it should not make the judge unable to assess later steps.
    if detail["format_score"] > 0 and detail["bounds_score"] > 0:
        try:
            sys.path.insert(0, str(scoring_dir))
            from hidden_forward_model import simulate_from_answer  # type: ignore

            hidden_wells = pd.read_csv(scoring_dir / "hidden_wells.csv")
            public_wells_path = case_dir / "public_wells.csv"
            if public_wells_path.exists():
                public_wells = pd.read_csv(public_wells_path)
                pred_wells = pd.concat([hidden_wells, public_wells], ignore_index=True)
                pred_wells = pred_wells.drop_duplicates(subset=["well_id"], keep="first")
            else:
                pred_wells = hidden_wells
            hidden_obs = pd.read_csv(scoring_dir / "hidden_monitoring_data.csv")
            times = sorted(hidden_obs["time_days"].astype(float).unique())
            pred = simulate_from_answer(answer, pred_wells, np.asarray(times, dtype=float), config)
            metrics = compute_grouped_hidden_metrics(hidden_obs, pred)
            detail["metrics"] = metrics

            metric_score, metric_score_detail = score_metric_linear(metrics)
            detail["metric_score_linear"] = metric_score
            detail["metric_score_detail"] = metric_score_detail
            region_physics_score, region_physics_detail = score_region_physics(
                answer, metrics, config, scoring_dir, physical_detail
            )
            detail["region_physics_score"] = region_physics_score
            detail["region_physics_detail"] = region_physics_detail

        except Exception as exc:
            detail["errors"].append(f"Hidden metric scoring failed: {exc}")
            detail["metrics"] = {}
    else:
        detail["region_physics_detail"] = {
            "gated": True,
            "gate_reason": (
                "answer_format_or_bounds_invalid"
            ),
        }

    # Keep shallow/public-only answers from receiving large easy points.
    detail["format_score"] = min(float(detail["format_score"]), 1.0)
    detail["bounds_score"] = min(float(detail["bounds_score"]), 1.0)
    detail["public_sanity_score"] = min(float(detail["public_sanity_score"]), 2.0)
    detail["public_sanity_score_before_ade_gate"] = min(
        float(detail["public_sanity_score_before_ade_gate"]), 2.0
    )
    detail["method_workflow_score"] = min(float(detail["method_workflow_score"]), 1.0)
    detail["transport_equation_score"] = min(float(detail["transport_equation_score"]), 2.0)

    # Old judge versions zeroed all post-ADE components here when the submitted
    # forward model missed the ADE threshold. That created a hard plateau. The
    # new process judge keeps those isolated component scores and applies a soft
    # native dependency multiplier in compute_process_scores().
    detail["post_ade_components_zeroed"] = False
    detail["post_ade_zeroed_reason"] = None
    detail["post_ade_zeroed_components"] = []

    detail["region_physics_score_before_gate"] = round3(detail["region_physics_score"])
    detail["region_physics_score"] = min(float(detail["region_physics_score"]), 15.0)

    metrics_for_caps = detail.get("metrics", {}) or {}
    detail["prediction_band"] = public_prediction_band(metrics_for_caps)

    legacy_raw_total = (
        detail["format_score"]
        + detail["bounds_score"]
        + detail["public_sanity_score"]
        + detail["method_workflow_score"]
        + detail["transport_equation_score"]
        + detail["ade_correctness_score"]
        + detail["metric_score_linear"]
        + detail["region_physics_score"]
    )
    detail["legacy_raw_total_before_cap"] = float(legacy_raw_total)
    # Official score: keep the process/DAG scoring and caps. Agent-visible
    # feedback is a separate project-review translation layer below.
    metric_reason = metric_cap_reason(metrics_for_caps)
    process_detail = compute_process_scores(detail)
    detail["process_scores"] = process_detail
    detail["raw_total_before_cap"] = float(process_detail["raw_process_score"])
    total_score = float(process_detail["score_after_caps"])
    prediction_cap_value, prediction_cap_reason = prediction_quality_cap(detail)
    ade_cap_value, ade_cap_reason = ade_quality_cap(detail)
    total_score = min(total_score, prediction_cap_value, ade_cap_value)

    model_reason = process_detail.get("structural_cap_reason")
    phys_reason = None
    if detail["physical_constraint_status"] != "passed":
        phys_reason = "physical_constraints_reduce_source_physics_component"
    detail["metric_cap_reason"] = metric_reason
    detail["prediction_quality_cap"] = round3(prediction_cap_value)
    detail["prediction_quality_cap_reason"] = prediction_cap_reason
    detail["ade_quality_cap"] = round3(ade_cap_value)
    detail["ade_quality_cap_reason"] = ade_cap_reason
    detail["physical_cap_reason"] = phys_reason
    detail["model_cap_reason"] = model_reason
    cap_parts = []
    if model_reason:
        cap_parts.append(str(model_reason))
    if prediction_cap_reason != "none":
        cap_parts.append(str(prediction_cap_reason))
    if ade_cap_reason != "none":
        cap_parts.append(str(ade_cap_reason))
    if process_detail.get("time_cap_applied"):
        cap_parts.append("time_progress_cap")
    if process_detail.get("soft_end_to_end_cap", PROCESS_SCORE_MAX) < process_detail.get("raw_process_score", 0.0):
        cap_parts.append("soft_end_to_end_cap")
    detail["cap_reason"] = "+".join(cap_parts) if cap_parts else "none"
    total_score = min(total_score, PROCESS_SCORE_MAX)
    detail["total_score"] = round3(total_score)

    # Internal artifact review is used only to translate feedback into
    # project-review language. It does not determine official score/caps.
    detail["project_artifact_review"] = score_realistic_project_stages(detail, submission_dir, case_dir, answer)

    save_json(detail, output)

    print(f"CASE {TASK_NAME} OK score={detail['total_score']}")
    print(f"CASE borden_inverse OK score={float(detail['total_score']):.3f}")
    print(f"TOTAL_SCORE {float(detail['total_score']):.3f}")
    review_feedback_for_stdout = project_review_feedback(detail)
    print(f"REVIEW_STATUS {review_feedback_for_stdout.get('review_status', 'needs_revision')}")
    print(f"PROCESS_STAGE {review_feedback_for_stdout.get('process_stage', 'unstarted')}")
    print(f"MODEL_STATUS {review_feedback_for_stdout.get('model_status', 'not_reviewed')}")
    print(f"PUBLIC_FIT {review_feedback_for_stdout.get('public_fit', 'not_reviewed')}")
    print(f"VALIDATION_STATUS {review_feedback_for_stdout.get('validation_status', 'not_reviewed')}")
    print("SUBMISSION_STATUS evaluated")

    # ---- SE-Bench structured_json feedback block ----
    try:
        _detail = locals().get("detail", locals().get("score_detail", {}))

        def _safe_float(x, default=999.0):
            try:
                v = float(x)
                if v != v:
                    return default
                return v
            except Exception:
                return default

        _total_score = _safe_float(
            _detail.get("total_score", locals().get("total_score", locals().get("score", 0.0))),
            default=0.0,
        )
        _pass_rate = max(0.0, min(1.0, _total_score / 100.0))
        _format_ok = _safe_float(_detail.get("format_score", 0), 0) > 0
        _submission_ok = not bool(_detail.get("errors"))
        _review_state = project_review_state(_detail)
        _review_feedback = project_review_feedback(_detail)

        structured_result = {
            "valid": True,
            "score": float(_total_score),
            "pass_rate": _pass_rate,
            "summary": safe_summary(_detail),
            "metrics": _review_feedback,
            "details": [
                {
                    "name": "data_understanding",
                    "status": "PASSED" if _review_state.get("data_understanding") in {"adequate", "strong"} else "NEEDS_REVIEW",
                    "score": None,
                    "weight": None,
                    "message": "data audit covers coordinates, units, monitoring records, detection limits, and breakthrough timing",
                },
                {
                    "name": "forward_model",
                    "status": "PASSED" if _review_state.get("forward_model") in {"adequate", "strong"} else "NEEDS_REVIEW",
                    "score": None,
                    "weight": None,
                    "message": "forward ADE model is reviewed against public source-superposition and release-time behavior",
                },
                {
                    "name": "inversion_framework",
                    "status": "PASSED" if _review_state.get("inversion_framework") in {"adequate", "strong"} else "NEEDS_REVIEW",
                    "score": None,
                    "weight": None,
                    "message": "inversion parameterization, bounds, objective function, and baseline solver evidence are reviewed",
                },
                {
                    "name": "optimization_evidence",
                    "status": "PASSED" if _review_state.get("optimization_evidence") in {"adequate", "strong"} else "NEEDS_REVIEW",
                    "score": None,
                    "weight": None,
                    "message": "optimization trace, candidate comparison, and public residual diagnostics are reviewed",
                },
                {
                    "name": "report_quality",
                    "status": "PASSED" if _review_state.get("report_quality") in {"adequate", "strong"} else "NEEDS_REVIEW",
                    "score": None,
                    "weight": None,
                    "message": "final report, uncertainty discussion, public predictions, and reproducibility notes are reviewed",
                },
                {
                    "name": "withheld_validation_review",
                    "status": "PASSED" if _review_state.get("validation_status") in {"acceptable", "defensible"} else "NEEDS_REVIEW",
                    "score": None,
                    "weight": None,
                    "message": _review_feedback.get("validation_review", "withheld predictive review not defensible"),
                },
                {
                    "name": "evaluation",
                    "status": "PASSED" if _submission_ok else "FAILED",
                    "score": None,
                    "weight": 100,
                    "message": "submission evaluated; project-review feedback returned without hidden residual disclosure",
                },
            ],
        }

        print(">>>>> Start Structured Result")
        print(json.dumps(structured_result, ensure_ascii=False))
        print(">>>>> End Structured Result")
    except Exception as _structured_exc:
        print("STRUCTURED_RESULT_ERROR", repr(_structured_exc))
    # ---- End SE-Bench structured_json feedback block ----
    if detail["warnings"]:
        print("WARNINGS", json.dumps(detail["warnings"], ensure_ascii=False))
    if detail["errors"]:
        print("ERRORS", json.dumps(detail["errors"], ensure_ascii=False))

    return detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--scoring_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evaluate(
        submission_dir=Path(args.submission_dir),
        case_dir=Path(args.case_dir),
        scoring_dir=Path(args.scoring_dir),
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
