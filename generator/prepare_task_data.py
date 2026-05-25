
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from borden_scene import (
    build_problem_config,
    default_region_source,
    default_public_wells,
    default_hidden_wells,
    public_times,
    hidden_well_times,
    future_times,
    simulate_region_monitoring_concentrations,
    add_measurement_noise,
    write_grid_npz,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated_noisy_slope_v3"
TASK_DIR = OUT / "borden_inverse"
SCORING_DIR = TASK_DIR / "scoring"
TASKS_DIR = OUT / "tasks"


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_zip(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir.parent))


def force_rmtree(path: Path):
    def onerror(func, p, _exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except Exception:
            raise

    shutil.rmtree(path, onerror=onerror)


def main():
    if OUT.exists():
        force_rmtree(OUT)
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    config = build_problem_config()
    source = default_region_source()
    public_wells = default_public_wells()
    hidden_wells = default_hidden_wells()

    public_clean = simulate_region_monitoring_concentrations(source, public_wells, public_times(), config)
    public_obs = add_measurement_noise(public_clean, noise_relative=0.070, noise_absolute=0.004, seed=20260522)
    public_obs["eval_split"] = "public_noisy_censored_mixed_times"

    hidden_clean = simulate_region_monitoring_concentrations(source, hidden_wells, hidden_well_times(), config)
    hidden_obs = add_measurement_noise(hidden_clean, noise_relative=0.025, noise_absolute=0.002, seed=20260523)
    hidden_obs["eval_split"] = "hidden_well"

    future_clean = simulate_region_monitoring_concentrations(source, public_wells, future_times(), config)
    future_obs = add_measurement_noise(future_clean, noise_relative=0.025, noise_absolute=0.002, seed=20260524)
    future_obs["eval_split"] = "future_public"

    # Agent visible files
    write_json(config, TASK_DIR / "public_problem_config.json")
    write_json(config["source_search_bounds_for_agent"], TASK_DIR / "public_source_prior.json")
    public_wells.to_csv(TASK_DIR / "public_wells.csv", index=False)
    public_agent = public_obs.drop(
        columns=["concentration_mg_L", "concentration_clean_mg_L"],
        errors="ignore",
    )
    public_agent.to_csv(TASK_DIR / "public_monitoring_data.csv", index=False)
    public_obs.drop(columns=["concentration_mg_L"], errors="ignore").to_csv(
        SCORING_DIR / "public_monitoring_data_with_clean.csv",
        index=False,
    )
    write_grid_npz(TASK_DIR / "borden_grid.npz")

    # Judge hidden files
    hidden_eval_config = {
        "scoring_policy": "finite_region_hidden_monitoring_prediction",
        "public_data_policy": "agent sees only noisy censored public observations; clean public concentrations are private",
        "easy_points_max_after_caps": 5,
        "public_observed_sanity_score": 10,
        "transport_equation_score": 8,
        "hidden_coarse_score": 14,
        "hidden_precision_score": 71,
        "region_physics_score": 15,
        "caps": {
            "if_hidden_or_future_rrmse_ge_1.60": 15,
            "if_hidden_rrmse_ge_0.85_or_future_rrmse_ge_0.90": 30,
            "if_hidden_rrmse_ge_0.35_or_future_rrmse_ge_0.40": 45
        }
    }
    write_json(hidden_eval_config, SCORING_DIR / "hidden_eval_config.json")
    write_json(source, SCORING_DIR / "hidden_true_region_source.json")
    hidden_wells.to_csv(SCORING_DIR / "hidden_wells.csv", index=False)
    pd.concat([hidden_obs, future_obs], ignore_index=True).drop(columns=["concentration_mg_L"]).to_csv(SCORING_DIR / "hidden_monitoring_data.csv", index=False)

    # Keep generation record outside task bundle too, for author-side reproducibility.
    write_json({"generated_region_source_not_for_agent": source}, OUT / "private_generation_record_not_for_agent_or_judge.json")

    template_root = ROOT / "templates"
    for name in ["hidden_forward_model.py", "evaluate.py"]:
        shutil.copyfile(template_root / name, SCORING_DIR / name)
    for name in ["README.md", "requirements.txt", "answer_template.json", "baseline_solver.py"]:
        shutil.copyfile(template_root / name, TASK_DIR / name)

    # Baseline answer
    b = config["source_search_bounds_for_agent"]
    def mid(lo, hi): return 0.5 * (float(lo) + float(hi))
    baseline_answer = {
        "source_type": "rectangular_region",
        "dimension": 3,
        "x_center": mid(b["x_center_min"], b["x_center_max"]),
        "y_center": mid(b["y_center_min"], b["y_center_max"]),
        "z_center": mid(b["z_center_min"], b["z_center_max"]),
        "half_length_x": mid(b["half_length_x_min"], b["half_length_x_max"]),
        "half_length_y": mid(b["half_length_y_min"], b["half_length_y_max"]),
        "half_length_z": mid(b["half_length_z_min"], b["half_length_z_max"]),
        "C0": mid(b["C0_min"], b["C0_max"]),
        "t_start": mid(b["t_start_min"], b["t_start_max"]),
        "duration": mid(b["duration_min"], b["duration_max"]),
        "transport_model": {
            "equation_type": "advection_dispersion_reaction",
            "governing_equation": "R*dC/dt = div(D grad C) - v dot grad C - lambda*C + source",
            "velocity_m_per_day": config.get("hydrogeological_parameters", {}).get("velocity_m_per_day", 0.0),
            "alpha_L_m": config.get("hydrogeological_parameters", {}).get("alpha_L_m", 0.0),
            "alpha_TH_m": config.get("hydrogeological_parameters", {}).get("alpha_TH_m", 0.0),
            "alpha_TV_m": config.get("hydrogeological_parameters", {}).get("alpha_TV_m", 0.0),
            "porosity": config.get("hydrogeological_parameters", {}).get("porosity", 0.0),
            "retardation_factor": config.get("hydrogeological_parameters", {}).get("retardation_factor", 1.0),
            "lambda_per_day": config.get("hydrogeological_parameters", {}).get("lambda_per_day", 0.0),
            "numerical_approach": "baseline placeholder; replace with calibrated ADE region-source model",
        },
        "method": "baseline center of finite-duration rectangular-region source bounds; replace with optimized inversion result",
    }
    write_json(baseline_answer, TASK_DIR / "answer.json")

    shutil.copyfile(template_root / "borden_inverse.json", TASKS_DIR / "borden_inverse.json")
    shutil.copyfile(template_root / "README_DEPLOY.md", OUT / "README_DEPLOY.md")

    make_zip(TASK_DIR, OUT / "borden_inverse_task_bundle.zip")
    print("Generated:", OUT)
    print("Task bundle:", OUT / "borden_inverse_task_bundle.zip")


if __name__ == "__main__":
    main()
