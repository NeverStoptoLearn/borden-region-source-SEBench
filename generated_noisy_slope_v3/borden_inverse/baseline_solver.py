
import json
from pathlib import Path

with open("public_problem_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

b = cfg["source_search_bounds_for_agent"]

def mid(lo, hi):
    return 0.5 * (float(lo) + float(hi))

answer = {
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
    "method": "baseline center of finite-duration rectangular-region source bounds; replace with optimized inversion result"
}

Path("answer.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")
print("Wrote baseline answer.json")
