# Borden Region-Source SE-Bench Task

This repository contains a SE-Bench task for a **Borden-style 3D groundwater contaminant source inversion benchmark**. The task requires an Agent to infer a **finite-duration rectangular-region contaminant source** from public monitoring-well concentration readings.

The benchmark is designed for SE-Bench Agent evaluation. The Agent receives only public hydrogeological and monitoring data, while the Judge evaluates submitted source parameters using hidden monitoring wells and hidden future-time observations.

---

## 1. Task Summary

**Task ID**

```text
borden_inverse
```

**Task type**

```text
3D groundwater contaminant source inversion
```

**Source parameterization**

```text
finite-duration rectangular-region source
```

The Agent must create an `answer.json` file in the task root using the following schema:

```json
{
  "source_type": "rectangular_region",
  "dimension": 3,
  "x_center": 0.0,
  "y_center": 0.0,
  "z_center": 0.0,
  "half_length_x": 0.0,
  "half_length_y": 0.0,
  "half_length_z": 0.0,
  "C0": 0.0,
  "t_start": 0.0,
  "duration": 0.0,
  "method": "brief description of your inversion method"
}
```

The Judge does **not** directly score old point-source parameters such as `x0`, `y0`, `z0`, and `C0`. Instead, it first checks whether the submitted `forward_model.py` solves the finite-region ADE on public probes, then evaluates whether the submitted rectangular-region source can predict withheld monitoring readings.

---

## 2. Repository Structure

```text
.
├── generated_noisy_slope_v3/
│   ├── borden_inverse/
│   │   ├── scoring/
│   │   │   ├── evaluate.py
│   │   │   ├── hidden_eval_config.json
│   │   │   ├── hidden_forward_model.py
│   │   │   ├── hidden_monitoring_data.csv
│   │   │   ├── hidden_true_region_source.json
│   │   │   ├── hidden_wells.csv
│   │   │   └── public_monitoring_data_with_clean.csv
│   │   ├── answer_template.json
│   │   ├── answer.json
│   │   ├── baseline_solver.py
│   │   ├── borden_grid.npz
│   │   ├── public_monitoring_data.csv
│   │   ├── public_problem_config.json
│   │   ├── public_source_prior.json
│   │   ├── public_wells.csv
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── tasks/
│   │   └── borden_inverse.json
│   ├── borden_inverse_task_bundle.zip
│   ├── private_generation_record_not_for_agent_or_judge.json
│   └── README_DEPLOY.md
├── generator/
│   ├── borden_scene.py
│   └── prepare_task_data.py
├── templates/
│   ├── answer_template.json
│   ├── baseline_solver.py
│   ├── borden_inverse.json
│   ├── evaluate.py
│   ├── hidden_forward_model.py
│   ├── README_DEPLOY.md
│   ├── README.md
│   └── requirements.txt
├── FILE_STATUS.md
├── README.md
└── TASK_JSON_SCHEMA_NOTE.md
```

---

## 3. Important Directories

### `generated_noisy_slope_v3/`

This is the generated SE-Bench task package. It contains the Agent-visible task directory, Judge scoring files, task JSON, and deployable task bundle.

### `generated_noisy_slope_v3/borden_inverse/`

This directory is copied into the SE-Bench work and judge containers as the task workspace.

Agent-visible files include:

- `public_problem_config.json`
- `public_source_prior.json`
- `borden_grid.npz`
- `public_wells.csv`
- `public_monitoring_data.csv`
- `baseline_solver.py`
- `answer_template.json`
- `requirements.txt`
- `README.md`

### `generated_noisy_slope_v3/borden_inverse/scoring/`

This directory contains Judge-only files:

- `evaluate.py`
- `hidden_forward_model.py`
- `hidden_eval_config.json`
- `hidden_monitoring_data.csv`
- `hidden_true_region_source.json`
- `hidden_wells.csv`

These files are copied into `/opt/borden_scoring` in the Judge image and hidden from the Agent work image.

### `generated_noisy_slope_v3/tasks/`

Contains the SE-Bench task definition:

```text
generated_noisy_slope_v3/tasks/borden_inverse.json
```

### `generator/`

Contains scripts used to generate the Borden-style groundwater scene and task data.

### `templates/`

Contains reusable templates for regenerating the task, including the template scoring code, baseline solver, answer schema, task JSON, and deployment notes.

---

## 4. SE-Bench Task Configuration

Core task JSON:

```text
generated_noisy_slope_v3/tasks/borden_inverse.json
```

Key fields:

```text
task_id: borden_inverse
cwd: /home/workspace/borden_inverse
submit_paths: ["."]
parser: structured_json
selection: score_first
eval_timeout: 600
```

The work image hides the `scoring/` directory so that the Agent cannot access hidden evaluation files. The Judge image copies the scoring files into:

```text
/opt/borden_scoring
```

---

## 5. Scoring Rule

The final score is capped at 100 points. The latest version makes the finite-region ADE operator public through `public_forward_model.py`, `local_validate_forward_model.py`, and source-flux/discretization constants in `public_problem_config.json`. The judge still checks static `forward_model.py` relevance and dynamic public ADE correctness with a `7.5 / 10` threshold, but score separation is intended to come primarily from hidden-well and future-time prediction rather than reverse-engineering the forward operator. Internally the judge keeps the process/DAG score and caps; externally it translates that state into project-review style feedback.

| Component | Points |
|---|---:|
| Format validity | 2 |
| Parameter bounds validity | 1 |
| Public observation sanity check | 2 |
| ADE / rectangular-region method explanation | 2 |
| Static ADE model relevance | 2 / gate feedback |
| Dynamic public ADE correctness | 8 / gate feedback |
| Hidden monitoring-well prediction | 45 |
| Hidden future-time extrapolation | 30 |
| Region shape and physical consistency | 8 |
| Anti-cheating | penalty / zero only |

The private main metrics are:

- `hidden_well rRMSE`
- `hidden_well log_rmse`
- `future_public rRMSE`
- `future_public log_rmse`

During iterative evaluation, the Agent sees natural-language review fields such as `REVIEW_STATUS`, `PROCESS_STAGE`, `MODEL_STATUS`, `PUBLIC_FIT`, `VALIDATION_STATUS`, `NEXT_REVIEW`, and qualitative stage reviews. Exact hidden/future metric values, component scores, cap internals, hidden observations, true source parameters, and pointwise hidden residuals are kept inside `score.json` and are not printed in the structured result block.

The intended interpretation is:

- structured `score`: official internal process score after caps. Use this for final ranking.
- review fields: qualitative natural-language feedback derived from the internal score state, phrased as project review rather than hidden metric disclosure.
- `score.json`: author-side internal diagnostics; not intended as agent-visible feedback.

See `JUDGE_FEEDBACK_POLICY.md` for the saved policy summary.

### 5.1 Truncated Logarithmic Power Score

For each lower-is-better error metric, the Judge uses a truncated logarithmic power score:

```text
if error <= good:
    score = points
elif error >= bad:
    score = 0
else:
    z = (log(bad) - log(error)) / (log(bad) - log(good))
    score = points * z^gamma
```

Current thresholds:

| Metric | good | bad | points | gamma |
|---|---:|---:|---:|---:|
| `hidden_well rRMSE` | 0.04 | 0.35 | 38 | 2.6 |
| `future_public rRMSE` | 0.05 | 0.40 | 30 | 2.6 |
| `hidden_well log_rmse` | 0.010 | 0.045 | 12 | 2.2 |
| `future_public log_rmse` | 0.012 | 0.055 | 12 | 2.2 |

### 5.2 Quality Caps

Quality caps are applied after computing the internal raw process score:

```text
hidden_well rRMSE >= 1.60 or future_public rRMSE >= 1.60:
    TOTAL_SCORE <= 15

hidden_well rRMSE >= 0.85 or future_public rRMSE >= 0.90:
    TOTAL_SCORE <= 30

hidden_well rRMSE >= 0.35 or future_public rRMSE >= 0.40:
    TOTAL_SCORE <= 45

source region outside domain or saturated aquifer:
    TOTAL_SCORE <= 30

source region outside original source-zone prior:
    TOTAL_SCORE <= 45

weak region physics consistency:
    TOTAL_SCORE <= 60
```

### 5.3 Region Shape and Physical-Consistency Gate

The region-shape and physical-consistency score is at most 15 points. It checks the submitted rectangle against the model domain, saturated vertical extent, original source-zone prior, hidden generated source shape, release timing, and a source-mass proxy. This prevents shallow public-data fits or unphysical surrogate regions from receiving high physical-reasonableness points when prediction quality is not physically explainable.

---

## 6. Baseline

Inside the task workspace:

```bash
python baseline_solver.py
cat answer.json
```

This creates a valid but low-quality baseline answer.

---

## 7. Manual Evaluation

From the server:

```bash
BASE=/root/borden_inverse_pkg/borden_adepy_generated_task/generated_noisy_slope_v3/borden_inverse

python3 $BASE/scoring/evaluate.py \
  --submission_dir $BASE \
  --case_dir $BASE \
  --scoring_dir $BASE/scoring \
  --output /tmp/manual_score.json

cat /tmp/manual_score.json
```

Expected output includes:

```text
CASE borden_inverse OK score=...
TOTAL_SCORE ...
REVIEW_STATUS ...
PROCESS_STAGE ...
MODEL_STATUS ...
PUBLIC_FIT ...
VALIDATION_STATUS ...
```

---

## 8. Build SE-Bench Images

Start the bundle file server:

```bash
cd /root/borden_inverse_pkg/borden_adepy_generated_task/generated_noisy_slope_v3
python3 -m http.server 8000 --bind 0.0.0.0
```

Build images:

```bash
cd /root/SE-bench-main

SEBENCH_EXTRA_HOSTS="host.docker.internal:host-gateway,github.com:140.82.114.4" \
SEBENCH_APT_MIRROR_URL="https://mirrors.aliyun.com" \
SEBENCH_PYPI_INDEX_URL="https://mirrors.ivolces.com/pypi/simple/" \
uv run python -m sebench build --task borden_inverse
```

---

## 9. Run Judge Server

```bash
cd /root/SE-bench-main
uv run python -m sebench serve
```

---

## 10. Run Agent

Example 30-minute run:

```bash
cd /root/SE-bench-main

read -s API_KEY
export SEBENCH_AGENT_API_KEY="$API_KEY"
export SEBENCH_AGENT_API_BASE_URL="https://api.seededge.top/v1"

uv run python -m sebench run \
  --task borden_inverse \
  --agent codex-or \
  --model gpt-5.5-0424 \
  --eval-interval 300 \
  --run-id borden-region-logpower-001-30min \
  --timeout 1800
```

Example 2-hour run:

```bash
cd /root/SE-bench-main

read -s API_KEY
export SEBENCH_AGENT_API_KEY="$API_KEY"
export SEBENCH_AGENT_API_BASE_URL="https://api.seededge.top/v1"

uv run python -m sebench run \
  --task borden_inverse \
  --agent codex-or \
  --model gpt-5.5-0424 \
  --eval-interval 300 \
  --run-id borden-region-logpower-002-2h \
  --timeout 7200
```

---

## 11. Inspect Results

```bash
RUN_ID=borden-region-logpower-001-30min

cat /root/SE-bench-main/logs/runs/$RUN_ID/borden_inverse/final_result.json
```

Inspect submission details:

```bash
cd /root/SE-bench-main/logs/runs/$RUN_ID/borden_inverse/submissions

for d in $(ls -d agent-* auto-* 2>/dev/null | sort -V); do
  echo "================ $d ================"
  grep -E "CASE borden_inverse|TOTAL_SCORE|REVIEW_STATUS|PROCESS_STAGE|MODEL_STATUS|PUBLIC_FIT|VALIDATION_STATUS" "$d/test_output.txt"
done
```

---

## 12. Privacy and Hidden Files

This repository contains Judge-side hidden scoring files. For internal SE-Bench task delivery, use a private GitHub repository.

If this repository must be public, remove or exclude:

```text
generated_noisy_slope_v3/borden_inverse/scoring/
generated_noisy_slope_v3/private_generation_record_not_for_agent_or_judge.json
**/hidden_*
```

---

## 13. Delivery Metadata

```text
task_id: borden_inverse
cwd: /home/workspace/borden_inverse
task json: generated_noisy_slope_v3/tasks/borden_inverse.json
task bundle: generated_noisy_slope_v3/borden_inverse_task_bundle.zip
work image: sebench.work.borden_inverse:latest
judge image: sebench.judge.borden_inverse:latest
base image: sebench.base.python310:latest
```
