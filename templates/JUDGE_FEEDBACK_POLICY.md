# Borden Inverse Judge Feedback Policy

This task uses the original process/DAG score internally, but translates
agent-visible feedback into a project-review style that mirrors a staged
groundwater source inversion workflow.

1. `FINAL_SCORE` / structured `score`
   - Internal process score after soft end-to-end, ADE, prediction-quality, and
     optional time caps.
   - This is the authoritative score for ranking.

2. Internal `LEARNING_SCORE` / raw process score
   - The judge still computes the raw process score from schema/bounds, static
     model relevance, ADE correctness, workflow, public fit, hidden fit, future
     fit, and source physics.
   - This raw score is stored internally, but is not exposed as a visible metric
     in the structured feedback.

3. Visible review feedback
   - Agents receive stage statuses and qualitative review comments, not exact
     hidden residuals, component scores, or direct hidden-answer hints.
   - Hidden/future prediction metrics remain internal to the judge.

The ADE forward operator is public: `public_forward_model.py`,
`local_validate_forward_model.py`, and the required source-flux/discretization
constants are included in the task bundle. The judge still evaluates hidden
monitoring and future-time predictions with private observations and hidden
well/time splits, but iterative feedback is phrased like a technical review.

## Scoring Sequence

1. Compute the internal process components:
   - schema/bounds;
   - static forward-model relevance;
   - public ADE correctness;
   - workflow/transport-equation evidence;
   - public censored-monitoring sanity fit;
   - hidden-well prediction;
   - future-time prediction;
   - source-region physical consistency.
2. Aggregate them into the raw process score.
3. Apply soft end-to-end caps, ADE caps, hidden/future prediction caps, and any
   time cap to get the official score.
4. Translate the internal state into visible project-review feedback:
   data understanding, forward model, inversion framework, optimization
   evidence, report quality, public residual review, and coarse validation
   defensibility.

The structured result exposes review statuses and qualitative comments. It does
not expose hidden observations, true source parameters, exact pointwise hidden
residuals, hidden metric bands, raw hidden/future component scores, or direct
parameter-tuning hints.

## Structured Feedback Fields

- `review_status`: invalid, needs_revision, provisional, acceptable, or defensible.
- `process_stage`: the completed project stages inferred from submitted files.
- `model_status`: public forward-model review status.
- `public_fit`: public monitoring residual review status.
- `validation_status`: coarse withheld predictive defensibility review.
- `next_review`: a project-review recommendation, not a hidden-answer hint.
- `data_understanding`, `forward_model`, `inversion_framework`,
  `optimization_evidence`, and `report_quality`: qualitative stage statuses.
- `public_residual_review`: qualitative feedback based only on public data.
- `validation_review`: coarse withheld-review language without hidden residuals.

The internal `score.json` may contain full diagnostic fields for author-side
analysis, but `sebench-submit` only prints the structured review fields above.

## Hint Semantics

Feedback should resemble what a technical reviewer or project lead could tell a
modeler: whether data audit, forward model, inversion framework, optimization
evidence, report quality, and overall predictive defensibility are sufficient.
It should not reveal exact hidden residuals, hidden-well directionality, or a
specific source-parameter adjustment.

## Rationale

This keeps the mathematical scoring stable while making the iterative feedback
look like real source-inversion review: if ADE is weak, the visible review says
the current task is forward-model rebuilding; if public calibration is weak, it
points to public residual diagnosis; if hidden/future prediction is weak, it
only reports predictive defensibility, without exposing hidden residuals.
