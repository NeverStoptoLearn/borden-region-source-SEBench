
# Deploy Borden finite-region source task to SE-Bench

## 1. Upload/unzip

```bash
scp borden_region_source_task.zip root@<SERVER>:/root/
ssh root@<SERVER>
cd /root
rm -rf /root/borden_inverse_pkg
mkdir -p /root/borden_inverse_pkg
unzip -o borden_region_source_task.zip -d /root/borden_inverse_pkg
cd /root/borden_inverse_pkg/borden_adepy_generated_task
```

## 2. Optional regeneration

```bash
python3 generator/prepare_task_data.py
```

## 3. Copy task config

```bash
cp /root/borden_inverse_pkg/borden_adepy_generated_task/generated/tasks/borden_inverse.json /root/SE-bench-main/tasks/borden_inverse.json
```

## 4. Serve bundle

Open a persistent terminal:

```bash
cd /root/borden_inverse_pkg/borden_adepy_generated_task/generated_noisy_slope_v3
python3 -m http.server 8000 --bind 0.0.0.0
```

## 5. Rebuild images

```bash
docker rmi sebench.work.borden_inverse:latest 2>/dev/null || true
docker rmi sebench.judge.borden_inverse:latest 2>/dev/null || true
cd /root/SE-bench-main
SEBENCH_EXTRA_HOSTS="host.docker.internal:host-gateway,github.com:140.82.114.4" \
SEBENCH_APT_MIRROR_URL="https://mirrors.aliyun.com" \
SEBENCH_PYPI_INDEX_URL="https://mirrors.ivolces.com/pypi/simple/" \
uv run python -m sebench build --task borden_inverse
```

## 6. Run judge and agent

Terminal A:

```bash
cd /root/SE-bench-main
uv run python -m sebench serve
```

Terminal B:

```bash
cd /root/SE-bench-main
read -s API_KEY
export SEBENCH_AGENT_API_KEY="$API_KEY"
export SEBENCH_AGENT_API_BASE_URL="https://api.seededge.top/v1"
uv run python -m sebench run --task borden_inverse --agent codex-or --model gpt-5.5-0424 --eval-interval 300 --run-id borden-region-001 --timeout 7200
```

## Notes

- The new task is a finite-duration rectangular-region source inverse problem, not a point-source inverse problem.
- Agent-visible public observations are noisy/censored and do not include clean concentrations.
- Easy points are capped tightly; hidden monitoring prediction dominates the strict final score only after the submitted `forward_model.py` passes both static relevance and dynamic public ADE correctness gates. The dynamic ADE threshold is `7.5 / 10`; below it, public sanity, hidden/future prediction, hidden/future log fit, and region physics are zeroed.
- Iterative feedback returns `FINAL_SCORE`, continuous `LEARNING_SCORE`/component feedback, `ADE_CORRECTNESS_SCORE`, `ADE_CORRECTNESS_STATUS`, coarse ADE probe bands, metric bands, cap reason, weakest component, model/physics failures, and a next-step hint without exposing hidden observations or true source parameters.
- Physical source-region constraints are enforced: the rectangle must stay inside the domain, saturated aquifer, and original Borden source-zone prior.
- Existing Borden grid/hydrogeological parameters are reused.
