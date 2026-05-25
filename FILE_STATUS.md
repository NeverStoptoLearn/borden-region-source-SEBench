
# 文件状态说明

## 不需要改动 / 直接沿用

1. `reference/borden_adepy_reproduction.ipynb`
   - 状态：未改动。
   - 用途：保留原始 Borden-AdePy 复现场景来源，作为出题人侧参考文件。

2. Borden 网格与水文背景参数
   - 状态：沿用原任务中的 Borden 网格、水文参数、底板剖面、监测井坐标体系。
   - 体现文件：`generator/borden_scene.py` 中的 `GRID_DEFAULTS`、`HYDRO_DEFAULTS`、`BOTTOM_PROFILE`；生成后的 `borden_grid.npz`。

3. SE-Bench 基本包装方式
   - 状态：沿用原任务的外层结构：`templates/`、`generator/`、`generated/`、`generated/tasks/borden_inverse.json`、`generated/borden_inverse_task_bundle.zip`。

## 需要修改 / 已直接给出

1. `generator/borden_scene.py`
   - 修改原因：由点源正演改为有限持续释放的矩形区域源正演。
   - 核心变化：新增 `default_region_source()`、`simulate_region_monitoring_concentrations()`、区域子源点离散、持续释放时间离散、区域源参数边界。

2. `generator/prepare_task_data.py`
   - 修改原因：生成区域源任务数据，而不是点源任务数据。
   - 核心变化：生成 `public_source_prior.json`、隐藏区域源、隐藏井、未来时间段数据、`hidden_true_region_source.json`。

3. `templates/hidden_forward_model.py`
   - 修改原因：judge 端需要根据 Agent 输出的区域源参数计算隐藏监测井预测浓度。
   - 核心变化：自包含有限持续区域源正演逻辑，不依赖 generator。

4. `templates/evaluate.py`
   - 修改原因：原评分标准过多依赖格式/说明项，不能满足“前期浅层结果低分”的任务目标。
   - 核心变化：隐藏井预测 35 分、未来时间预测 20 分、区域形状 15 分、物理合理性 15 分；简单格式/先验/说明最多 15 分；低质量隐藏预测触发封顶。

5. `templates/README.md`
   - 修改原因：任务题干从点源反演改为有限持续区域源反演。

6. `templates/answer_template.json`
   - 修改原因：输出 schema 从 `x0,y0,z0,C0` 改为区域源参数。

7. `templates/baseline_solver.py`
   - 修改原因：生成合法但低分的区域源 baseline answer。

8. `templates/borden_inverse.json`
   - 修改原因：SE-Bench agent prompt 与 setup 需要同步区域源任务。

9. `templates/README_DEPLOY.md`
   - 修改原因：部署说明路径和任务描述更新。

## 新增文件

1. `public_source_prior.json`
   - Agent 可见，只包含区域源参数搜索范围，不包含答案。

2. `scoring/hidden_true_region_source.json`
   - Judge 隐藏文件，用于区域形状评分，不暴露给 Agent。

3. `FILE_STATUS.md`
   - 当前文件状态说明。

## 生成文件

运行：

```bash
python3 generator/prepare_task_data.py
```

会生成：

```text
generated/borden_inverse/
generated/borden_inverse/scoring/
generated/tasks/borden_inverse.json
generated/borden_inverse_task_bundle.zip
```
