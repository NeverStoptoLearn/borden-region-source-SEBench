# Task JSON schema update

`generated/tasks/borden_inverse.json` and `templates/borden_inverse.json` have been rewritten to match the SE-Bench task schema shown in the project documentation.

Important fields added/renamed:

- Added top-level `language`, `base_image`, `platform`, `cwd`, `submit_paths`, `submit_exclude`, `internet`, `status`, and `source`.
- Renamed `work.setup` to `work.setup_cmds`.
- Removed `work.base_image` and `work.submitted_files`; submission paths are now top-level `submit_paths`.
- Renamed `judge.setup` to `judge.setup_cmds`.
- Renamed `judge.test` to `judge.eval_cmd`.
- Added `judge.eval_timeout`, `judge.parser`, `judge.score_direction`, and `judge.selection`.
- Set `judge.parser` to `structured_json` and `judge.selection` to `score_first` for continuous scoring.
- `evaluate.py` still prints `CASE ...` and `TOTAL_SCORE ...` lines for human readability, but the SE-Bench result is now read from the structured result block.
- The structured result intentionally exposes only score, coarse prediction band, cap reason, and physical-constraint status. It does not expose hidden/future RMSE or hidden answer summaries.

The scientific task files and region-source generator are otherwise unchanged.
