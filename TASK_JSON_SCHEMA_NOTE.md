# Task JSON schema update

`generated/tasks/borden_inverse.json` and `templates/borden_inverse.json` have been rewritten to match the SE-Bench task schema shown in the project documentation.

Important fields added/renamed:

- Added top-level `language`, `base_image`, `platform`, `cwd`, `submit_paths`, `submit_exclude`, `internet`, `status`, and `source`.
- Renamed `work.setup` to `work.setup_cmds`.
- Removed `work.base_image` and `work.submitted_files`; submission paths are now top-level `submit_paths`.
- Renamed `judge.setup` to `judge.setup_cmds`.
- Renamed `judge.test` to `judge.eval_cmd`.
- Added `judge.eval_timeout`, `judge.parser`, `judge.score_direction`, and `judge.selection`.
- Set `judge.parser` to `score_sum`. Therefore `evaluate.py` now prints lines such as `CASE borden_inverse OK score=15` and `TOTAL_SCORE 15`.

The scientific task files and region-source generator are otherwise unchanged.
