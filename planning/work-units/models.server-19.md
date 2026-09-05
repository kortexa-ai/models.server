# Snappy MLX upgrade and inspection

Owning issue: https://github.com/kortexa-ai/models.server/issues/19

Scope: Snappy shared .venv-mlx only. No model servers were running at baseline.
Keep Smarty untouched by explicit user direction. Upgrade mlx-lm/mlx-vlm to the
latest stable resolver candidates without upgrading unrelated dependencies.
Record the package inventory, validate imports and dependency compatibility,
inspect execution controls, and rerun the same 230M tiny-request calibration.

Rollback: only mlx-vlm is scheduled to change (0.6.15 to 0.6.17); use uv pip
install --python .venv-mlx/bin/python mlx-vlm==0.6.15 if validation fails.
Confirm the old package resolves offline before mutation. Preserve inventories
and note that the unchanged text backend cannot establish an upgrade speed gain.
