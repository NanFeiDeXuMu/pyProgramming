# AGENTS.md

## Project

SemNav is a Python-first semantic navigation research project. Use [PRD.md](PRD.md) as the product source of truth, [DECISIONS.md](DECISIONS.md) for architectural decisions, and [CURRENT_STATUS.md](CURRENT_STATUS.md) for current progress.

## Working Rules

- Keep the end-to-end baseline runnable after each change.
- Work in small, testable increments; do not implement stretch goals early.
- Keep Habitat-specific code behind an environment adapter.
- Keep perception, memory, frontier detection, exploration, planning, evaluation, and visualization modular.
- Do not use unavailable ground-truth information in navigation logic.
- Externalize experiment settings and seed stochastic components.
- Add or update tests for behavior changed by a patch.
- Update `CURRENT_STATUS.md` when project state or next steps change.
- Record durable architectural choices in `DECISIONS.md`.

## Runtime

- Keep project files in this Windows workspace; access it from Ubuntu at `/mnt/c/Users/nanfe/pyProgramming`.
- Run Habitat commands in WSL2 Ubuntu 22.04 using the Conda environment `semnav`.
- Activate it with `source ~/miniforge3/etc/profile.d/conda.sh && conda activate semnav`.
- Keep scene data outside Git at `/home/nanfe/semnav-data/data`.
- In WSL, set Habitat `gpu_device_id` to `-1`; this lets WSLg select the D3D12-backed EGL device.
- Required Ubuntu packages are `git-lfs`, `libegl1`, `libegl-mesa0`, `libglvnd0`, and `mesa-utils`.

## Initial Quality Checks

Until project tooling is established, verify that:

- New modules import successfully.
- Relevant unit tests pass.
- The smallest available end-to-end smoke test passes.
- Configuration and output formats remain reproducible.

## Scope Guardrails

Do not build foundation models, SLAM, humanoid locomotion, ROS 2 integration, or advanced UI/3D rendering unless they become an explicit project priority.
