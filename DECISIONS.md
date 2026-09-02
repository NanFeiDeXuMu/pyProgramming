# Decision Log

Record decisions that affect architecture, experiments, or project scope. Add new entries rather than rewriting historical ones; mark replaced decisions as superseded.

## Confirmed Decisions

### D-001 — Python-first implementation

- **Status:** Accepted
- **Reason:** Supports rapid research iteration and the target ecosystem.

### D-002 — Environment adapter boundary

- **Status:** Accepted
- **Decision:** Core navigation must not import or depend directly on Habitat APIs.
- **Reason:** Enables later real RGB-D and humanoid adapters without rewriting navigation logic.

### D-003 — Modular navigation pipeline

- **Status:** Accepted
- **Decision:** Separate perception, spatial memory, frontier detection, exploration, path planning, evaluation, and visualization.
- **Reason:** Allows independent testing, replacement, and ablation.

### D-004 — Runnable baseline before semantic features

- **Status:** Accepted
- **Decision:** Build and preserve a simple end-to-end frontier-navigation baseline first.
- **Reason:** Prevents optional research features from blocking integration and evaluation.

### D-005 — Initial mapping and planning

- **Status:** Accepted
- **Decision:** Start with a 2D occupancy map and A* planner.
- **Reason:** They satisfy the minimum scope with low implementation risk.

### D-006 — Reproducible evaluation

- **Status:** Accepted
- **Decision:** Use explicit configurations and seeds, and export episode results to CSV or JSON.
- **Reason:** Baseline comparisons and ablations must be repeatable.

### D-007 — Initial runtime stack

- **Status:** Accepted
- **Decision:** Use WSL2 Ubuntu 22.04, Miniforge, Python 3.9, NumPy 1.26.4, and Habitat-Sim 0.3.3 `headless_nobullet`.
- **Reason:** This is a small, reproducible stack compatible with the approved baseline scope.

### D-008 — Weeks 1–2 integration scope

- **Status:** Accepted
- **Decision:** Use Habitat-Sim directly with the official test scenes; do not add Habitat-Lab yet. Implement random exploration before frontier navigation.
- **Reason:** This isolates simulator integration before navigation complexity is introduced.

### D-009 — WSL graphics device selection

- **Status:** Accepted
- **Decision:** Set Habitat `gpu_device_id=-1` under WSL.
- **Reason:** WSLg exposes the RTX GPU through Mesa/D3D12 rather than a native CUDA-EGL device. The default value `0` fails EGL initialization, while `-1` is verified on this machine.

## Open Decisions

- Package and configuration layout
- Test, lint, and type-check tools
- Initial perception model
- Map resolution and coordinate conventions
- Action space and success threshold
