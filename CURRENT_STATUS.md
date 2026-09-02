# Current Status

**Updated:** 2026-09-02
**Phase:** Project setup
**Overall status:** Habitat environment ready; baseline implementation not started

## Completed

- Defined the product and research scope in [PRD.md](PRD.md).
- Established repository guidance in [AGENTS.md](AGENTS.md).
- Captured initial architectural decisions in [DECISIONS.md](DECISIONS.md).
- Installed WSL2 Ubuntu 22.04 with user `nanfe` and verified workspace access.
- Installed Miniforge, created the `semnav` environment, and captured it in [environment.yml](environment.yml).
- Installed Habitat-Sim 0.3.3 `headless_nobullet` with Python 3.9.23 and NumPy 1.26.4.
- Installed the Ubuntu EGL/GLVND runtime and verified RTX 5070 Ti rendering through WSLg D3D12.
- Downloaded the official Habitat test scenes to `/home/nanfe/semnav-data/data`.
- Passed a Skokloster RGB-D/navmesh/turn-action smoke test.

## Current Focus

Create the smallest runnable Habitat baseline while keeping core navigation independent of Habitat APIs.

## Next Steps

1. Create the package, test, configuration, and experiment layout.
2. Define shared observation, action, episode, and adapter interfaces.
3. Implement a minimal Habitat adapter with `gpu_device_id=-1` under WSL.
4. Add a committed environment smoke test.
5. Add seeded random exploration.
6. Add episode logging and a short baseline run.

## Milestone Progress

- [ ] Weeks 1–2: Habitat exploration baseline
- [ ] Weeks 3–4: Occupancy and semantic memory
- [ ] Weeks 5–6: Open-vocabulary target search
- [ ] Weeks 7–8: Semantic exploration and baseline comparison
- [ ] Week 9: Language and commonsense reasoning
- [ ] Week 10: Integrated visualization
- [ ] Week 11: Experiments and ablations
- [ ] Week 12: Evaluation and final deliverables

## Known Blockers

None.

## Known Constraints

- WSL prints a localhost-proxy warning, but direct package downloads work.
- Hugging Face connections are reset on the current network; the official legacy Habitat mirror was used for test scenes.
- The test scenes have no semantic annotations, so semantic-sensor warnings are expected during this baseline phase.
- Habitat must use `gpu_device_id=-1` in WSL; `0` requests an unavailable native CUDA-EGL mapping.
