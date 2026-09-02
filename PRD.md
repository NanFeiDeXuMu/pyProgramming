# SemNav: Embodiment-Agnostic Semantic Navigation

## 1. Overview

SemNav is a Python-first system that explores unseen indoor environments, builds spatial and semantic memory from RGB or RGB-D observations, and navigates to a target.

The 12-week project will be developed and evaluated in Habitat. Core navigation must remain independent of simulator APIs so the same system can later accept real RGB-D input or control a simulated humanoid through adapters.

## 2. Research Goal

**Question:** Can semantic context and language-based commonsense improve active exploration efficiency in unseen environments?

The main contribution is a frontier-selection strategy that combines information gain, semantic relevance, commonsense priors, and travel cost. The project will compare this strategy with non-semantic exploration baselines.

## 3. Core Scenario

Example instruction: **“Find something I can use to make coffee.”**

The agent repeatedly:

1. Observes the environment.
2. Updates its map, pose, and semantic memory.
3. Detects and scores unexplored frontiers.
4. Plans and follows a path to the best frontier.
5. Stops when it reaches the target.

```text
Observation -> Perception -> Spatial Memory -> Frontier Detection
            -> Goal Selection -> Path Planning -> Action -> Observation
```

## 4. Architecture

```text
EnvironmentAdapter
        |
        v
RGB / Depth / Pose / Camera Parameters
        |
        v
Semantic Navigator
  |-- Perception
  |-- Spatial Memory
  |-- Language Reasoning
  |-- Active Exploration
  `-- Path Planning
        |
        v
Waypoint or Navigation Action
```

The navigator may only access the environment through a simulator-independent interface:

```python
class EnvironmentAdapter:
    def reset(self): ...
    def get_observation(self): ...
    def execute_action(self, action): ...
    def is_done(self): ...
```

An observation contains RGB, optional depth and pose, camera intrinsics, and a timestamp. Ground-truth data may support debugging and evaluation but must not leak unavailable global information into the navigation policy.

## 5. Core Modules

### Environment

Use Habitat-Sim or Habitat-Lab with HM3D or a compatible indoor dataset. The adapter handles scene loading, observations, actions, and optional ground-truth pose.

### Perception

Use modular pretrained models for visual features, semantic labels, object candidates, and 3D observations. Candidate components include CLIP, DINO-style features, semantic segmentation, and open-vocabulary detection.

### Spatial Memory

The minimum representation includes a 2D occupancy map, agent trajectory, and semantic observations. A semantic 3D map is preferred; a scene graph is optional.

### Frontier Detection

Detect boundaries between explored free space and unexplored space. Record each frontier’s position, travel distance, expected information gain, and nearby semantic context.

### Language Reasoning

Support object goals first, then natural-language goals. Convert instructions into search targets and commonsense priors. Cache LLM or VLM results and call models only for target interpretation, new semantic context, or major replanning.

### Path Planning

Implement A* over the occupancy map. Active exploration decides **where to go**; path planning decides **how to get there**.

## 6. Semantic Active Exploration

This is the primary research component.

Baseline:

```text
U(f) = I(f) - lambda * D(f)
```

Proposed method:

```text
U(f, g) = alpha * I(f)
        + beta  * S(f, g)
        + gamma * C(f, g)
        - lambda * D(f)
```

Where:

- `I(f)`: expected information gain
- `S(f, g)`: semantic relevance to goal `g`
- `C(f, g)`: commonsense prior
- `D(f)`: travel cost

For example, when searching for a microwave, a frontier near a fridge, sink, and counter should rank above one near a bed and wardrobe, even if it is slightly farther away.

## 7. Baselines and Evaluation

Compare:

- **B0 — Random:** random valid navigation
- **B1 — Nearest Frontier:** closest unexplored frontier
- **B2 — Information Gain:** information gain minus travel cost
- **B3 — Semantic Frontier:** B2 plus semantic similarity
- **B4 — Proposed Method:** B3 plus commonsense priors

Run at least 50–100 episodes across unseen scenes, target categories, and random seeds.

Primary metrics:

- Success Rate (SR) and SPL
- Path length, steps, and time to goal
- Exploration coverage

System metrics:

- Planning and perception latency
- LLM or VLM call count
- Memory usage

Use deterministic seeds where possible, export results to CSV or JSON, and run ablations for major scoring components.

## 8. Visualization

The demo must show:

- First-person RGB view
- Occupancy map and explored area
- Agent position and trajectory
- Semantic objects and detected target
- Candidate frontier scores and selected frontier
- Planned path and current reasoning summary

Visualization must remain independent from navigation. Advanced 3D reconstruction is optional.

## 9. Scope

### Must Have

- Habitat adapter with RGB or RGB-D input
- Occupancy and semantic memory
- Frontier detection and A* planning
- Baseline and semantic exploration policies
- Reproducible evaluation and visualization

### Should Have

- Open-vocabulary perception
- Semantic 3D memory
- Natural-language goals and commonsense scoring
- Ablation studies
- Real RGB-D adapter

### Stretch Goals

- Semantic scene graph
- Advanced VLM reasoning or 3D visualization
- Simulated humanoid adapter
- Autonomous robot deployment

### Non-Goals

- Training a foundation model from scratch
- Building SLAM or humanoid locomotion from scratch
- Early ROS 2 or physical robot integration
- Reimplementing a complete ObjectNav system
- UI or rendering work unrelated to evaluation

## 10. Schedule

| Period | Focus | Deliverable |
| --- | --- | --- |
| Weeks 1–2 | Habitat baseline | Agent explores an unseen scene |
| Weeks 3–4 | Mapping and perception | Occupancy and semantic memory |
| Weeks 5–6 | Open-vocabulary search | Text-specified target localization |
| Weeks 7–8 | Semantic exploration | Proposed policy and baseline comparison |
| Week 9 | Language reasoning | Context affects frontier selection |
| Week 10 | Visualization and optional RGB-D adapter | Integrated demo |
| Week 11 | Experiments and ablations | Reproducible results |
| Week 12 | Feature freeze | Evaluation, demo, figures, report, and presentation |

## 11. Success Criteria

- **Minimum:** The Habitat agent completes the full navigation loop and outperforms random or nearest-frontier exploration on at least one meaningful metric.
- **Target:** Semantic exploration consistently improves over non-semantic frontier exploration in unseen environments.
- **Excellent:** The system also supports open-vocabulary or natural-language goals, strong ablations, and real RGB-D validation.
- **Exceptional:** The same core navigator works in Habitat, with a human-carried RGB-D camera, and on a simulated humanoid.

## 12. Engineering Principles

- Keep a runnable end-to-end baseline throughout development.
- Place simulator-specific code behind adapters.
- Separate perception, memory, exploration, planning, and visualization.
- Externalize configuration and script experiments.
- Add major features behind configuration flags.
- Prioritize correctness, reproducibility, and measurable research value.
- Do not let stretch goals block the core system.
