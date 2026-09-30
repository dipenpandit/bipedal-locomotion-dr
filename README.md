# Domain Randomization Study in Bipedal Locomotion 

An empirical study on the limits and capabilities of Domain Randomization (DR) for closing the Sim-to-Real gap in bipedal reinforcement learning. 

## Overview
This repository contains the code, configuration, and analysis for an empirical
study of domain randomization in bipedal locomotion. The study trains Proximal
Policy Optimization (PPO) agents in the MuJoCo `Walker2d-v5` environment and
compares nominal training with training under randomized friction and torso
mass. It evaluates learning performance, robustness to out-of-distribution
physics, and recovery from external horizontal pushes to determine whether
randomizing physical parameters improves generalization and dynamic stability.

## Key Findings
1. **DR improves robustness:** Under severe changes to friction and mass, the Nominal agent's reward fell by about 70%, while the DR agent's reward fell by about 37%.
2. **DR reduces nominal performance:** The DR agent achieved about 22% less peak reward than the Nominal agent in the standard environment.
3. **DR does not improve push recovery:** Both agents had similar survival times when exposed to random horizontal pushes. Robustness to parameter changes did not automatically produce better recovery from external disturbances.

## Project Structure
```text
bipedal-locomotion-dr/
├── assets/                   # Generated plots and evaluation results
│   ├── eval_matrix.json
│   ├── push_sweep.json
│   └── *.png
├── README.md                 # Project overview
├── REPORT.md                 # Research report
└── src/
    ├── config/
    │   ├── config.py         # Configuration loader
    │   └── config.yaml       # Training and evaluation settings
    ├── core/
    │   └── logger.py         # Logging configuration
    ├── evaluate.py           # Robustness evaluation
    ├── evaluate_push.py      # Push-recovery evaluation
    ├── plot_results.py       # Results visualization
    ├── train.py              # PPO training
    └── wrappers.py           # Domain-randomization and push wrappers
```

## Setup
### 1. Prerequisites

Install Python 3.14 or newer and [uv](https://docs.astral.sh/uv/).

### 2. Clone the repository

```bash
git clone https://github.com/dipenpandit/bipedal-locomotion-dr
cd bipedal-locomotion-dr
```

### 3. Install dependencies

`uv` creates the project virtual environment and installs the dependencies
declared in `pyproject.toml`:

```bash
uv sync
```

You can run project commands through the managed environment with `uv run`.

## Usage
### 1. Training

Train both the nominal and domain-randomized PPO agents for 3,000,000 steps:

```bash
uv run python -m src.train
```

The trained models are saved under `models/`, TensorBoard logs under `logs/`,
and intermediate checkpoints under `models/checkpoints/`.

### 2. Robustness evaluation

Evaluate both agents in nominal, mildly randomized, and severely randomized
environments. The results are saved to `assets/eval_matrix.json`:

```bash
uv run python -m src.evaluate
```

### 3. Push-recovery evaluation

Run the push-probability ablation study and save the results to
`assets/push_sweep.json`:

```bash
uv run python -m src.evaluate_push
```

### 4. Visualization

Generate the training-progress, robustness-matrix, and push-recovery figures.
The figures are saved to `assets/`:

```bash
MPLBACKEND=Agg uv run python -m src.plot_results
```

## Configuration

Environment settings, PPO hyperparameters, randomization ranges, evaluation
settings, and checkpoint frequency are centralized in
`src/config/config.yaml`. The configuration loader in `src/config/config.py`
converts these values to a dataclass and validates them before training.