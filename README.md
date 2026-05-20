## [Neural Control: Adjoint Learning Through Equilibrium Constraints](https://henryzh007.github.io/neural-control/index.html)

This repository contains the C++ simulator, Python
learning scripts and website codes.

The simulator is exposed to Python through
`pybind11` as the module `nn_der`. The simulation codes are in `src/` and the learning scripts in
`learning_scripts/`. These python scripts use different method to solve the gradient for the three
control tasks reported in the paper.

Here's our [website](https://henryzh007.github.io/neural-control/index.html).

***

### Repository layout

- `src/` &mdash; C++ quasi-static elastic-rod simulator (stretching, bending,
  twisting, gravity, damping, IMC contact, Newton solver with line search).
  `src/app.cpp` exposes the simulator to Python through `pybind11`.
- `nn_der/` &mdash; build output directory for the Python extension
  `nn_der.nn_der` (`nn_der*.so`).
- `learning_scripts/` &mdash; Python control scripts for the three tasks, one
  file per (task, method) pair.
- `learning_scripts/inputs/` &mdash; initial rod geometry and target shapes
  (`vertices*.txt`, `C_initial.txt`, `M_initial.txt`, `U_initial.txt`).
- `targets/` &mdash; target trajectories / shapes used by Tasks 2 and 3.
- `common.py`, `utils.py` &mdash; shared helpers (policy network, simulator
  reset, animation, thread configuration).
  experiments back-to-back.
- `experimental_results/`, `simulation_results/` &mdash; output directories populated by the learning scripts.

***

### How to use

#### 1. Build the C++ simulator binding

The simulator must be compiled before any learning script can run. The build
follows a standard CMake + `pip install -e .` flow and produces
`nn_der/nn_der*.so`, which the Python scripts import as `nn_der.nn_der`.

System dependencies (tested on Ubuntu 20.04&ndash;24.04 with Python 3.10+):

- Eigen 3.4.0
- Intel oneAPI MKL (Pardiso + BLAS/LAPACK backend for Eigen)
- SymEngine (built with `-DWITH_LLVM=on`)
- OpenGL / GLUT (`libglu1-mesa-dev freeglut3-dev mesa-common-dev`)
- pybind11 (`pip install pybind11`)
- Python packages: `torch`, `numpy`

Build and install the Python binding:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
pip install -e .
```

#### 2. Run a single control experiment

Each script in `learning_scripts/` is self-contained and configures itself
through a top-level `CONFIG` dict at the top of the file (cases, horizon `T`,
learning rate, optimizer hyperparameters, etc.). To run a single experiment,
launch the corresponding script from the repository root, for example:

```bash
# Task 1 (any-node reaching) with the proposed Adjoint + RHC method.
python3 learning_scripts/any_node_adjoint_RHC.py

# Task 2 (middle-node trajectory tracking) with the baseline MPC.
python3 learning_scripts/middle_tracking_MPC.py

# Task 3 (shape control toward a letter target) with iCEM.
python3 learning_scripts/letter_curve_icem.py
```

Each script writes its outputs as `.txt` files next to itself in
`learning_scripts/`: a summary table (e.g.
`middle_tracking_MPC.txt`, `letter_curve_icem.txt`) with per-case timing
and best loss, plus per-case trajectories — control sequences
(`*_u.txt`), node position histories (`*_positions.txt`), or loss
histories (`*_loss.txt`) depending on the task. The `_adjoint_RHC.py`
scripts additionally pop up a matplotlib animation window of the
rollout (not saved to disk). The pre-populated `experimental_results/`
and `simulation_results/` directories are reference outputs from the
paper, not produced by these runs.

#### 3. Tasks and methods

The naming convention is `<task>_<method>.py`:

| Task prefix              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `any_node_*`             | Task 1 &mdash; drive a selected node of the elastic strip to a target.      |
| `middle_tracking_*`      | Task 2 &mdash; trace the middle node along a prescribed trajectory.         |
| `letter_curve_*`         | Task 3 &mdash; shape control toward a prescribed letter-shaped target.      |

| Method suffix            | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `*_adjoint_RHC.py`       | Proposed method: adjoint learning with receding-horizon control.            |
| `*_MPC.py`               | Adjoint-based MPC baseline (re-plans at every step, no policy).             |
| `*_noMPC.py`             | Open-loop adjoint optimization without receding-horizon control.            |
| `*_cem.py`               | Derivative-free baseline: CEM.                                              |
| `*_icem.py`              | Derivative-free baseline: iCEM.                                             |
| `*_spsa.py`              | Derivative-free baseline: SPSA.                                             |

Any of the nine (task, method) combinations above can be launched directly.

***

### Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{neuralcontrol2026,
  title     = {Neural Control: Adjoint Learning Through Equilibrium Constraints},
  author    = {Author One and Author Two and Author Three and Author Four},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
