## [Neural Control: Adjoint Learning Through Equilibrium Constraints](https://github.com/StructuresComp/neural-control/index.html)

We present Neural Control, an adjoint-based learning framework
for controlling soft, deformable structures whose dynamics are governed by static
equilibrium constraints. By differentiating through the equilibrium via implicit
differentiation and pairing it with receding-horizon control, our method avoids
unrolling expensive forward simulations and scales to high-dimensional shape
objectives. We validate the approach on three representative tasks &mdash; node
targeting, trajectory tracing, and shape control &mdash; where it achieves
orders-of-magnitude lower tracking error at a fraction of the compute cost of
derivative-free baselines.

Go to our [website](https://structurescomp.github.io/neural-control/) for more details.

The simulator is exposed to Python through
`pybind11` as the module `nn_der`. The simulation codes are in `src/` and the learning scripts in
`learning_scripts/`. These python scripts use different method to solve the gradient for the three
control tasks reported in the paper.

***

### How to use

#### 1. Build the C++ simulator binding

The simulator must be compiled before any learning script can run. The build
follows a standard CMake + `pip install -e .` flow and produces
`nn_der/nn_der*.so`, which the Python scripts import as `nn_der.nn_der`.

System dependencies (recommend Ubuntu 20.04 with Python 3.11):

- Eigen 3.4.0
- Intel oneAPI MKL (Pardiso + BLAS/LAPACK backend for Eigen)
- SymEngin
- OpenGL / GLUT
- pybind11
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
`learning_scripts/`: a summary table, node position histories (`*_positions.txt`), or loss
histories (`*_loss.txt`) depending on the task. The `_adjoint_RHC.py`
scripts additionally pop up a matplotlib animation window of the
rollout. 

The pre-populated `experimental_results/`
and `simulation_results/` directories are reference outputs from the
paper.

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
  author    = {Dezhong Tong, Jiawen Wang, Hengyi Zhou, Yinlong Shen, Xiaonan Huang, M. Khalid Jawed},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
