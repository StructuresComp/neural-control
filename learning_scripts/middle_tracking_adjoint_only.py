import os, math, copy, time, random, sys
import numpy as np
import torch
import torch.nn as nn

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_policy_model
from trajectory import generate_trajectory, get_trajectory_description


# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # Trajectory types to test: 'sin', 'cos', 'triangle', 'semicircle', 'square'
    "trajectory_types": ["square", "cos", "triangle", "semicircle"],
    
    # Trajectory-specific parameters
    "trajectory_params": {
        # For sin/cos trajectories
        "amplitude": 0.05,          # Wave amplitude
        "frequency": 3.0,           # Wave frequency (number of cycles)
        
        # For triangle wave
        "period": 0.5,              # Period of triangle wave
        
        # For semicircle
        "radius": 0.25,             # Radius of semicircle
        "direction": "down",        # 'up' or 'down'
        
        # For square wave
        "square_amplitude": 0.12,   # Amplitude of square wave
        "num_segments": 10,         # Number of segments
    },
    
    # Target node index (which node to track)
    "target_index": 50,
    
    # Optimization parameters
    "T": 101,                       # Number of time steps
    "learning_rate": 0.01,
    "iteration_number": 1000,         # Stop after this many iterations
    "loss_threshold": 1e-7,
    
    # Network parameters
    "hidden_sizes": [64, 64],
}


# =============================================================================
# Thread safety / stability
# =============================================================================
def configure_threads(num_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# =============================================================================
# Simulator helper functions
# =============================================================================
def resetSim(sim_manager):
    sim_manager.resetSim()


# =============================================================================
# Core: compute gradients dL/dtheta
# =============================================================================
def compute_dL_dtheta(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,
    sim_manager,
    target: np.ndarray,
    target_index: int,
    dlam: float,
    jac_reg: float = 1e-6,
    compute_grads: bool = True,
):
    """
    Compute gradients for trajectory tracking task.
    
    Parameters
    ----------
    target : np.ndarray
        Shape (T, 2) - target trajectory for the tracked node
    target_index : int
        Index of the node to track
    compute_grads : bool
        If True, compute gradients. If False, only do forward rollout.
    
    Returns
    -------
    grads_list : list[torch.Tensor] or None
        Gradients w.r.t. policy_model parameters. None if compute_grads=False.
    L_total : float
        Scalar loss value.
    vertices_list : list[np.ndarray]
        List of vertices at each time step.
    """
    policy_model.eval()

    # Query policy for control sequence
    T = int(lams.numel())
    u_seq_torch = policy_model(lams.view(T, 1))
    u_seq = u_seq_torch.detach().cpu().numpy()

    # Forward rollout in simulator
    resetSim(sim_manager)

    verts0 = np.asarray(sim_manager.getAllVertices()).copy()
    verts0_xy = verts0[:, :2]
    N = verts0_xy.shape[0]

    xb_k = verts0_xy[[0, 1, -2, -1], :].reshape(-1).copy()

    # Pre-allocate lists for adjoint
    A_list = np.zeros((T, 8, 8), dtype=np.float64)
    B_list = np.zeros((T, 8, 2), dtype=np.float64)
    # Matrix-free adjoint: store regularized G_x and -G_z per step instead of
    # forming the dense sensitivity S = G_x^{-1}(-G_z).
    Gx_list = []
    Gz_list = []
    vertices_list = []

    for i in range(T):
        uk = u_seq[i]
        dx1, dx2 = uk * dlam

        v0 = xb_k[:2].copy()
        v1 = xb_k[2:4].copy()
        v2 = xb_k[4:6].copy()
        v3 = xb_k[6:8].copy()

        v0[0] += dx1
        v1[0] += dx1
        v2[0] += dx2
        v3[0] += dx2

        xb_k = np.hstack((v0, v1, v2, v3))

        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        sim_manager.step()

        jac = np.asarray(sim_manager.getJacobian()).copy()
        verts_xy = np.asarray(sim_manager.getAllVertices()).copy()[:, :2]
        vertices_flat = verts_xy.reshape(-1)

        lhs = jac[4:-4, 4:-4]
        rhs = -np.hstack((jac[4:-4, :4], jac[4:-4, -4:]))

        lhs_reg = lhs + jac_reg * np.eye(lhs.shape[0], dtype=np.float64)

        # Matrix-free adjoint: never form S; cache G_x (regularized) and -G_z.
        Gx_list.append(lhs_reg)
        Gz_list.append(rhs)

        A = np.zeros((8, 8), dtype=np.float64)
        B = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
        ], dtype=np.float64)

        A_list[i] = A
        B_list[i] = B.T

        vertices_list.append(vertices_flat.copy())

    # Compute loss (tracking loss over all steps)
    a_q_array = np.zeros((T, vertices_list[0].shape[0]), dtype=np.float64)
    L_total = 0.0
    for i in range(T):
        v_i = vertices_list[i].reshape(-1, 2)[target_index]
        dv = v_i - target[i]
        L_total += 0.5 * (dv @ dv) * dlam
        a_q = np.zeros((2 * N,), dtype=np.float64)
        a_q[2 * target_index : 2 * target_index + 2] = dv
        a_q_array[i] = a_q.flatten()

    if not compute_grads:
        return None, L_total, vertices_list

    # Backward adjoint: single sweep with proxy costates (Algorithm 1 / Eqs. 40-44).
    # Running loss is accumulated into the free/boundary costates a, g; one
    # matrix-free S^T solve per step. A_list is zero here, so the (I + dlam A^T)
    # propagation and the dlam A^T s cross-term vanish (g is a pure accumulator).
    def StProd(i, v):
        # Matrix-free S^T v (Eqs. 31-32): solve G_x^T p = v, return -G_z^T p.
        try:
            p = np.linalg.solve(Gx_list[i].T, v)
        except np.linalg.LinAlgError:
            p = np.linalg.lstsq(Gx_list[i].T, v, rcond=None)[0]
        return Gz_list[i].T @ p

    v_u = np.zeros((T, 2), dtype=np.float64)
    a = np.zeros(2 * N - 8, dtype=np.float64)   # free costate
    g = np.zeros(8, dtype=np.float64)           # boundary costate

    for i in range(T - 1, -1, -1):
        a_q = a_q_array[i] * dlam
        a = a + a_q[4:-4]                                   # accumulate grad_x ell_i (Eq. 40)
        g = g + np.concatenate([a_q[:4], a_q[-4:]])         # accumulate grad_z ell_i (Eq. 41; A=0)
        s_i = StProd(i, a)                                  # S_i^T a (one solve per step)
        v_u[i] = dlam * (B_list[i].T @ (g + s_i))           # Eqs. 42-43

    v_u_torch = torch.tensor(v_u, dtype=u_seq_torch.dtype, device=u_seq_torch.device)
    surrogate = (u_seq_torch * v_u_torch).sum()

    # Torch VJP
    params = [p for p in policy_model.parameters() if p.requires_grad]
    grads_list = torch.autograd.grad(
        surrogate, params, retain_graph=False, create_graph=False, allow_unused=False
    )

    return grads_list, L_total, vertices_list


def reinit_net_(net: nn.Module):
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
    net.apply(_init)

    with torch.no_grad():
        for name in ["log_mag", "log_mag_xy", "log_mag_a", "rho_xy", "rho_a", "log_metric"]:
            if hasattr(net, name):
                getattr(net, name).zero_()


if __name__ == "__main__":
    configure_threads(1)
    set_seed(42)
    device = torch.device("cpu")

    # Simulator setup
    sim_manager = py_der.SimulationManager()
    sim_manager.configure({
        "youngM": 1e5,
        "Poisson": 0.5,
        "density": 1000,
        "deltaTime": 0.01,
        "totalTime": 10.0,
        "gVector": np.array([0, 0, -0.0]),
        "viscosity": 0.000,
        "tol": 1e-4,
        "maxIter": 5000,
        "stol": 1e-4,
        "rodRadius": 1e-3,
        "geometry_file": "vertices.txt",
        "d_h": 0.001,
        "col_limit": 0.01,
        "k_scaler": 1.0,
    })

    controller_type = [0, 0, 0, 0]
    control_dofs = [0, 1, 99, 100]
    control_info = np.array([controller_type, control_dofs]).T
    sim_manager.defineController(control_info)
    resetSim(sim_manager)

    verts_init = np.asarray(sim_manager.getAllVertices()).copy()
    N = verts_init.shape[0]

    # Load configuration parameters
    trajectory_types = CONFIG["trajectory_types"]
    trajectory_params = CONFIG["trajectory_params"]
    target_index = CONFIG["target_index"]
    T = CONFIG["T"]
    learning_rate = CONFIG["learning_rate"]
    iteration_number = CONFIG["iteration_number"]
    loss_threshold = CONFIG["loss_threshold"]
    hidden_sizes = CONFIG["hidden_sizes"]

    # Time discretization
    lams_np = np.linspace(0, 1, T).astype(np.float32)
    lams = torch.tensor(lams_np, dtype=torch.float32, device=device)
    dlam = float(lams_np[1] - lams_np[0])

    bounds = torch.tensor([0.1 / dlam, 0.1 / dlam], dtype=torch.float32)

    # Store results for all trajectories
    all_results = []

    print(f"\n{'='*60}")
    print(f"Testing {len(trajectory_types)} trajectory types")
    print(f"{'='*60}\n")

    for traj_idx, trajectory_type in enumerate(trajectory_types):
        # Reset seed for each trajectory to ensure fair comparison
        set_seed(42)
        
        # Generate target trajectory
        middle_node = verts_init[target_index, :].copy()
        target = generate_trajectory(trajectory_type, middle_node, T, trajectory_params)
        traj_desc = get_trajectory_description(trajectory_type, trajectory_params)

        # Print configuration
        print(f"\n{'='*60}")
        print(f"[{traj_idx+1}/{len(trajectory_types)}] Trajectory: {traj_desc}")
        print(f"  Target node index: {target_index}")
        print(f"  Number of time steps: {T}")
        print(f"  Number of iterations: {iteration_number}")
        print(f"{'='*60}\n")

        # Create fresh network and optimizer for each trajectory
        net = create_policy_model(
            input_size=1,
            hidden_sizes=hidden_sizes,
            output_size=2,
            bounds=bounds,
        ).to(device)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

        loss_hist = []
        epoch_dt_hist = []
        position_history = []  # Store middle point position at each epoch

        best_loss = float("inf")
        best_state = None

        # Training loop
        total_start_time = time.perf_counter()

        for epoch in range(iteration_number):
            t0 = time.perf_counter()

            optimizer.zero_grad()

            grads_list, loss, vertices_list = compute_dL_dtheta(
                net,
                lams,
                sim_manager,
                target,
                target_index,
                dlam,
            )

            params = [p for p in net.parameters() if p.requires_grad]
            for p, g in zip(params, grads_list):
                p.grad = g.detach()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            loss_val = float(loss)
            loss_hist.append(loss_val)
            
            # Collect middle point positions at each time step for this epoch
            epoch_positions = []
            for v_flat in vertices_list:
                v_xy = v_flat.reshape(-1, 2)
                epoch_positions.append(v_xy[target_index].tolist())
            position_history.append(epoch_positions)

            epoch_dt = time.perf_counter() - t0
            epoch_dt_hist.append(epoch_dt)

            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": copy.deepcopy(net.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "lams": lams_np.copy(),
                    "target": target.copy(),
                    "target_index": target_index,
                }

            grad_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_list)).cpu())
            print(f"Epoch {epoch:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e} | dt {epoch_dt*1e3:.1f} ms")

            if loss_val < loss_threshold:
                print(f"\nReached loss threshold at epoch {epoch}")
                break

        total_time = time.perf_counter() - total_start_time
        avg_epoch_time = np.mean(epoch_dt_hist)

        print(f"\n{'='*60}")
        print(f"[{trajectory_type}] Training completed!")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  Best loss: {best_loss:.6e}")
        print(f"  Average epoch time: {avg_epoch_time*1e3:.3f} ms")
        print(f"{'='*60}\n")

        # Store results
        all_results.append({
            "trajectory_type": trajectory_type,
            "trajectory_desc": traj_desc,
            "total_time": total_time,
            "best_loss": best_loss,
            "best_epoch": best_state['epoch'] if best_state else -1,
            "avg_epoch_time": avg_epoch_time,
            "total_epochs": len(loss_hist),
            "position_history": position_history,
        })

    # Save all results to current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "middle_tracking_noMPC.txt")

    # Save summary to txt file
    with open(output_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Summary Table:\n")
        f.write("="*60 + "\n")
        f.write(f"{'Trajectory':<20} {'Total Time (s)':<15} {'Best Loss':<15} {'Avg Epoch (ms)':<15}\n")
        f.write("-"*65 + "\n")
        for result in all_results:
            f.write(f"{result['trajectory_type']:<20} {result['total_time']:<15.4f} {result['best_loss']:<15.6e} {result['avg_epoch_time']*1e3:<15.3f}\n")
        f.write("-"*65 + "\n")
        
        # Compute averages
        avg_total_time = np.mean([r['total_time'] for r in all_results])
        avg_best_loss = np.mean([r['best_loss'] for r in all_results])
        avg_epoch_time_all = np.mean([r['avg_epoch_time'] for r in all_results])
        
        f.write(f"{'AVERAGE':<20} {avg_total_time:<15.4f} {avg_best_loss:<15.6e} {avg_epoch_time_all*1e3:<15.3f}\n")
        f.write("="*60 + "\n")

    print(f"\n{'='*60}")
    print(f"Overall Averages:")
    print(f"  Average total time: {avg_total_time:.4f} s")
    print(f"  Average best loss: {avg_best_loss:.6e}")
    print(f"  Average epoch time: {avg_epoch_time_all*1e3:.3f} ms")
    print(f"{'='*60}")
    print(f"\nAll results saved to {output_file}")
    
    # Save per-step middle point positions for each case to txt files
    for traj_idx, result in enumerate(all_results):
        traj_type = result['trajectory_type']
        position_hist = result['position_history']
        pos_file = os.path.join(script_dir, f"middle_tracking_noMPC_case{traj_idx}_{traj_type}_positions.txt")
        with open(pos_file, "w") as f:
            f.write("# Per-step middle point position history for middle_tracking_noMPC\n")
            f.write(f"# Case {traj_idx}: Trajectory type = {traj_type}\n")
            f.write(f"# Target node index: {target_index}\n")
            f.write("# Format: Epoch, TimeStep, X, Y\n")
            for epoch_idx, epoch_positions in enumerate(position_hist):
                for step_idx, (x, y) in enumerate(epoch_positions):
                    f.write(f"{epoch_idx}, {step_idx}, {x:.10e}, {y:.10e}\n")
        print(f"Position history saved to: {pos_file}")

