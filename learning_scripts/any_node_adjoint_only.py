import os, math, copy, time, random, sys
import numpy as np
import torch

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_policy_model


# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # Multiple cases: each case is (target_index, target_position)
    # Each case will be run independently from scratch
    "cases": [
        {"target_index": 20, "target_position": [0.2, 0.2]},
        {"target_index": 40, "target_position": [0.2, 0.2]},
        {"target_index": 60, "target_position": [-0.05, 0.1]},
        {"target_index": 80, "target_position": [-0.05, 0.1]},
    ],
    
    # Optimization parameters
    "T": 101,                       # Number of time steps
    "learning_rate": 0.01,
    "iteration_number": 2,        # Stop after this many iterations
    "loss_threshold": 1e-20,
    
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


def get_sim_states(sim_manager):
    return {
        "vertices": np.asarray(sim_manager.getAllVertices()).copy(),
        "frames":   np.asarray(sim_manager.get_all_frames()).copy(),
    }


def set_sim_states(sim_manager, state):
    sim_manager.set_all_vertices(np.ascontiguousarray(state["vertices"], dtype=np.float64).reshape(-1))
    sim_manager.set_all_frames(np.ascontiguousarray(state["frames"], dtype=np.float64))


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
    Compute gradients for point-to-point reaching task.
    
    Parameters
    ----------
    target : np.ndarray
        Shape (2,) - target position for the tracked node
    target_index : int
        Index of the node to control
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

    # Compute loss (only at final step)
    final_vertices_flat = vertices_list[-1]
    v_f = final_vertices_flat.reshape(-1, 2)[target_index]
    dv = v_f - target
    L_total = 0.5 * float(dv @ dv)

    if not compute_grads:
        return None, L_total, vertices_list

    # Adjoint sensitivity
    a_q = np.zeros((2 * N,), dtype=np.float64)
    a_q[2 * target_index : 2 * target_index + 2] = dv

    lam_f = a_q[4:-4]
    lam_b = np.concatenate([a_q[:4], a_q[-4:]])

    v_u = np.zeros((T, 2), dtype=np.float64)
    I8 = np.eye(8, dtype=np.float64)

    def StProd(i, v):
        # Matrix-free S^T v (Eqs. 31-32): solve G_x^T p = v, return -G_z^T p.
        try:
            p = np.linalg.solve(Gx_list[i].T, v)
        except np.linalg.LinAlgError:
            p = np.linalg.lstsq(Gx_list[i].T, v, rcond=None)[0]
        return Gz_list[i].T @ p

    for i in range(T - 1, -1, -1):
        A = A_list[i]
        B = B_list[i]

        s_i = StProd(i, lam_f)
        v_u[i] = dlam * (B.T @ lam_b) + dlam * (B.T @ s_i)
        lam_b = (I8 + dlam * A.T) @ lam_b + dlam * (A.T @ s_i)

    # Torch VJP
    v_u_torch = torch.tensor(v_u, dtype=u_seq_torch.dtype, device=u_seq_torch.device)
    surrogate = (u_seq_torch * v_u_torch).sum()

    params = [p for p in policy_model.parameters() if p.requires_grad]
    grads_list = torch.autograd.grad(
        surrogate, params, retain_graph=False, create_graph=False, allow_unused=False
    )

    return grads_list, L_total, vertices_list


import torch.nn as nn

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
        "maxIter": 10000,
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
    cases = CONFIG["cases"]
    T = CONFIG["T"]
    learning_rate = CONFIG["learning_rate"]
    iteration_number = CONFIG["iteration_number"]
    loss_threshold = CONFIG["loss_threshold"]
    hidden_sizes = CONFIG["hidden_sizes"]

    # Time discretization
    lams_np = np.linspace(0, 1, T).astype(np.float32)
    lams = torch.tensor(lams_np, dtype=torch.float32, device=device)
    dlam = float(lams_np[1] - lams_np[0])

    bounds = torch.tensor([0.05 / dlam, 0.05 / dlam], dtype=torch.float32)

    # Store results for all cases
    all_case_results = []

    # Run each case independently
    for case_idx, case in enumerate(cases):
        target_index = case["target_index"]
        target = np.array(case["target_position"], dtype=np.float64)
        
        # Print configuration for this case
        print(f"\n{'='*60}")
        print(f"Case {case_idx + 1}/{len(cases)}:")
        print(f"  Target node index: {target_index}")
        print(f"  Target position: {target.tolist()}")
        print(f"  Number of time steps: {T}")
        print(f"  Number of iterations: {iteration_number}")
        print(f"{'='*60}\n")

        # Reset seed for reproducibility (each case starts fresh)
        set_seed(42)

        # Create fresh network and optimizer for each case
        net = create_policy_model(
            input_size=1,
            hidden_sizes=hidden_sizes,
            output_size=2,
            bounds=bounds,
        ).to(device)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

        loss_hist = []
        epoch_dt_hist = []

        best_loss = float("inf")
        best_state = None

        # Training loop for this case
        case_start_time = time.perf_counter()
        
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
            print(f"[Case {case_idx+1}] Epoch {epoch:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e} | dt {epoch_dt*1e3:.1f} ms")

            if loss_val < loss_threshold:
                print(f"\n[Case {case_idx+1}] Reached loss threshold at epoch {epoch}")
                break

        case_total_time = time.perf_counter() - case_start_time
        case_avg_epoch_time = np.mean(epoch_dt_hist)
        
        print(f"\n[Case {case_idx+1}] Training completed!")
        print(f"  Total time: {case_total_time:.3f} s")
        print(f"  Best loss: {best_loss:.6e}")
        print(f"  Average epoch time: {case_avg_epoch_time*1e3:.3f} ms")

        # Store results for this case
        all_case_results.append({
            "case_idx": case_idx,
            "target_index": target_index,
            "target_position": target.tolist(),
            "best_loss": best_loss,
            "total_time": case_total_time,
            "avg_epoch_time": case_avg_epoch_time,
            "best_epoch": best_state["epoch"] if best_state else -1,
            "total_epochs": len(loss_hist),
            "loss_history": loss_hist.copy(),
        })

    # Compute average statistics across all cases
    avg_best_loss = np.mean([r["best_loss"] for r in all_case_results])
    avg_total_time = np.mean([r["total_time"] for r in all_case_results])
    avg_epoch_time_all = np.mean([r["avg_epoch_time"] for r in all_case_results])

    print(f"\n{'='*60}")
    print(f"All Cases Summary:")
    print(f"{'='*60}")
    for r in all_case_results:
        print(f"  Case {r['case_idx']+1}: node={r['target_index']}, best_loss={r['best_loss']:.6e}, time={r['total_time']:.3f}s")
    print(f"{'='*60}")
    print(f"Average Statistics:")
    print(f"  Average best loss: {avg_best_loss:.6e}")
    print(f"  Average total time: {avg_total_time:.3f} s")
    print(f"  Average epoch time: {avg_epoch_time_all*1e3:.3f} ms")
    print(f"{'='*60}\n")

    # Save result to current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "any_node_noMPC.txt")
    
    # Save summary to txt file
    with open(output_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Training Results Summary (Multiple Cases)\n")
        f.write("="*60 + "\n\n")
        f.write("Global Configuration:\n")
        f.write(f"  Number of time steps: {T}\n")
        f.write(f"  Iteration number: {iteration_number}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Hidden sizes: {hidden_sizes}\n\n")
        
        f.write("Individual Case Results:\n")
        f.write("-"*60 + "\n")
        for r in all_case_results:
            f.write(f"  Case {r['case_idx']+1}:\n")
            f.write(f"    Target node index: {r['target_index']}\n")
            f.write(f"    Target position: {r['target_position']}\n")
            f.write(f"    Best loss: {r['best_loss']:.10e}\n")
            f.write(f"    Best epoch: {r['best_epoch']}\n")
            f.write(f"    Total time: {r['total_time']:.6f} s\n")
            f.write(f"    Average epoch time: {r['avg_epoch_time']*1e3:.3f} ms\n")
            f.write(f"    Total epochs run: {r['total_epochs']}\n")
            f.write("-"*60 + "\n")
        
        f.write("\nAverage Statistics:\n")
        f.write(f"  Average best loss: {avg_best_loss:.10e}\n")
        f.write(f"  Average total time: {avg_total_time:.6f} s\n")
        f.write(f"  Average epoch time: {avg_epoch_time_all*1e3:.3f} ms\n")
        f.write("="*60 + "\n")
    
    # Save per-step loss for each case to txt files
    for r in all_case_results:
        case_idx = r['case_idx']
        target_index = r['target_index']
        loss_hist = r['loss_history']
        loss_file = os.path.join(script_dir, f"any_node_noMPC_case{case_idx}_node{target_index}_loss.txt")
        with open(loss_file, "w") as f:
            f.write("# Per-step loss history for any_node_noMPC\n")
            f.write(f"# Case {case_idx}: Node {target_index} -> {r['target_position']}\n")
            f.write("# Step, Loss\n")
            for step, loss_val in enumerate(loss_hist):
                f.write(f"{step}, {loss_val:.10e}\n")
        print(f"Loss history saved to: {loss_file}")
