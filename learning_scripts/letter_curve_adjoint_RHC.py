"""
Adjoint-based Receding Horizon Control for letter curve tracking task.

This script implements MPC control to deform a rod into target letter shapes (C, U, M, etc.).
"""

import os
import time
import numpy as np
import torch

import nn_der.nn_der as py_der

from utils import create_policy_model, to_3d, translate_and_rotate_segment, to_one_hot
from common import (
    configure_threads,
    set_seed,
    get_sim_states,
    reset_sim_with_state,
    reinit_net_,
    rebuild_optimizer,
    show_animation_letter_curve,
)

# =============================================================================
# Configuration - All parameters
# =============================================================================
CONFIG = {
    # List of cases: each case is a dict with "initial" and "target" file paths
    "cases": [
        {"initial": "C_initial.txt", "target": "targets/target_C.txt"},
        {"initial": "U_initial.txt", "target": "targets/target_U.txt"},
        {"initial": "M_initial.txt", "target": "targets/target_M.txt"},
    ],
    
    # MPC parameters
    "max_total_iterations": 500,   # Maximum total iterations per case
    "inner_iterations": 50,         # Inner optimization iterations per MPC step
    "learning_rate": 0.01,
    
    # Early stopping
    "patience": 5,
    "min_delta_rel": 1e-4,
    "loss_threshold": 1e-7,
    
    # Time discretization
    "T": 11,                        # Number of time steps per MPC horizon
    
    # Network parameters
    "hidden_sizes": [64, 64],
    
    # Control bounds (will be divided by dlam)
    "bounds_xy": 0.02,
    "bounds_a": 0.2,
}


# =============================================================================
# Module-level state for MPC reset
# =============================================================================
reset_state = None


def resetSim(sim_manager):
    """Reset simulator and optionally restore to saved state."""
    reset_sim_with_state(sim_manager, reset_state)


# =============================================================================
# Core: compute gradients dL/dtheta
# =============================================================================
def compute_dL_dtheta(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,                 # (T,) torch
    sim_manager,
    target: np.ndarray,                 # (2,) numpy
    dlam: float,
    jac_reg: float = 1e-6,
):
    """
    Compute gradients for letter curve tracking task with MPC.
    
    Returns
    -------
    grads_list : list[torch.Tensor]
        Gradients w.r.t. policy_model parameters.
    L_total : float
        Scalar loss value.
    buckled : bool
        Whether the rod buckled during simulation.
    u_seq : np.ndarray
        Control sequence used.
    vertices_list : list
        List of vertex states.
    """
    policy_model.eval()

    # set the target with the curvature
    kap_target = sim_manager.compute_curvature(to_3d(target))

    # ---- 0) controls with torch graph ----
    T = int(lams.numel())
    u_seq_torch = policy_model(lams.unsqueeze(-1))  # (T, 3)
    u_seq = u_seq_torch.detach().cpu().numpy()      # (T, 3)

    # --------------------------------------
    # 1) Forward rollout in simulator
    # --------------------------------------
    resetSim(sim_manager)

    verts0 = np.asarray(sim_manager.getAllVertices()).copy()
    N = verts0.shape[0]

    xb_k = verts0[[0, 1, -2, -1], :].reshape(-1).copy()

    v0_fixed = xb_k[0:2].copy() 
    v1_fixed = xb_k[2:4].copy()

    # store A_i, B_i, and dxf/dxb for adjoint
    A_list = np.zeros((T, 4, 4), dtype=np.float32)
    B_list = np.zeros((T, 4, 3), dtype=np.float32)
    # Matrix-free adjoint: store regularized G_x and -G_z per step instead of
    # forming the dense sensitivity S = G_x^{-1}(-G_z).
    Gx_list = []
    Gz_list = []
    vertices_list = []
    
    verts = np.asarray(sim_manager.getAllVertices(), dtype=np.float32)
    buckled = False

    for i in range(T):
        uk = u_seq[i]
        dx, dy, da = uk * dlam
        xb0_k = xb_k.copy()
        xf0_k = verts.reshape(-1)[4:-4].copy()  # free vertices

        v2 = xb_k[4:6]
        v3 = xb_k[6:8]
        v2_1, v3_1 = translate_and_rotate_segment(v2, v3, dx, dy, da)
        xb_k = np.hstack([v0_fixed, v1_fixed, v2_1, v3_1])

        sim_manager.setControlInputs(np.ascontiguousarray(xb_k.reshape(-1, 2), dtype=np.float64))
        sim_manager.step()

        # get states from simulator
        jac = np.asarray(sim_manager.getJacobian(), dtype=np.float32)
        verts = np.asarray(sim_manager.getAllVertices(), dtype=np.float32)

        lhs = jac[4:-4, 4:-4]
        rhs = -jac[4:-4, -4:]
        lhs_reg = lhs + jac_reg * np.eye(lhs.shape[0], dtype=np.float32)
        # Matrix-free adjoint: never form S; cache G_x (regularized) and -G_z.
        Gx_list.append(lhs_reg)
        Gz_list.append(rhs)

        # Check for buckling. The forward product S @ delta is computed as a
        # single forward solve G_x (S @ delta) = (-G_z) @ delta, without S.
        delta = xb_k[-4:] - xb0_k[-4:]
        try:
            Svd = np.linalg.solve(lhs_reg, rhs @ delta)
        except np.linalg.LinAlgError:
            Svd = np.linalg.lstsq(lhs_reg, rhs @ delta, rcond=None)[0]
        xf_try = xf0_k + Svd
        xf_k = verts.reshape(-1)[4:-4]
        e_metric = np.linalg.norm(xf_try - xf_k)
        if e_metric > 0.1 and i != 0:
            buckled = True

        # Build A, B for boundary states
        x2, y2 = v2_1
        x3, y3 = v3_1  
        a = float(uk[2])
        A = np.array([
            [0.0,   -a/2, 0.0,   a/2],
            [a/2,   0.0, -a/2,  0.0],
            [0.0,    a/2, 0.0,  -a/2],
            [-a/2,  0.0,  a/2,  0.0],
        ], dtype=np.float64)

        B = np.array([
            [1.0, 0.0, -0.5 * (y2 - y3)],
            [0.0, 1.0,  0.5 * (x2 - x3)],
            [1.0, 0.0,  0.5 * (y2 - y3)],
            [0.0, 1.0, -0.5 * (x2 - x3)],
        ], dtype=np.float64)

        A_list[i] = A
        B_list[i] = B
        vertices_list.append(verts.copy())


    # ----- 2) compute loss and its gradient w.r.t q ------
    coeff_b = np.array([[1e-3, 0.0], [0.0, 1e-3]])  # bending stiffness
    L_kap = sim_manager.compute_curvature_loss(kap_target, coeff_b)
    dkap = sim_manager.compute_dcurvature(kap_target, coeff_b)  # (N,) numpy
    dkap = to_one_hot(dkap)

    L_stretch = sim_manager.compute_stretch_loss(1.0)
    dstretch = sim_manager.compute_stretch_grad(1.0)
    dstretch = to_one_hot(dstretch)

    L_total = L_kap + L_stretch
    
    a_q = dkap + dstretch  # (N*2,) numpy
    lam_f = a_q[4:-4]
    lam_b = a_q[-4:]

    # ----- 3) Backward adjoint ------
    v_u = np.zeros((T, 3), dtype = np.float32)
    I4 = np.eye(4, dtype=np.float32)

    def StProd(i, v):
        # Matrix-free S^T v (Eqs. 31-32): solve G_x^T p = v, return -G_z^T p.
        try:
            p = np.linalg.solve(Gx_list[i].T, v)
        except np.linalg.LinAlgError:
            p = np.linalg.lstsq(Gx_list[i].T, v, rcond=None)[0]
        return Gz_list[i].T @ p

    for i in range(T-1, -1, -1):
        s_i = StProd(i, lam_f)
        v_u[i] = dlam * B_list[i].T @ lam_b + dlam * B_list[i].T @ s_i
        lam_b = (I4 + dlam * A_list[i].T) @ lam_b + dlam * A_list[i].T @ s_i

    # ---- 4) one torch VJP: grads = (du/dtheta)^T v_u ----
    v_u_torch = torch.tensor(v_u, dtype=u_seq_torch.dtype, device=u_seq_torch.device)  # (T,3)

    surrogate = (u_seq_torch * v_u_torch).sum()

    params = [p for p in policy_model.parameters() if p.requires_grad]
    grads_list = torch.autograd.grad(surrogate, params, retain_graph=False, create_graph=False)

    return grads_list, L_total, buckled, u_seq, vertices_list


def run_single_case(
    sim_manager,
    initial_file: str,
    target_file: str,
    config: dict,
    device: torch.device,
    case_idx: int,
):
    """
    Run MPC training for a single case.
    
    Returns
    -------
    result : dict
        Contains total_time, best_loss, best_u, target, mpc_step_vertices, etc.
    """
    global reset_state
    
    # Reconfigure simulator with new geometry file
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
        "geometry_file": initial_file,
        "d_h": 0.001,
        "col_limit": 0.01,
        "k_scaler": 1.0,
    })

    # Controller : BC controller
    controller_type = [0, 0, 0, 0]
    control_dofs = [0, 1, 99, 100]
    control_info = np.array([controller_type, control_dofs]).T
    sim_manager.defineController(control_info)
    sim_manager.resetSim()
    
    # Initialize reset_state to None for fresh start
    reset_state = None

    # Load target
    target = np.loadtxt(target_file)
    target = target.reshape(-1)

    # Training setup
    T = config["T"]
    max_total_iterations = config["max_total_iterations"]
    inner_iterations = config["inner_iterations"]
    learning_rate = config["learning_rate"]
    hidden_sizes = config["hidden_sizes"]
    bounds_xy = config["bounds_xy"]
    bounds_a = config["bounds_a"]
    patience = config["patience"]
    min_delta_rel = config["min_delta_rel"]
    loss_threshold = config["loss_threshold"]
    
    # Collect u and vertices at end of each MPC step
    mpc_step_u = []  # u[0] from each MPC step
    mpc_step_vertices = []

    lams_np = np.linspace(0.0, 1.0, T).astype(np.float32)
    lams = torch.tensor(lams_np, device=device, requires_grad=True)
    dlam = float(lams_np[1] - lams_np[0])

    bounds = torch.tensor([bounds_xy/dlam, bounds_xy/dlam, bounds_a/dlam], dtype=torch.float32)

    net = create_policy_model(
        input_size=1,
        hidden_sizes=hidden_sizes,
        output_size=3,
        bounds=bounds,
    ).to(device)

    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

    # MPC Training loop
    best_loss = float('inf')
    epoch_dt_hist = []

    start_time = time.perf_counter()
    
    mpc_step = 0
    total_iterations = 0
    while total_iterations < max_total_iterations and best_loss > loss_threshold:
        t0 = time.perf_counter()
        
        # Save current state for MPC horizon
        reset_state = get_sim_states(sim_manager)
        
        # Reinitialize network and optimizer for new MPC step
        reinit_net_(net)
        optimizer = rebuild_optimizer(optimizer, net)
        
        best_so_far = float('inf')
        stale_steps = 0
        buckled = False
        early_stop = False
        iter_inner = 0
        
        while (iter_inner <= inner_iterations or buckled) and total_iterations < max_total_iterations:
            optimizer.zero_grad(set_to_none=True)
            
            grads_list, loss, buckled, u_seq, vertices_list = compute_dL_dtheta(
                net,
                lams,
                sim_manager,
                target,
                dlam,
            )
            
            loss_val = float(loss)
            improve = (best_so_far - loss_val) / max(abs(best_so_far), 1e-12)
            if loss_val < best_so_far:
                best_so_far = loss_val
            
            if improve < min_delta_rel:
                stale_steps += 1
            else:
                stale_steps = 0
            
            if stale_steps >= patience:
                early_stop = True
            
            params = [p for p in net.parameters() if p.requires_grad]
            for p, g in zip(params, grads_list):
                p.grad = g.detach()
            
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            if loss_val < best_loss:
                best_loss = loss_val
            
            grad_norm = float(torch.sqrt(sum((g.detach()**2).sum() for g in grads_list)).cpu())
            print(f"Case {case_idx:02d} | MPC {mpc_step:03d} | iter {iter_inner:03d} | Loss {loss_val:.6e} | grad_norm {grad_norm:.3e} | buckled: {buckled}")
            
            if early_stop:
                break
            iter_inner += 1
            total_iterations += 1
        
        epoch_dt = time.perf_counter() - t0
        epoch_dt_hist.append(epoch_dt)
        
        # Record current state at end of this MPC step
        current_verts = np.asarray(sim_manager.getAllVertices()).copy()
        mpc_step_vertices.append(current_verts)
        mpc_step_u.append(u_seq[0].copy())  # Only first control input is executed
        
        print(f"\n[Case {case_idx}] MPC step {mpc_step} completed. Best loss so far: {best_loss:.6e}\n")
        mpc_step += 1

    total_time = time.perf_counter() - start_time
    avg_mpc_time = np.mean(epoch_dt_hist) if epoch_dt_hist else 0.0

    return {
        "initial_file": initial_file,
        "target_file": target_file,
        "total_time": total_time,
        "best_loss": best_loss,
        "total_mpc_steps": mpc_step,
        "avg_mpc_step_time": avg_mpc_time,
        "mpc_step_u": mpc_step_u,
        "target": target,
        "mpc_step_vertices": mpc_step_vertices,
    }


if __name__ == "__main__":
    configure_threads(num_threads=1)
    set_seed(1234, deterministic=True)

    device = torch.device("cpu")

    # Load configuration
    cases = CONFIG["cases"]
    
    # Create simulator
    sim_manager = py_der.SimulationManager()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*70}")
    print(f"Running {len(cases)} cases with MPC")
    print(f"{'='*70}\n")

    for case_idx, case in enumerate(cases):
        # Reset seed for each case to ensure fair comparison
        set_seed(1234)
        
        initial_file = case["initial"]
        target_file = case["target"]

        print(f"\n{'='*70}")
        print(f"[{case_idx+1}/{len(cases)}] Case: {os.path.basename(initial_file)} -> {os.path.basename(target_file)}")
        print(f"{'='*70}\n")

        result = run_single_case(
            sim_manager,
            initial_file,
            target_file,
            CONFIG,
            device,
            case_idx,
        )

        # Print optimal loss and total time
        print(f"\n{'='*70}")
        print(f"[Case {case_idx}] Completed!")
        print(f"  Best Loss: {result['best_loss']:.6e}")
        print(f"  Total Time: {result['total_time']:.4f} s")
        print(f"{'='*70}")
        
        # Save MPC control sequence to txt file (one u per MPC step)
        init_name = os.path.basename(result['initial_file']).replace('.txt', '')
        u_file = os.path.join(script_dir, f"letter_curve_case{case_idx}_{init_name}_u.txt")
        with open(u_file, "w") as f:
            f.write(f"# Control sequence for letter_curve tracking (one u per MPC step)\n")
            f.write(f"# Case {case_idx}: {result['initial_file']} -> {result['target_file']}\n")
            f.write(f"# Best Loss: {result['best_loss']:.10e}\n")
            f.write(f"# Total Time: {result['total_time']:.4f} s\n")
            f.write(f"# Format: mpc_step, u_x, u_y, u_a\n")
            for step, u in enumerate(result['mpc_step_u']):
                f.write(f"{step}, {u[0]:.10e}, {u[1]:.10e}, {u[2]:.10e}\n")
        print(f"Control sequence saved to: {u_file}")
        
        # Show animation (one frame per MPC step)
        if result['mpc_step_vertices']:
            show_animation_letter_curve(
                result['mpc_step_vertices'],
                result['target'],
            )
