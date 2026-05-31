import os, math, copy, time, sys
import numpy as np
import torch
import torch.nn as nn

import nn_der.nn_der as py_der

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_policy_model
from utils import to_3d, translate_and_rotate_segment, to_one_hot


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
    
    # Training parameters
    "max_epochs": 2,             # Maximum epochs per case
    "learning_rate": 0.01,
    
    # Time discretization
    "T": 21,                        # Number of time steps
    
    # Network parameters
    "hidden_sizes": [64, 64],
    
    # Control bounds (will be divided by dlam)
    "bounds_xy": 0.05,
    "bounds_a": 0.3,
}

# =============================================================================
# Thread safety / stability (avoid BLAS oversubscription + nondeterminism)
# =============================================================================
def configure_threads(num_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True avoids hard crash if some ops are nondeterministic
        torch.use_deterministic_algorithms(True, warn_only=True)

# =============================================================================
# Small helpers for gradient check
# =============================================================================
def parameters_to_vector(params):
    """Flatten a list of parameters into one 1D tensor."""
    return torch.cat([p.detach().flatten() for p in params])


def grads_to_vector(grads_list):
    """Flatten a list of grads into one 1D tensor."""
    return torch.cat([g.detach().flatten() for g in grads_list])


def reinit_net_(net: nn.Module):
    """Reinitialize network weights."""
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
    net.apply(_init)

    with torch.no_grad():
        for name in ["log_mag", "log_mag_xy", "log_mag_a", "rho_xy", "rho_a", "log_metric"]:
            if hasattr(net, name):
                getattr(net, name).zero_()


def compute_dL_dtheta(
    policy_model: torch.nn.Module,
    lams: torch.Tensor,                 # (T,) torch
    sim_manager,
    target: np.ndarray,                 # (2,) numpy
    dlam: float,
    jac_reg: float = 1e-6,
):
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
    sim_manager.resetSim()

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

    for i in range(T):
        uk = u_seq[i]
        dx, dy, da = uk * dlam

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

    return grads_list, L_total


def run_single_case(
    sim_manager,
    initial_file: str,
    target_file: str,
    config: dict,
    device: torch.device,
    case_idx: int,
):
    """
    Run training for a single case.
    
    Returns
    -------
    result : dict
        Contains total_time, best_loss, total_epochs, etc.
    """
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

    # Load target
    target = np.loadtxt(target_file)
    target = target.reshape(-1)

    # Training setup
    T = config["T"]
    max_epochs = config["max_epochs"]
    learning_rate = config["learning_rate"]
    hidden_sizes = config["hidden_sizes"]
    bounds_xy = config["bounds_xy"]
    bounds_a = config["bounds_a"]

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

    # Training loop
    best_loss = float('inf')
    best_state = None
    loss_hist = []

    start_time = time.perf_counter()

    for epoch in range(max_epochs):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        grads_list, loss = compute_dL_dtheta(
            net,
            lams,
            sim_manager,
            target,
            dlam,
        )

        params = [p for p in net.parameters() if p.requires_grad]
        for p, g in zip(params, grads_list):
            p.grad = g.detach()
        
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        loss_val = float(loss)
        loss_hist.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                "epoch": epoch,
                "best_loss": loss_val,
                "model_state_dict": copy.deepcopy(net.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "lams": lams_np.copy(),
            }

        epoch_dt = time.perf_counter() - t0
        grad_norm = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_list)).cpu())
        print(f"Case {case_idx:02d} | Epoch {epoch:04d} | Loss: {loss_val:.6e} | Grad Norm: {grad_norm:.6e} | Time: {epoch_dt:.3f}s")

    total_time = time.perf_counter() - start_time

    return {
        "initial_file": initial_file,
        "target_file": target_file,
        "total_time": total_time,
        "best_loss": best_loss,
        "total_epochs": max_epochs,
        "avg_epoch_time": total_time / max_epochs if max_epochs > 0 else 0.0,
        "loss_history": loss_hist,
    }


if __name__ == "__main__":
    configure_threads(num_threads=1)
    set_seed(1234, deterministic=True)

    device = torch.device("cpu")

    # Load configuration
    cases = CONFIG["cases"]
    
    # Create simulator
    sim_manager = py_der.SimulationManager()

    # Store results for all cases
    all_results = []

    print(f"\n{'='*70}")
    print(f"Running {len(cases)} cases")
    print(f"Max epochs per case: {CONFIG['max_epochs']}")
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

        all_results.append(result)

        print(f"\n[Case {case_idx}] Completed!")
        print(f"  Total time: {result['total_time']:.3f} s")
        print(f"  Best loss: {result['best_loss']:.6e}")
        print(f"  Avg epoch time: {result['avg_epoch_time']:.3f} s")

    # Compute averages
    avg_total_time = np.mean([r['total_time'] for r in all_results])
    avg_best_loss = np.mean([r['best_loss'] for r in all_results])
    avg_epoch_time = np.mean([r['avg_epoch_time'] for r in all_results])

    # Save results to txt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "letter_curve_noMPC.txt")

    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Summary Table (Letter Curve noMPC Method):\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration: Max epochs={CONFIG['max_epochs']}, T={CONFIG['T']}, LR={CONFIG['learning_rate']}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Case':<6} {'Initial':<25} {'Target':<25} {'Time (s)':<12} {'Best Loss':<15}\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(all_results):
            init_name = os.path.basename(result['initial_file'])
            target_name = os.path.basename(result['target_file'])
            f.write(f"{i:<6} {init_name:<25} {target_name:<25} {result['total_time']:<12.4f} {result['best_loss']:<15.6e}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'AVG':<6} {'':<25} {'':<25} {avg_total_time:<12.4f} {avg_best_loss:<15.6e}\n")
        f.write("="*80 + "\n")
        f.write(f"\nAverage epoch time: {avg_epoch_time:.4f} s\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"All Cases Completed!")
    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Total cases: {len(all_results)}")
    print(f"  Average total time: {avg_total_time:.4f} s")
    print(f"  Average best loss: {avg_best_loss:.6e}")
    print(f"  Average epoch time: {avg_epoch_time:.4f} s")
    print(f"{'='*70}")
    print(f"\nResults saved to {output_file}")
    
    # Save per-step loss for each case to txt files
    for case_idx, result in enumerate(all_results):
        init_name = os.path.basename(result['initial_file']).replace('.txt', '')
        loss_hist = result['loss_history']
        loss_file = os.path.join(script_dir, f"letter_curve_noMPC_case{case_idx}_{init_name}_loss.txt")
        with open(loss_file, "w") as f:
            f.write("# Per-step loss history for letter_curve_noMPC\n")
            f.write(f"# Case {case_idx}: {result['initial_file']} -> {result['target_file']}\n")
            f.write("# Step, Loss\n")
            for step, loss_val in enumerate(loss_hist):
                f.write(f"{step}, {loss_val:.10e}\n")
        print(f"Loss history saved to: {loss_file}")
