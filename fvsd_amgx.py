
#!/usr/bin/env python3
# fvsd_amgx.py — Displacement-controlled F–v–d sweep with AMGX (GPU) linear solver
# Requires: jax, jax_fem, meshio, numpy, scipy, pyamgx
# Notes:
#   - Ensure your AMGX/pyamgx build precision matches dtypes below (float64 by default).
#   - If your pyamgx wheel is FP32-only, change dtype conversions to float32.
#   - AMGX is initialized/finalized once per process via atexit hook.

import os
import csv
import atexit
import logging

import meshio
import jax.numpy as np
import jax
import numpy as onp
import scipy
from scipy.sparse import csr_array

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh
from jax_fem import logger

logger.setLevel(logging.DEBUG)

print("JAX devices:", jax.devices())

csv_path = "force_displacement_progressTesterAMGX.csv"

# --- create file with header if it doesn't exist ---
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["delta", "-delta", "force_N", "solver", "notes"])

# --- read existing progress so you can resume ---
done_deltas = set()
with open(csv_path, "r") as f:
    next(f)  # skip header
    for line in f:
        d = float(line.split(",")[0])
        done_deltas.add(round(d, 12))   # safer rounding

# Material properties.
E = 68.9e9      # Young’s modulus (Pa)
nu = 0.33       # Poisson’s ratio
rho = 2700      # Density (kg/m³)

# Derived Lamé parameters
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
m = -90.322
area = (0.01103884 * 2 * 0.022225) - (4 * onp.pi * 0.00254 * 0.00254)

# ---- Mesh (TET10) ----
mesh_raw = meshio.read("fvsd-solver-test.nas")
print("Available cell types:", mesh_raw.cells_dict.keys())
points = mesh_raw.points
cells = mesh_raw.cells_dict["tetra10"]
mesh = Mesh(points=points, cells=cells)

# ---- BC helpers ----
def fixed_constraint(point):
    return np.logical_or(
        point[1] <= -0.05181350956 + 1e-6,
        point[1] >=  0.05181350956 - 1e-6
    )

def zero_disp(point):
    return 0.

def boundary_load(point):
    y = point[1]
    z = point[2]
    return np.logical_and(
        np.isclose(z, 6.35e-4, atol=1e-6),
        np.logical_and(
            y >= -0.01103884 - 1e-6,
            y <=  0.01103884 + 1e-6
        )
    )

def rigid_connector(point):
    y = point[1]
    z = point[2]
    condition_y = np.logical_and(
        y >= -0.01103884 - 1e-6,
        y <=  0.01103884 + 1e-6
    )
    condition_z = np.logical_or(
        np.isclose(z, 6.35e-4, atol=1e-6),
        z <= 0.0 + 1e-6
    )
    return np.logical_and(condition_y, condition_z)

# Cache rigid region node ids to file
rigid_file = "rigid_node_ids.txt"
if os.path.exists(rigid_file):
    print("Loading rigid node IDs from file...")
    rigid_node_ids = onp.loadtxt(rigid_file, dtype=int)
else:
    print("Computing rigid node IDs...")
    rigid_mask = onp.array([rigid_connector(p) for p in mesh.points])
    rigid_node_ids = onp.where(rigid_mask)[0]
    onp.savetxt(rigid_file, rigid_node_ids, fmt="%d")
    print(f"Saved {len(rigid_node_ids)} node IDs to {rigid_file}")

# ---- Problem (Saint-Venant–Kirchhoff, TET10) ----
class A(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            F = np.eye(self.dim) + u_grad
            Eten = 0.5 * (F.T @ F - np.eye(self.dim))
            S = lmbda * np.trace(Eten) * np.eye(self.dim) + 2 * mu * Eten
            P = F @ S
            return P
        return stress

# DOF indices for rigid z components
def rigid_z_dof_indices(problem, rigid_node_ids):
    fe = problem.fes[0]
    base = problem.offset[0]
    return (onp.array(rigid_node_ids, dtype=onp.int64) * fe.vec + 2 + base).astype(onp.int64)

# ---------------- AMGX (GPU) solver ----------------
HAVE_AMGX = False
AMGX_CFG  = None
AMGX_RSRC = None

try:
    import pyamgx
    HAVE_AMGX = True
except Exception as e:
    logger.warning("pyamgx import failed; AMGX solver unavailable: %s", e)

if HAVE_AMGX:
    try:
        pyamgx.initialize()
        # Minimal deterministic config (tune as needed)
        AMGX_CFG = pyamgx.Config().create_from_dict({
            "config_version": 2,
            "deterministic": 1,
            "solver": {
                "preconditioner": {
                    "solver": "AMG",
                    "algorithm": "CLASSICAL",
                    "smoother": "JACOBI",
                    "strength": 0.25,
                    "max_levels": 10
                },
                "solver": "FGMRES",
                "max_iters": 200,
                "gmres_n_restart": 50,
                "tolerance": 1e-8,
                "print_solve_stats": 1,
                "monitor_residual": 1
            }
        })
        AMGX_RSRC = pyamgx.Resources().create_simple(AMGX_CFG)
    except Exception as e:
        logger.warning("AMGX init failed; falling back when used: %s", e)
        HAVE_AMGX = False

    @atexit.register
    def _amgx_teardown():
        try:
            if AMGX_RSRC is not None: AMGX_RSRC.destroy()
            if AMGX_CFG  is not None: AMGX_CFG.destroy()
        finally:
            try:
                pyamgx.finalize()
            except Exception:
                pass

def amgx_solver(A, b, x0, solver_options):
    if not HAVE_AMGX or AMGX_RSRC is None or AMGX_CFG is None:
        logger.warning("AMGX requested but not available; aborting.")
        raise RuntimeError("AMGX not available")

    logger.debug("AMGX Solver - Solving linear system on GPU")
    indptr, indices, data = A.getValuesCSR()
    nrows, ncols = A.getSize()

    # Adjust dtype (float64 by default; switch to float32 if needed)
    indptr  = onp.asarray(indptr,  dtype=onp.int32,  order='C')
    indices = onp.asarray(indices, dtype=onp.int32,  order='C')
    data    = onp.asarray(data,    dtype=onp.float64, order='C')

    b_np = onp.asarray(b,  dtype=data.dtype, order='C')
    x_np = onp.zeros_like(b_np) if x0 is None else onp.asarray(x0, dtype=data.dtype, order='C')

    A_m = pyamgx.Matrix().create(AMGX_RSRC)
    r_v = pyamgx.Vector().create(AMGX_RSRC)
    x_v = pyamgx.Vector().create(AMGX_RSRC)
    slv = pyamgx.Solver().create(AMGX_RSRC, AMGX_CFG)

    try:
        A_m.upload_CSR(indptr, indices, data, nrows, ncols)
        r_v.upload(b_np)
        x_v.upload(x_np)
        slv.setup(A_m)
        slv.solve(r_v, x_v)
        x_out = x_v.download()
    finally:
        try:
            slv.destroy(); x_v.destroy(); r_v.destroy(); A_m.destroy()
        except Exception:
            pass

    return x_out

# ---------------- Sweep setup ----------------
deltas = onp.arange(-0.001, 0.003 + 1e-12, 0.0001)

# Build BCs
loc_list = [fixed_constraint, fixed_constraint, fixed_constraint]
comp_list = [0, 1, 2]
val_list = [zero_disp, zero_disp, zero_disp]

# Rigid block: ux=0, uy=0, uz=delta (updated per step)
loc_list += [rigid_connector, rigid_connector, rigid_connector]
comp_list += [0, 1, 2]
val_list += [zero_disp, zero_disp, (lambda point: 0.0)]

dirichlet_bc_info = [loc_list, comp_list, val_list]

problem = A(mesh, vec=3, dim=3, ele_type="TET10", dirichlet_bc_info=dirichlet_bc_info)
idx_rigid_uz = len(val_list) - 1

prev_sol = None
forces = []

for delta in deltas:
    if round(delta, 12) in done_deltas:
        continue

    logger.info(f"Solving for rigid uz = {-delta:.7e} m")
    logger.info("Linear solver: %s", "AMGX (GPU)" if HAVE_AMGX else "AMGX (requested) but unavailable")

    # Freeze current delta into BC
    val_list[idx_rigid_uz] = (lambda point, d=delta: -d)
    problem.fes[0].update_Dirichlet_boundary_conditions([loc_list, comp_list, val_list])

    # Solve with warm-start if available
    sol_list = solver(problem, solver_options={'custom_solver': amgx_solver})
    prev_sol = sol_list[0]

    # Reaction -> applied force
    res_unc = jax.flatten_util.ravel_pytree(problem.compute_residual(sol_list))[0]
    iz = rigid_z_dof_indices(problem, rigid_node_ids)
    Rz = np.sum(res_unc[iz])
    F_applied = -Rz

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([delta, -delta, float(F_applied), "AMGX(GPU)", "FGMRES+AMG"])

    forces.append(float(F_applied))

print("Done. Appended results to", csv_path)
