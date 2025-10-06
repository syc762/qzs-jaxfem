
#!/usr/bin/env python3
# fvsd_petsc_gpu.py — Displacement-controlled F–v–d sweep
# Prefers PETSc KSP on GPU (via petsc4py). Auto-falls back to Pardiso (CPU) if PETSc-GPU is unavailable.
# Usage examples:
#   python fvsd_petsc_gpu.py -vec_type cuda -mat_type aijcusparse -ksp_type gmres -pc_type gamg -ksp_rtol 1e-8 -ksp_monitor
#   python fvsd_petsc_gpu.py -vec_type cuda -mat_type aijcusparse -ksp_type cg -pc_type gamg -ksp_rtol 1e-10 -ksp_converged_reason

import os, csv, logging
import meshio, jax.numpy as np, jax, numpy as onp, scipy
from scipy.sparse import csr_array
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh
from jax_fem import logger

logger.setLevel(logging.DEBUG)
print("JAX devices:", jax.devices())

# ---- CSV progress ----
csv_path = "force_displacement_progressTesterPetscy.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["delta", "-delta", "force_N", "solver", "notes"])

done_deltas = set()
with open(csv_path, "r") as f:
    next(f, None)
    for line in f:
        try:
            d = float(line.split(",")[0]); done_deltas.add(round(d, 12))
        except Exception:
            pass

# ---- Material ----
E, nu, rho = 68.9e9, 0.33, 2700
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
    return np.logical_or(point[1] <= -0.05181350956 + 1e-6, point[1] >=  0.05181350956 - 1e-6)
def zero_disp(point): return 0.
def rigid_connector(point):
    y, z = point[1], point[2]
    cy = np.logical_and(y >= -0.01103884 - 1e-6, y <= 0.01103884 + 1e-6)
    cz = np.logical_or(np.isclose(z, 6.35e-4, atol=1e-6), z <= 0.0 + 1e-6)
    return np.logical_and(cy, cz)

# Cache rigid region ids
rigid_file = "rigid_node_ids.txt"
if os.path.exists(rigid_file):
    print("Loading rigid node IDs from file..."); rigid_node_ids = onp.loadtxt(rigid_file, dtype=int)
else:
    print("Computing rigid node IDs...")
    rigid_mask = onp.array([rigid_connector(p) for p in mesh.points])
    rigid_node_ids = onp.where(rigid_mask)[0]
    onp.savetxt(rigid_file, rigid_node_ids, fmt="%d")
    print(f"Saved {len(rigid_node_ids)} node IDs to {rigid_file}")

# ---- Problem ----
class A(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            F = np.eye(self.dim) + u_grad
            Eten = 0.5 * (F.T @ F - np.eye(self.dim))
            S = lmbda * np.trace(Eten) * np.eye(self.dim) + 2 * mu * Eten
            return F @ S
        return stress

def rigid_z_dof_indices(problem, rigid_node_ids):
    fe = problem.fes[0]; base = problem.offset[0]
    return (onp.array(rigid_node_ids, dtype=onp.int64) * fe.vec + 2 + base).astype(onp.int64)

# ---- Pardiso fallback (CPU) ----
import pypardiso
def pardiso_solver(A, b, x0, solver_options):
    logger.debug("Pardiso Solver - CPU")
    indptr, indices, data = A.getValuesCSR()
    A_sp = csr_array((data, indices, indptr), shape=A.getSize())
    return pypardiso.spsolve(A_sp, onp.asarray(b))

# ---- PETSc (GPU) preferred ----
def petsc_gpu_solver(A, b, x0, solver_options):
    try:
        from petsc4py import PETSc
    except Exception as e:
        logger.warning("petsc4py import failed; falling back to Pardiso: %s", e)
        return pardiso_solver(A, b, x0, solver_options)

    logger.debug("PETSc-GPU Solver (KSP) — Using PETSc options for GPU types")
    nrows, _ = A.getSize()

    # Build PETSc Vecs; types (cuda/hips/cpu) come from runtime -vec_type/-mat_type
    r = PETSc.Vec().createMPI(nrows)
    x = PETSc.Vec().createMPI(nrows)

    # dtype: PETSc default is float64 unless built single
    b_np = onp.asarray(b, dtype=onp.float64, order='C')
    r.setArray(b_np)

    if x0 is not None:
        x.setArray(onp.asarray(x0, dtype=b_np.dtype, order='C'))
    else:
        x.set(0)

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    # Reasonable defaults; can be overridden by CLI flags
    ksp.setType('gmres')
    pc = ksp.getPC(); pc.setType('gamg')
    ksp.setFromOptions()

    try:
        ksp.solve(r, x)
        sol = x.getArray(readonly=True).copy()
    finally:
        try:
            ksp.destroy(); pc.destroy(); r.destroy(); x.destroy()
        except Exception:
            pass

    return sol

# ---- Sweep ----
deltas = onp.arange(-0.001, 0.003 + 1e-12, 0.0001)
loc_list = [fixed_constraint, fixed_constraint, fixed_constraint]
comp_list = [0, 1, 2]
val_list = [zero_disp, zero_disp, zero_disp]
loc_list += [rigid_connector, rigid_connector, rigid_connector]
comp_list += [0, 1, 2]
val_list += [zero_disp, zero_disp, (lambda point: 0.0)]
dirichlet_bc_info = [loc_list, comp_list, val_list]
problem = A(mesh, vec=3, dim=3, ele_type="TET10", dirichlet_bc_info=dirichlet_bc_info)
idx_rigid_uz = len(val_list) - 1

prev_sol, forces = None, []
for delta in deltas:
    if round(delta, 12) in done_deltas:
        continue
    logger.info(f"Solving for rigid uz = {-delta:.7e} m")
    logger.info("Linear solver preference: PETSc (GPU) with CPU fallback via Pardiso")

    val_list[idx_rigid_uz] = (lambda point, d=delta: -d)
    problem.fes[0].update_Dirichlet_boundary_conditions([loc_list, comp_list, val_list])

    # Try PETSc-GPU first; if users launch with CPU types, it'll still run on CPU inside PETSc
    try:
        sol_list = solver(problem, solver_options={'custom_solver': petsc_gpu_solver})
    except Exception as e:
        logger.warning("PETSc-GPU path raised; falling back to Pardiso. Err: %s", e)
        sol_list = solver(problem, init_sol=prev_sol, solver_options={'custom_solver': pardiso_solver})

    prev_sol = sol_list[0]

    res_unc = jax.flatten_util.ravel_pytree(problem.compute_residual(sol_list))[0]
    iz = rigid_z_dof_indices(problem, rigid_node_ids)
    Rz = np.sum(res_unc[iz])
    F_applied = -Rz

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([delta, -delta, float(F_applied), "PETSc->GPU or Pardiso->CPU", "-vec_type/-mat_type control"])
    forces.append(float(F_applied))

print("Done. Appended results to", csv_path)
