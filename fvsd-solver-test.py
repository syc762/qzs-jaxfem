# Import some useful modules.
import meshio
import jax.numpy as np
import jax
print(jax.devices())
import numpy as onp
import os
import csv
import pypardiso
import atexit
import scipy
from scipy.sparse import lil_matrix, vstack, csr_matrix, eye, hstack
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem import logger


# # Try importing gpu-based solver
# try:
#     import pyamgx
#     HAVE_AMGX = True
# except Exception:
#     HAVE_AMGX = False
    
# # ---- AMGX one-time setup (only if available) ----
# AMGX_CFG = None
# AMGX_RSRC = None

# if HAVE_AMGX:
#     # Minimal, stable config (FGMRES + Classical AMG)
#     AMGX_CFG = pyamgx.Config().create_from_dict({
#         "config_version": 2,
#         "deterministic": 1,
#         "solver": {
#             "preconditioner": {
#                 "solver": "AMG",
#                 "algorithm": "CLASSICAL",
#                 "smoother": "JACOBI",
#                 "strength": 0.25,
#                 "max_levels": 10
#             },
#             "solver": "FGMRES",
#             "max_iters": 200,
#             "gmres_n_restart": 50,
#             "tolerance": 1e-8,
#             "print_solve_stats": 1,
#             "monitor_residual": 1
#         }
#     })
#     AMGX_RSRC = pyamgx.Resources().create_simple(AMGX_CFG)

#     @atexit.register
#     def _amgx_teardown():
#         try:
#             if AMGX_RSRC is not None:
#                 AMGX_RSRC.destroy()
#             if AMGX_CFG is not None:
#                 AMGX_CFG.destroy()
#         except Exception:
#             pass


import logging
logger.setLevel(logging.DEBUG)

csv_path = "force_displacement_progressTester_SOlverTest.csv"
# csv_path = "force_displacement_progress4nodes.csv"

# --- create file with header if it doesn't exist ---
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["delta", "-delta", "force_N"])

# --- read existing progress so you can resume ---
done_deltas = set()
with open(csv_path, "r") as f:
    next(f)  # skip header
    for line in f:
        d = float(line.split(",")[0])
        done_deltas.add(round(d, 10))   # store computed deltas

# Material properties.
E = 68.9e9      # Young’s modulus (Pa)
nu = 0.33       # Poisson’s ratio
rho = 2700      # Density (kg/m³)


# Derived Lamé parameters (for JAX-FEM)
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
m = -90.322
area = (0.01103884 * 2 * 0.022225) - (4 * onp.pi * 0.00254 * 0.00254)

mesh_raw = meshio.read("fvsd-solver-test.nas")
print(mesh_raw.cells_dict.keys())
points = mesh_raw.points              # N x 3 array of node coordinates
cells = mesh_raw.cells_dict["tetra10"]  # or "triangle" for 2D
mesh = Mesh(points=points, cells=cells)

# mesh_raw = meshio.read("mesh.msh")
# print(mesh_raw.cells_dict.keys())
# points = mesh_raw.points              # N x 3 array of node coordinates
# cells = mesh_raw.cells_dict["tetra"]  # or "triangle" for 2D
# mesh = Mesh(points=points, cells=cells)

# onp.savetxt("points.txt", points)
# onp.savetxt("cells.txt", cells, fmt="%d")



# def fixed_constraint(point):
#     return np.logical_or(point[1] <= -0.05181350956 + 1e-6) , (point[1] >= 0.05181350956 - 1e-6)

def fixed_constraint(point):
    return np.logical_or(
        point[1] <= -0.05181350956 + 1e-6,
        point[1] >= 0.05181350956 - 1e-6
    )

def zero_disp(point):
    return 0.

dirichlet_bc_info = [
    [fixed_constraint] * 3, [0, 1, 2], [zero_disp] * 3
]

print("a'")



# def added_mass(point):
#     return np.isclose(point[1], 0.0, atol=1e-6) and np.isclose(point[2], 0.0, atol=1e-6)

def added_mass(point):
    return np.logical_and(
        np.isclose(point[1], 0.0, atol=1e-6),
        np.isclose(point[2], 0.0, atol=1e-6)
    )


# def boundary_load(point):
#     y = point[1]
#     z = point[2]
#     return (
#         np.isclose(z, 6.35e-4, atol=1e-6) and
#         y >= -0.01103884 - 1e-6 and
#         y <=  0.01103884 + 1e-6
#     )
    
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


# def rigid_connector(point):
#     y = point[1]
#     z = point[2]
#     return (
#         -0.01103884 - 1e-6 <= y <= 0.01103884 + 1e-6 and
#         (np.isclose(z, 6.35e-4, atol=1e-6) or z <= 0.0 + 1e-6)
#     )
    
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
  
print("b1")
# Apply to all nodes in mesh
rigid_file = "rigid_node_ids.txt"
# rigid_file = "rigid_node_ids4nodes.txt"

if os.path.exists(rigid_file):
    print("Loading rigid node IDs from file...")
    rigid_node_ids = onp.loadtxt(rigid_file, dtype=int)
else:
    print("Computing rigid node IDs...")
    mesh_nodes = mesh.points   # (num_nodes, 3)
    rigid_mask = onp.array([rigid_connector(p) for p in mesh_nodes])
    rigid_node_ids = onp.where(rigid_mask)[0]  # Indices of rigid region nodes
    onp.savetxt(rigid_file, rigid_node_ids, fmt="%d")
    print(f"Saved {len(rigid_node_ids)} node IDs to {rigid_file}")
    
    
# mesh_nodes = mesh.points   # (num_nodes, 3)
# rigid_mask = np.array([rigid_connector(p) for p in mesh_nodes])
# rigid_node_ids = np.where(rigid_mask)[0]  # Indices of rigid region nodes


print("b2")

# Define the weak form.
class A(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, `jax_fem`
    # solves -div.f(u_grad) = b. Here, we define f to be the indentity function.
    # We will see how f is deined as more complicated to solve non-linear problems
    # in later examples.
        
        
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def get_tensor_map(self):
        def stress(u_grad):
            F = np.eye(self.dim) + u_grad
            E = 0.5*(F.T @ F - np.eye(self.dim))
            # epsilon = 0.5 * (u_grad + u_grad.T)
            S = lmbda * np.trace(E) * np.eye(self.dim) + 2*mu*E
            P = F @ S
            return P
        return stress


    # # Define the source term b
    # def get_mass_map(self):
    #     def mass_map(u, x):
    #         gravity = np.array([0., 0., scipy.constants.g])  # in m/s²
    #         # rho = 2700.0  # kg/m³ for aluminum
    #         return rho * gravity  # body force per unit volume
    #     return mass_map

    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         return np.array([0.0, 0.0, m / area])  # traction in z-direction
    #     return [surface_map]
    
    
    
    
# # Define Neumann boundary locations.
# location_fns = [boundary_load]
    


# # Create an instance of the problem.
# problem = Poisson(mesh, vec=3, dim=3, ele_type="TET4", 
#                   dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns) 

# # Building matrix to apply rigid constraint
# def build_rigid_body_constraint_matrix_3d(rigid_node_ids, coords):
#     """
#     Parameters
#     ----------
#     rigid_node_ids : array-like
#         Node indices of the rigid region.
#     coords : array-like
#         Array of shape (num_nodes, 3), giving coordinates of all mesh points.

#     Returns
#     -------
#     P : scipy.sparse.csr_matrix
#         Constraint matrix mapping 6 reduced DOFs to full DOFs of rigid nodes.
#     """
#     x0 = onp.mean(coords[rigid_node_ids], axis=0)
#     dof_per_node = 3
#     total_dofs = dof_per_node * coords.shape[0]
#     P = lil_matrix((total_dofs, 6))

#     for nid in rigid_node_ids:
#         xi = coords[nid]
#         dx, dy, dz = xi - x0

#         row_base = nid * 3
#         # Fill P rows for ux, uy, uz
#         # Translation part
#         P[row_base + 0, 0] = 1  # u_x
#         P[row_base + 1, 1] = 1  # u_y
#         P[row_base + 2, 2] = 1  # u_z

#         # Rotation part (cross product)
#         # [ω × (x - x0)] = [ -ωz dy + ωy dz,
#         #                    ωz dx - ωx dz,
#         #                   -ωy dx + ωx dy ]
#         P[row_base + 0, 4] = dz   # ωy
#         P[row_base + 0, 5] = -dy  # -ωz

#         P[row_base + 1, 3] = -dz  # -ωx
#         P[row_base + 1, 5] = dx   # ωz

#         P[row_base + 2, 3] = dy   # ωx
#         P[row_base + 2, 4] = -dx  # -ωy

#     return P.tocsr()

# P_mat = build_rigid_body_constraint_matrix_3d(rigid_node_ids, problem.fes[0].points)

# num_nodes = mesh.points.shape[0]
# dof_per_node = 3
# total_dofs = num_nodes * dof_per_node

# rigid_set = set(onp.array(rigid_node_ids).tolist())
# free_node_ids = [i for i in range(num_nodes) if i not in rigid_set]
# num_free = len(free_node_ids)

# P_full = lil_matrix((total_dofs, 6 + num_free * dof_per_node))

# for node_id in range(num_nodes):
#     for d in range(dof_per_node):
#         row = node_id * dof_per_node + d
#         if node_id in rigid_set:
#             # Only fill for rigid node rows
#             P_full[row, :6] = P_mat[row, :]
#         else:
#             free_idx = free_node_ids.index(node_id)
#             col = 6 + free_idx * dof_per_node + d
#             P_full[row, col] = 1.0

#### problem.P_mat = P_full.tocsr()

# problem.P_mat = P_mat

def pardiso_solver(A, b, x0, solver_options):
    """
    Solves Ax=b with x0 being the initial guess.

    A: PETSc sparse matrix
    b: JAX array
    x0: JAX array (forward problem) or None (adjoint problem)
    solver_options: anything the user defines, at least satisfying solver_options['custom_solver'] = pardiso_solver
    """
    logger.debug(f"Pardiso Solver - Solving linear system")

    # If you need to convert PETSc to scipy
    indptr, indices, data = A.getValuesCSR()
    A_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
    x = pypardiso.spsolve(A_sp_scipy, onp.array(b))
    return x

def amgx_solver(A, b, x0, solver_options):
    """
    AMGX (GPU) linear solve: Ax = b
    A: PETSc sparse (device-agnostic); b, x0: JAX arrays
    Returns: NumPy array (same shape as b)
    """
    if not HAVE_AMGX or AMGX_RSRC is None or AMGX_CFG is None:
        logger.warning("AMGX requested but not available; falling back to Pardiso.")
        return pardiso_solver(A, b, x0, solver_options)

    logger.debug("AMGX Solver - Solving linear system on GPU")

    # --- Convert PETSc to SciPy CSR ---
    indptr, indices, data = A.getValuesCSR()
    nrows, ncols = A.getSize()
    # Ensure contiguous arrays (AMGX likes that)
    indptr  = onp.asarray(indptr,  dtype=onp.int32, order='C')
    indices = onp.asarray(indices, dtype=onp.int32, order='C')
    data    = onp.asarray(data,    dtype=onp.float64, order='C')  # or float32 if your build is FP32

    # RHS / initial guess
    b_np = onp.asarray(b,  dtype=data.dtype, order='C')
    if x0 is None:
        x_np = onp.zeros_like(b_np)
    else:
        x_np = onp.asarray(x0, dtype=data.dtype, order='C')

    # --- AMGX objects ---
    A_m = pyamgx.Matrix().create(AMGX_RSRC)
    r_v = pyamgx.Vector().create(AMGX_RSRC)
    x_v = pyamgx.Vector().create(AMGX_RSRC)
    slv = pyamgx.Solver().create(AMGX_RSRC, AMGX_CFG)

    try:
        # Upload matrix and vectors
        A_m.upload_CSR(indptr, indices, data, nrows, ncols)
        r_v.upload(b_np)
        x_v.upload(x_np)

        # Solve
        slv.setup(A_m)
        slv.solve(r_v, x_v)

        # Download solution
        x_out = x_v.download()
    finally:
        # Clean up GPU resources for this solve
        try:
            slv.destroy()
            x_v.destroy(); r_v.destroy(); A_m.destroy()
        except Exception:
            pass

    return x_out


# =========================
# DOF indices for rigid z components
# =========================
def rigid_z_dof_indices(problem, rigid_node_ids):
    fe = problem.fes[0]
    base = problem.offset[0]
    return (onp.array(rigid_node_ids, dtype=onp.int64) * fe.vec + 2 + base).astype(onp.int64)  # z-comp

# # Solve the defined problem.
# sol_list = solver(problem, solver_options={'custom_solver': pardiso_solver})

# # =========================
# # Displacement-controlled sweep
# # =========================
deltas = onp.arange(-0.001, 0.003 + 1e-12, 0.0001)  # [-1e-3, 3e-3], step 1e-4


# Build BCs (only here; not outside)
# Ends clamped (ux=uy=uz=0)
loc_list = [fixed_constraint, fixed_constraint, fixed_constraint]
comp_list = [0, 1, 2]
val_list = [zero_disp, zero_disp, zero_disp]

# Rigid block: ux=0, uy=0, uz=delta
loc_list += [rigid_connector, rigid_connector, rigid_connector]
comp_list += [0, 1, 2]
val_list += [zero_disp, zero_disp, (lambda point: 0.0)]

dirichlet_bc_info = [loc_list, comp_list, val_list]

print("c")


# No surface maps: omit location_fns
problem = A(
    mesh, vec=3, dim=3, ele_type="TET10",     #ele_type="TET4",
    dirichlet_bc_info=dirichlet_bc_info
)
    
# Keep track of which entry is the rigid uz value function (last one we added)
idx_rigid_uz = len(val_list) - 1

# # Optional: keep previous solution to warm-start Newton
prev_sol = None    
    
    
forces = []

for delta in deltas:
    if round(delta, 10) in done_deltas:   # skip already done
        continue
    logger.info(f"Solving for rigid uz = {-delta:.7e} m")


    
    print("d")
    # IMPORTANT: freeze `delta` into the closure with a default arg
    val_list[idx_rigid_uz] = (lambda point, d=delta: -d)

    # Tell FE to rebuild only the Dirichlet vectors/masks with updated values
    problem.fes[0].update_Dirichlet_boundary_conditions([loc_list, comp_list, val_list])

    # Solve (optionally warm-start if your solver supports it)
    sol_list = solver(problem, solver_options={'custom_solver': amgx_solver})

    # # Save warm-start
    prev_sol = sol_list[0]

    # Reaction from unmasked residual on rigid z-DOFs
    res_unc = jax.flatten_util.ravel_pytree(problem.compute_residual(sol_list))[0]
    iz = rigid_z_dof_indices(problem, rigid_node_ids)
    Rz = np.sum(res_unc[iz])

    print("e")
    # Actuator force (equal and opposite to reaction)
    F_applied = -Rz
    
    # --- immediately append result to CSV ---
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([delta, -delta, float(F_applied)])
        
        
    forces.append(float(F_applied))

    
    # #### Von Misses Stress
    # # Postprocess for stress evaluations
    # # (num_cells, num_quads, vec, dim)
    # u_grad = problem.fes[0].sol_to_grad(sol_list[0])
    # epsilon = 0.5 * (u_grad + u_grad.transpose(0,1,3,2))
    # # (num_cells, bnum_quads, 1, 1) * (num_cells, num_quads, vec, dim)
    # # -> (num_cells, num_quads, vec, dim)
    # sigma = lmbda * np.trace(epsilon, axis1=2, axis2=3)[:,:,None,None] * np.eye(problem.dim) + 2*mu*epsilon
    # # (num_cells, num_quads)
    # cells_JxW = problem.JxW[:,0,:]
    # # (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1) ->
    # # (num_cells, vec, dim) / (num_cells, 1, 1)
    # #  --> (num_cells, vec, dim)
    # sigma_average = np.sum(sigma * cells_JxW[:,:,None,None], axis=1) / np.sum(cells_JxW, axis=1)[:,None,None]

    # # Von Mises stress
    # # (num_cells, dim, dim)
    # s_dev = (sigma_average - 1/problem.dim * np.trace(sigma_average, axis1=1, axis2=2)[:,None,None]
    #                                        * np.eye(problem.dim)[None,:,:])
    # # (num_cells,)
    # vm_stress = np.sqrt(3./2.*np.sum(s_dev*s_dev, axis=(1,2)))

    # # Store the solution to local file.
    # data_dir = os.path.join(os.path.dirname(__file__), 'data/continuous data')
    # vtk_path = os.path.join(data_dir, f'{-delta}displacement/u.vtu')
    # save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('vm_stress', vm_stress)])
    # #####

    
    
    
    
    
    
    
    

# =========================
# Plot Force–Displacement
# =========================
# plt.figure()
# plt.plot(deltas, forces, linewidth=2)
# plt.xlabel('Rigid displacement $u_z$ (m)')
# plt.ylabel('Force $F_z$ (N)')
# plt.title('Force–Displacement (displacement-controlled)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('force_displacement2.png', dpi=200, bbox_inches='tight')

# =========================
# CSV export
# =========================
# csv_path = 'force_displacement.csv'
# with open(csv_path, 'w', newline='') as f:
#     w = csv.writer(f)
#     w.writerow(['uz_m', 'Fz_N'])
#     for d, fz in zip(deltas, forces):
#         w.writerow([f'{d:.7e}', f'{fz:.7e}'])
# print(f'Wrote {csv_path}')

# one value just
# delta = -0.0015
# logger.info(f"Solving for rigid uz = {delta:.7e} m")

# # Build BCs (only here; not outside)
# # Ends clamped (ux=uy=uz=0)
# loc_list = [fixed_constraint, fixed_constraint, fixed_constraint]
# comp_list = [0, 1, 2]
# val_list = [zero_disp, zero_disp, zero_disp]

# # Rigid block: ux=0, uy=0, uz=delta
# loc_list += [rigid_connector, rigid_connector, rigid_connector]
# comp_list += [0, 1, 2]
# val_list += [zero_disp, zero_disp, (lambda point, d=delta: d)]

# dirichlet_bc_info = [loc_list, comp_list, val_list]

# # No surface maps: omit location_fns
# problem = Poisson(
#     mesh, vec=3, dim=3, ele_type="TET4",
#     dirichlet_bc_info=dirichlet_bc_info
# )

# sol_list = solver(problem, solver_options={'custom_solver': pardiso_solver})

# # Reaction from unmasked residual on rigid z-DOFs
# res_unc = jax.flatten_util.ravel_pytree(problem.compute_residual(sol_list))[0]
# iz = rigid_z_dof_indices(problem, rigid_node_ids)
# Rz = np.sum(res_unc[iz])

# # Actuator force (equal and opposite to reaction)
# F_applied = -Rz
# print(F_applied)


# # Solve the defined problem.
# sol_list = solver(problem)


##### Von Misses Stress
# # Postprocess for stress evaluations
# # (num_cells, num_quads, vec, dim)
# u_grad = problem.fes[0].sol_to_grad(sol_list[0])
# epsilon = 0.5 * (u_grad + u_grad.transpose(0,1,3,2))
# # (num_cells, bnum_quads, 1, 1) * (num_cells, num_quads, vec, dim)
# # -> (num_cells, num_quads, vec, dim)
# sigma = lmbda * np.trace(epsilon, axis1=2, axis2=3)[:,:,None,None] * np.eye(problem.dim) + 2*mu*epsilon
# # (num_cells, num_quads)
# cells_JxW = problem.JxW[:,0,:]
# # (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1) ->
# # (num_cells, vec, dim) / (num_cells, 1, 1)
# #  --> (num_cells, vec, dim)
# sigma_average = np.sum(sigma * cells_JxW[:,:,None,None], axis=1) / np.sum(cells_JxW, axis=1)[:,None,None]

# # Von Mises stress
# # (num_cells, dim, dim)
# s_dev = (sigma_average - 1/problem.dim * np.trace(sigma_average, axis1=1, axis2=2)[:,None,None]
#                                        * np.eye(problem.dim)[None,:,:])
# # (num_cells,)
# vm_stress = np.sqrt(3./2.*np.sum(s_dev*s_dev, axis=(1,2)))

# # Store the solution to local file.
# data_dir = os.path.join(os.path.dirname(__file__), 'data')
# vtk_path = os.path.join(data_dir, 'displacementBased1stwithoutGTry/u.vtu')
# save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('vm_stress', vm_stress)])
######

# # # after solve
# node_id = 11667 # from the msh file 
# u = sol_list[0]          # shape (num_nodes, 3) or (ndof, 3)
# disp_vector = u[node_id] # gives [ux, uy, uz] for that node
# print("Displacement of node 11667:", disp_vector)
