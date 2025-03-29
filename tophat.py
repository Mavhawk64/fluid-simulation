# run_jet.py

import matplotlib.pyplot as plt
import numpy as np

from jets import apply_boundary_conditions, get_gas, solve
from state import State

# 1. Simulation config
nx, ny = 200, 100
Lx, Ly = 1.0, 0.5
dx, dy = Lx / nx, Ly / ny
dt = 1e-4
n_steps = 1000
output_interval = 100

# 2. Gas and domain setup
gas = get_gas("air")
state = State(nx, ny, gas)

# 3. Set ambient (background) conditions
rho_ambient = 1.0
T_ambient = 300.0
u_ambient = 0.0
v_ambient = 0.0
state.set_uniform(rho_ambient, T_ambient, u_ambient, v_ambient)

# 4. Apply top-hat jet at the inlet (left boundary)
jet_diameter = int(ny * 0.25)  # 25% of domain height
j_start = ny // 2 - jet_diameter // 2
j_end = ny // 2 + jet_diameter // 2

u_jet = 300.0
T_jet = 600.0
rho_jet = rho_ambient  # or compute via p / (R * T)

state.u[0, j_start:j_end] = u_jet
state.T[0, j_start:j_end] = T_jet
state.rho[0, j_start:j_end] = rho_jet
state.update_energy()

# 5. Boundary conditions
bc = {
    "x": "open",  # inlet and outlet (for now, "open" is no-op)
    "y": "reflective",  # top and bottom wall
}

# 6. Solve
solve(state, dx, dy, dt, n_steps, bc, output_interval=output_interval)

# 7. Visualize
plt.figure(figsize=(10, 4))
plt.imshow(state.u.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis")
plt.colorbar(label="Velocity u (m/s)")
plt.title("Jet Velocity Field (u) at Final Time")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.show()
