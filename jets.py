from typing import Dict, Union

import numpy as np
import sympy as sp

from gas import Gas
from state import State

# INPUT PARAMETERS
GASES = {
    "air": Gas("air", R=287.0, gamma=1.4, mu=1.8e-5, k=0.026),
    "helium": Gas("helium", R=2077.0, gamma=1.66, mu=1.96e-5, k=0.151),
    "argon": Gas("argon", R=208.0, gamma=1.67, mu=2.2e-5, k=0.0177),
    "carbon_dioxide": Gas("CO2", R=189.0, gamma=1.3, mu=1.4e-5, k=0.0168),
    "hydrogen": Gas("H2", R=4124.0, gamma=1.41, mu=8.76e-6, k=0.180),
}


def get_gas(name: str) -> Gas:
    if name not in GASES:
        raise ValueError(f"Gas '{name}' not found in library.")
    return GASES[name]


def apply_boundary_conditions(state: State, bc: Dict[str, str]):
    """Applies boundary conditions to all four sides: left, right, bottom, top."""
    for axis, sides in enumerate(["x", "y"]):
        lo, hi = 0, -1

        if bc[sides] == "periodic":
            for field in [state.rho, state.T, state.u, state.v]:
                if axis == 0:  # x-dir
                    field[lo, :] = field[-2, :]
                    field[hi, :] = field[1, :]
                else:  # y-dir
                    field[:, lo] = field[:, -2]
                    field[:, hi] = field[:, 1]

        elif bc[sides] == "reflective":
            if axis == 0:
                state.u[lo, :] *= -1
                state.u[hi, :] *= -1
            else:
                state.v[:, lo] *= -1
                state.v[:, hi] *= -1

        elif bc[sides] == "sticky":
            if axis == 0:
                state.u[lo, :] = 0
                state.u[hi, :] = 0
            else:
                state.v[:, lo] = 0
                state.v[:, hi] = 0

        elif bc[sides] == "slippy":
            pass  # Already default unless you add viscosity terms


def compute_cfl_dt(state: State, dx: float, dy: float, CFL: float = 0.5) -> float:
    """Compute time step size based on CFL condition."""
    velocity = state.velocity_magnitude()
    sound = state.sound_speed()
    max_speed = np.max(velocity + sound)

    dt = CFL * min(dx, dy) / max_speed
    return dt


def compute_fluxes(state: State):
    """Compute inviscid (Euler) fluxes in x and y directions."""
    rho = state.rho
    u = state.u
    v = state.v
    p = state.pressure()
    e = state.total_specific_energy()

    # Conserved variables
    rhou = rho * u
    rhov = rho * v
    rhoE = rho * e

    # X-direction fluxes
    Fx = {"rho": rhou, "rhou": rhou * u + p, "rhov": rhou * v, "rhoE": (rhoE + p) * u}

    # Y-direction fluxes
    Fy = {"rho": rhov, "rhou": rhov * u, "rhov": rhov * v + p, "rhoE": (rhoE + p) * v}

    return Fx, Fy


def compute_rhs(state: State, dx, dy):
    """Compute the time derivative of the conserved variables."""
    Fx, Fy = compute_fluxes(state)

    def ddx(f):  # simple centered difference for now
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

    def ddy(f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

    rhs = {}

    rhs["rho"] = -(ddx(Fx["rho"]) + ddy(Fy["rho"]))
    rhs["rhou"] = -(ddx(Fx["rhou"]) + ddy(Fy["rhou"]))
    rhs["rhov"] = -(ddx(Fx["rhov"]) + ddy(Fy["rhov"]))
    rhs["rhoE"] = -(ddx(Fx["rhoE"]) + ddy(Fy["rhoE"]))

    return rhs


def step_euler(state: State, rhs, dt):
    """Take one Euler time step using RHS."""
    rho = state.rho + dt * rhs["rho"]
    u = (state.rho * state.u + dt * rhs["rhou"]) / rho
    v = (state.rho * state.v + dt * rhs["rhov"]) / rho
    energy = state.energy + dt * rhs["rhoE"]

    e = energy / rho - 0.5 * (u**2 + v**2)
    T = e / state.gas.cv

    state.rho = rho
    state.u = u
    state.v = v
    state.T = T
    state.update_energy()


def step_rk4(state: State, dt, dx, dy):
    """Advance the state one step using 4th-order Runge-Kutta."""
    s1 = state.copy()
    k1 = compute_rhs(s1, dx, dy)

    s2 = state.copy()
    step_euler(s2, k1, dt / 2)
    k2 = compute_rhs(s2, dx, dy)

    s3 = state.copy()
    step_euler(s3, k2, dt / 2)
    k3 = compute_rhs(s3, dx, dy)

    s4 = state.copy()
    step_euler(s4, k3, dt)
    k4 = compute_rhs(s4, dx, dy)

    # Final combined step
    rho = state.rho + (dt / 6) * (k1["rho"] + 2 * k2["rho"] + 2 * k3["rho"] + k4["rho"])
    rhou = state.rho * state.u + (dt / 6) * (
        k1["rhou"] + 2 * k2["rhou"] + 2 * k3["rhou"] + k4["rhou"]
    )
    rhov = state.rho * state.v + (dt / 6) * (
        k1["rhov"] + 2 * k2["rhov"] + 2 * k3["rhov"] + k4["rhov"]
    )
    rhoE = state.energy + (dt / 6) * (
        k1["rhoE"] + 2 * k2["rhoE"] + 2 * k3["rhoE"] + k4["rhoE"]
    )

    rho = np.maximum(rho, 1e-6)
    u = rhou / rho
    v = rhov / rho
    e = rhoE / rho - 0.5 * (u**2 + v**2)
    e = np.maximum(e, 1e-6)
    T = e / state.gas.cv

    state.rho = rho
    state.u = u
    state.v = v
    state.T = T
    state.update_energy()


def solve(
    state: State,
    dx: float,
    dy: float,
    dt: float,
    n_steps: int,
    bc: Dict[str, str],
    output_interval: int = 10,
    CFL: float = 0.5,
) -> State:
    """Solve the compressible Euler equations using a time-stepping method.
    Args:
        state: Initial state of the system.
        dx: Grid spacing in x-direction.
        dy: Grid spacing in y-direction.
        dt: Time step size.
        n_steps: Number of time steps to take.
        bc: Boundary conditions for each side (left, right, bottom, top).
        output_interval: Interval for printing output."
    """
    for step in range(n_steps):
        dt = compute_cfl_dt(state, dx, dy, CFL)
        # rhs = compute_rhs(state, dx, dy)
        # step_euler(state, rhs, dt)
        step_rk4(state, dt, dx, dy)
        apply_boundary_conditions(state, bc)

        if step % output_interval == 0:
            print(
                f"Step {step}: Max rho = {np.max(state.rho):.3f}, Max T = {np.max(state.T):.1f}"
            )

    return state
