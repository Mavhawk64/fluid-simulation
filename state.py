# state.py

import numpy as np

from gas import Gas


class State:
    def __init__(self, nx: int, ny: int, gas: Gas):
        self.nx = nx
        self.ny = ny
        self.gas = gas

        # Primary conserved variables
        self.rho = np.zeros((nx, ny))  # density
        self.u = np.zeros((nx, ny))  # velocity in x
        self.v = np.zeros((nx, ny))  # velocity in y
        self.T = np.zeros((nx, ny))  # temperature

        # Optional: Precompute total energy (ρe + ½ρv²)
        self.energy = np.zeros((nx, ny))  # total energy (ρ * ε)

    def update_energy(self):
        """Update total energy field from current T, u, v, rho."""
        internal_energy_fn = np.vectorize(self.gas.internal_energy)
        e = internal_energy_fn(self.T)
        self.energy = self.rho * (e + 0.5 * (self.u**2 + self.v**2))

    def pressure(self) -> np.ndarray:
        """Return pressure field (ideal gas law)."""
        return self.rho * self.gas.R * self.T

    def internal_energy(self) -> np.ndarray:
        """Return specific internal energy field."""
        internal_energy_fn = np.vectorize(self.gas.internal_energy)
        return internal_energy_fn(self.T)

    def velocity_magnitude(self) -> np.ndarray:
        """Return magnitude of velocity vector."""
        return np.sqrt(self.u**2 + self.v**2)

    def sound_speed(self) -> np.ndarray:
        """Return sound speed field (guarantee positive T)."""
        sound_speed_fn = np.vectorize(self._safe_sound_speed)
        return sound_speed_fn(self.T)

    def _safe_sound_speed(self, T_val: float) -> float:
        """Safely compute sound speed from scalar temperature."""
        T_val = max(T_val, 1e-6)  # prevent negative/zero temp
        return (self.gas.gamma * self.gas.R * T_val) ** 0.5

    def mach_number(self) -> np.ndarray:
        """Return Mach number field."""
        c = self.sound_speed()
        return self.velocity_magnitude() / c

    def kinetic_energy(self) -> np.ndarray:
        """Return specific kinetic energy field (½v²)."""
        return 0.5 * (self.u**2 + self.v**2)

    def total_specific_energy(self) -> np.ndarray:
        """Return ε = e + ½v² (per unit mass)."""
        return self.internal_energy() + self.kinetic_energy()

    def set_uniform(self, rho0: float, T0: float, u0: float = 0.0, v0: float = 0.0):
        """Initialize a uniform state."""
        self.rho.fill(rho0)
        self.T.fill(T0)
        self.u.fill(u0)
        self.v.fill(v0)
        self.update_energy()

    def copy(self):
        """Return a deep copy of the current state."""
        new = State(self.nx, self.ny, self.gas)
        new.rho = self.rho.copy()
        new.u = self.u.copy()
        new.v = self.v.copy()
        new.T = self.T.copy()
        new.energy = self.energy.copy()
        return new
