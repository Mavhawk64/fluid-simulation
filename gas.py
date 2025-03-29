class Gas:
    def __init__(self, name: str, R: float, gamma: float, mu: float, k: float):
        self.name = name  # Name of the gas
        self.R = R  # Specific gas constant (J/kg·K)
        self.gamma = gamma  # Ratio of specific heats (c_p / c_v)
        self.mu = mu  # Dynamic viscosity (Pa·s)
        self.k = k  # Thermal conductivity (W/m·K)

    @property
    def cv(self) -> float:
        # c_v = R / (gamma - 1)
        return self.R / (self.gamma - 1)

    @property
    def cp(self) -> float:
        # c_p = gamma * R / (gamma - 1)
        return self.gamma * self.R / (self.gamma - 1)

    def pressure(self, rho: float, T: float) -> float:
        """Compute pressure from density and temperature (ideal gas)."""
        return rho * self.R * T

    def internal_energy(self, T: float) -> float:
        """Compute specific internal energy from temperature."""
        return self.cv * T

    def total_energy(self, T: float, u: float, v: float) -> float:
        """Compute specific total energy (internal + kinetic)."""
        return self.cv * T + 0.5 * (u**2 + v**2)

    def sound_speed(self, T: float) -> float:
        """Compute sound speed."""
        return (self.gamma * self.R * T) ** 0.5

    def __repr__(self) -> str:
        return f"Gas(name={self.name}, R={self.R}, gamma={self.gamma})"

    def __str__(self) -> str:
        return f"Gas: {self.name}, R: {self.R}, gamma: {self.gamma}, mu: {self.mu}, k: {self.k}"
