"""GaitDynamics diffusion and physics refinement scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GaitDynamicsConfig:
    """Sequential kinetics window and force conversion contracts."""

    window_seconds: float = 1.5
    sample_hz: int = 120
    parameter_count: int = 32


class GaitDynamicsPipeline:
    """Kinetics reconstruction model scaffold.

    Data window tensor:
        X ∈ R^{Time × Parameters}

    Coordinate conversion:
        F_Sim(t) = R_adapt · F_GRF(t)

    Physics consistency gate:
        F_ext - m a = 0
    """

    def __init__(self, config: GaitDynamicsConfig | None = None) -> None:
        self.config = config or GaitDynamicsConfig()

    def build_window_tensor(self) -> np.ndarray:
        time_steps = int(self.config.window_seconds * self.config.sample_hz)
        return np.zeros((time_steps, self.config.parameter_count), dtype=np.float32)

    def adapt_forces(self, forces_grf: np.ndarray, r_adapt: np.ndarray) -> np.ndarray:
        return (r_adapt @ forces_grf.T).T

    def physics_residual(self, forces_ext: np.ndarray, mass: float, acceleration: np.ndarray) -> np.ndarray:
        return forces_ext - mass * acceleration
