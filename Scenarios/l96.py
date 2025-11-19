from __future__ import annotations
from typing import Tuple
import torch
import torch.distributions as dist
from Scenarios.abstract_scenario import Scenario

Tensor = torch.Tensor


class L96(Scenario):
    """
    Classic Lorenz-96 chaotic system.  The implementation trades clever vector tricks
    for a small collection of well-named helpers so that the data flow is easy to follow.
    """

    def __init__(
        self,
        *,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        F: float = 8.0,
        time_step: float = 0.01,
        substeps: int = 1,
        p: float = 1.0,
        eps: float = 1e-6,
        process_var: float = 1.0,
        observation_var: float = 1.0,
        initial_state_var: float = 1.0,
    ):
        self._state_dim = int(dimension)
        self.device = device
        self.dtype = dtype

        self.forcing = float(F)
        self.time_step = float(time_step)
        self.substeps = max(int(substeps), 1)
        self.measurement_power = float(p)
        self.measurement_eps = float(eps)
        self.process_var = float(process_var)
        self.measurement_var = float(observation_var)
        self.initial_state_var = float(initial_state_var)

        identity = torch.eye(self._state_dim, device=device, dtype=dtype)
        self.Q = self.process_var * identity
        self.R = self.measurement_var * identity
        self.P0 = self.initial_state_var * identity
        self.H = identity.clone()

        self.process_std = self.process_var ** 0.5
        self.measurement_std = self.measurement_var ** 0.5
        self.initial_state_std = self.initial_state_var ** 0.5

        zero = torch.zeros(self.state_dim, device=device, dtype=dtype)
        self.process_noise = dist.MultivariateNormal(loc=zero, covariance_matrix=self.Q)
        self.measurement_noise = dist.MultivariateNormal(loc=zero, covariance_matrix=self.R)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    # ------------------------------------------------------------------ helpers
    def _integrate_without_noise(self, state: Tensor) -> Tensor:
        """Integrate the deterministic part of the dynamics for one outer step."""
        step = self.time_step / self.substeps
        current = state
        for _ in range(self.substeps):
            current = self._rk4_step(current, step)
        return current

    def _rk4_step(self, state: Tensor, step: float) -> Tensor:
        k1 = self._state_derivative(state)
        k2 = self._state_derivative(state + 0.5 * step * k1)
        k3 = self._state_derivative(state + 0.5 * step * k2)
        k4 = self._state_derivative(state + step * k3)
        return state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _state_derivative(self, state: Tensor) -> Tensor:
        """Lorenz-96 right hand side with cyclic neighbors."""
        x_im2 = torch.roll(state, shifts=2, dims=-1)
        x_im1 = torch.roll(state, shifts=1, dims=-1)
        x_ip1 = torch.roll(state, shifts=-1, dims=-1)
        return (x_ip1 - x_im2) * x_im1 - state + self.forcing

    # ---------------------------------------------------------------- measurement
    def measure(self, state: Tensor) -> Tensor:
        projected = state @ self.H.T
        transformed = projected.abs().clamp_min(self.measurement_eps)
        transformed = transformed.pow(self.measurement_power)
        return torch.sign(projected) * transformed

    # ---------------------------------------------------------------- dynamics
    def run_step(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        next_state = self._integrate_without_noise(state)
        return next_state, self.measure(next_state)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_batch(
        self,
        T: int,
        B: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        X = torch.empty(B, T, self.state_dim, device=self.device, dtype=self.dtype)
        Y = torch.empty_like(X)

        x0 = torch.randn(B, self.state_dim, device=self.device, dtype=self.dtype) * self.initial_state_std
        process_noise = self.process_noise.rsample((B, T))
        measurement_noise = self.measurement_noise.rsample((B, T))

        state = x0
        for t in range(T):
            deterministic = self._integrate_without_noise(state)
            state = deterministic + process_noise[:, t]
            X[:, t] = state
            Y[:, t] = self.measure(state) + measurement_noise[:, t]

        return X, Y, x0
