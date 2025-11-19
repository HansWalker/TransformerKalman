from __future__ import annotations
from typing import Tuple
import torch
import torch.distributions as dist
from Scenarios.abstract_scenario import Scenario

Tensor = torch.Tensor


class RandomWalk(Scenario):
    """
    Independent random walk in every dimension:
        x_{t+1} = x_t + w_t,   y_t = x_t + v_t.
    The code intentionally mirrors the mathematical description so the scenario
    doubles as documentation.
    """

    def __init__(
        self,
        *,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        step_std: float = 1.0,
        measurement_std: float = 1.0,
        initial_state_std: float = 1.0,
    ):
        self._state_dim = int(dimension)
        self.device, self.dtype = device, dtype

        self.step_std = float(step_std)
        self.measurement_std = float(measurement_std)
        self.initial_state_std = float(initial_state_std)

        identity = torch.eye(self.state_dim, device=device, dtype=dtype)
        self.Q = (self.step_std ** 2) * identity
        self.R = (self.measurement_std ** 2) * identity
        self.P0 = (self.initial_state_std ** 2) * identity
        self.H = identity.clone()

        zeros = torch.zeros(self.state_dim, device=device, dtype=dtype)
        self.process_noise = dist.MultivariateNormal(loc=zeros, covariance_matrix=self.Q)
        self.measurement_noise = dist.MultivariateNormal(loc=zeros, covariance_matrix=self.R)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    # ---------------------------------------------------------------- measurement
    def measure(self, state: Tensor) -> Tensor:
        return state

    def run_step(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        measurement = self.measure(state)
        return state, measurement

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
            state = state + process_noise[:, t]
            X[:, t] = state
            Y[:, t] = self.measure(state) + measurement_noise[:, t]

        return X, Y, x0
