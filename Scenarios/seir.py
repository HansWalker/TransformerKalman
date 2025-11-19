from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.distributions as dist
from Scenarios.abstract_scenario import Scenario

Tensor = torch.Tensor


class SEIR(Scenario):
    """
    Multi-region Susceptible-Exposed-Infectious-Recovered model.  Each region keeps
    a 4-element state that always sums to one.  The code mirrors the textbook ODEs:
        dS = -beta * S * I_eff
        dE =  beta * S * I_eff - sigma * E
        dI =  sigma * E - gamma * I
        dR =  gamma * I
    Inter-region coupling is optional and defaults to the identity (no mixing).
    """

    def __init__(
        self,
        *,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        time_step: float = 0.05,
        substeps: int = 1,
        p: float = 1.0,
        eps: float = 1e-6,
        process_var: float = 1.0,
        observation_var: float = 1.0,
        initial_state_var: float = 1.0,
        beta: float = 0.6,
        sigma: float = 0.2,
        gamma: float = 0.1,
        coupling: Optional[Tensor] = None,
        enforce_simplex: bool = True,
    ):
        self.regions = int(dimension)
        self._state_dim = 4 * self.regions
        self.device = device
        self.dtype = dtype

        self.time_step = float(time_step)
        self.substeps = max(int(substeps), 1)
        self.measurement_power = float(p)
        self.measurement_eps = float(eps)

        self.beta = float(beta)
        self.sigma = float(sigma)
        self.gamma = float(gamma)

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

        zero = torch.zeros(self.state_dim, device=device, dtype=dtype)
        self.process_noise = dist.MultivariateNormal(loc=zero, covariance_matrix=self.Q)
        self.measurement_noise = dist.MultivariateNormal(loc=zero, covariance_matrix=self.R)

        if coupling is None:
            self.coupling = torch.eye(self.regions, device=device, dtype=dtype)
        else:
            C = coupling.to(device=device, dtype=dtype)
            row_sums = C.sum(dim=1, keepdim=True).clamp_min(1e-12)
            self.coupling = C / row_sums

        self.enforce_simplex = bool(enforce_simplex)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def _project_simplex(self, structured_state: Tensor) -> Tensor:
        state = structured_state.clamp_min(0.0)
        total = state.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return state / total

    def _state_derivative(self, flat_state: Tensor) -> Tensor:
        structured = flat_state.reshape(flat_state.shape[0], self.regions, 4)
        if self.enforce_simplex:
            structured = self._project_simplex(structured)
        S = structured[..., 0]
        E = structured[..., 1]
        I = structured[..., 2]
        Rv = structured[..., 3]
        I_eff = I @ self.coupling.T

        dS = -self.beta * S * I_eff
        dE = self.beta * S * I_eff - self.sigma * E
        dI = self.sigma * E - self.gamma * I
        dR = self.gamma * I

        derivatives = torch.stack([dS, dE, dI, dR], dim=-1)
        return derivatives.reshape(flat_state.shape[0], self.state_dim)

    def _integrate_without_noise(self, state: Tensor) -> Tensor:
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
        updated = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.enforce_simplex:
            structured = updated.reshape(updated.shape[0], self.regions, 4)
            projected = self._project_simplex(structured)
            updated = projected.reshape(updated.shape[0], self.state_dim)
        return updated

    def _initial_state(self, batch: int) -> Tensor:
        S = torch.full((batch, self.regions), 0.99, device=self.device, dtype=self.dtype)
        E = torch.zeros_like(S)
        I = (0.01 * torch.rand_like(S)).clamp_min(1e-4)
        Rv = torch.zeros_like(S)
        structured = torch.stack([S, E, I, Rv], dim=-1)
        structured = self._project_simplex(structured)
        return structured.reshape(structured.shape[0], self.state_dim)

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

        x0 = self._initial_state(B)
        dt_scale = (self.time_step ** 0.5)
        process_noise = self.process_noise.rsample((B, T)) * dt_scale
        measurement_noise = self.measurement_noise.rsample((B, T))

        state = x0
        for t in range(T):
            deterministic = self._integrate_without_noise(state)
            state = deterministic + process_noise[:, t]
            if self.enforce_simplex:
                structured = state.reshape(state.shape[0], self.regions, 4)
                projected = self._project_simplex(structured)
                state = projected.reshape(state.shape[0], self.state_dim)
            X[:, t] = state
            Y[:, t] = self.measure(state) + measurement_noise[:, t]

        return X, Y, x0
