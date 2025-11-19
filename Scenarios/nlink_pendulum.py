from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.distributions as dist
from Scenarios.abstract_scenario import Scenario

Tensor = torch.Tensor


class NLinkPendulum(Scenario):
    """
    Planar pendulum with N point masses and massless links of equal length.
    State ordering:
        [theta_1 .. theta_N, omega_1 .. omega_N]
    where theta is the angle from the downward vertical and omega is angular velocity.
    """

    def __init__(
        self,
        #default args
        dimension: int,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        time_step: float = 0.01,
        substeps: int = 1,
        #system args
        length: float = 1.0,
        mass: float = 1.0,
        g: float = 9.81,
        process_var: float = 1.0,
        #observation args
        observation_var: float = 1.0,
        initial_state_var: float = 1.0,
        p: float = 1.0,
    ):
        self.N = int(dimension)
        if self.N <= 0:
            raise ValueError("dimension must be positive")
        self._state_dim = 2 * self.N
        self.device, self.dtype = device, dtype

        self.time_step = float(time_step)
        self.substeps = max(int(substeps), 1)
        self.gravity = float(g)

        self.measurement_power = float(p)

        identity = torch.eye(self._state_dim, device=device, dtype=dtype)
        self.Q = process_var * identity
        self.R = observation_var * identity
        self.P0 = initial_state_var * identity
        self.H = identity.clone()

        self.link_lengths = torch.full((self.N,), length, device=self.device, dtype=self.dtype)
        self.link_masses = torch.full((self.N,), mass, device=self.device, dtype=self.dtype)
        self.setup_numerical_helpers()
        
        self.jitter_eye = 10**-8 * torch.eye(self.N, device=self.device, dtype=self.dtype)
        zeros = torch.zeros(self.state_dim, device=device, dtype=dtype)

        #Probability distributions
        self.process_noise = dist.MultivariateNormal(loc=zeros, covariance_matrix=self.Q)
        self.measurement_noise = dist.MultivariateNormal(loc=zeros, covariance_matrix=self.R)
        self.initial_state_noise = dist.MultivariateNormal(loc=zeros, covariance_matrix=self.P0)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def setup_numerical_helpers(self):
        masses = self.link_masses.to(self.dtype)
        lengths = self.link_lengths.to(self.dtype)
        tail = torch.flip(torch.cumsum(torch.flip(masses, dims=(0,)), dim=0), dims=(0,))
        idx = torch.arange(self.N, device=self.device)

        self.tail_lookup = tail[torch.maximum(idx[:, None], idx[None, :])]
        self.mass_coeff = self.tail_lookup * torch.outer(lengths, lengths)
        self.gravity_coeff = (tail * self.gravity * lengths).view(1, self.N)

    # ---------------------------------------------------------------- helpers
    def _state_derivative(self, state: Tensor) -> Tensor:
        state = state.to(self.dtype)
        theta = state[..., : self.N]
        omega = state[..., self.N :]

        # Mass matrix M(theta) = tail_mass * l_i l_r * (cos theta_i cos theta_r + sin theta_i sin theta_r).
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        coeff = self.mass_coeff.view(1, self.N, self.N)
        coscos = cos_t.unsqueeze(2) * cos_t.unsqueeze(1)
        sinsin = sin_t.unsqueeze(2) * sin_t.unsqueeze(1)
        M = coeff * (coscos + sinsin)

        # Coriolis/Centrifugal term: C_{ir}(theta) * omega_r^2, where
        #   C_{ir} = tail_mass * l_i l_r * (-cos theta_i sin theta_r + sin theta_i cos theta_r).
        cross = (-cos_t.unsqueeze(2) * sin_t.unsqueeze(1)) + (sin_t.unsqueeze(2) * cos_t.unsqueeze(1))
        omega_sq = omega.pow(2).unsqueeze(1)

        # Gravity term G_i(theta) = T_i * g * l_i * sin theta_i combined with velocity contributions.
        rhs = -(self.gravity_coeff * sin_t) - (coeff * cross * omega_sq).sum(dim=2)

        # Solve (M + epsilon I) theta_ddot = rhs for each batch element.
        theta_ddot = torch.linalg.solve(
            M + self.jitter_eye.view(1, self.N, self.N),
            rhs.unsqueeze(2),
        ).squeeze(2)

        # Return concatenated [theta_dot, theta_ddot].
        return torch.cat([omega, theta_ddot], dim=-1)

    #Simulate 1 timestep
    def _integrate_without_noise(self, state: Tensor) -> Tensor:
        step = self.time_step / self.substeps
        current = state.to(self.dtype)
        for _ in range(self.substeps):
            current = self._rk4_step(current, step)
        return current.to(state.dtype)

    def _rk4_step(self, state: Tensor, step: float) -> Tensor:
        k1 = self._state_derivative(state)
        k2 = self._state_derivative(state + 0.5 * step * k1)
        k3 = self._state_derivative(state + 0.5 * step * k2)
        k4 = self._state_derivative(state + step * k3)
        updated = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return updated

    # ---------------------------------------------------------------- measurement
    def measure(self, state: Tensor) -> Tensor:
        projected = state @ self.H.T
        return torch.sign(projected) * projected.abs().pow(self.measurement_power)

    # --------------------------------------------------------- noiseless dynamics
    def run_step(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        state_internal = state.to(self.dtype)
        next_state = self._integrate_without_noise(state_internal)
        next_state_out = next_state.to(self.dtype)
        return next_state_out, self.measure(next_state_out)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_batch(
        self,
        T: int,
        B: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        X = torch.empty(B, T, self.state_dim, device=self.device, dtype=self.dtype)
        Y = torch.empty_like(X)
        
        #Get initial state and additive gaussian noise
        x0 = self.initial_state_noise.rsample((B,)).to(self.dtype)
        state = x0.clone()

        process_noise = self.process_noise.rsample((B, T)).to(self.dtype)
        measurement_noise = self.measurement_noise.rsample((B, T)).to(self.dtype)

        #Run the simulation
        for t in range(T):
            #Advancing to the next step
            deterministic_step = self._integrate_without_noise(state)
            state = deterministic_step + process_noise[:, t]

            #saving the state
            state_out = state.to(self.dtype)
            X[:, t] = state_out
            Y[:, t] = self.measure(state_out) + measurement_noise[:, t]

        return X, Y, x0
