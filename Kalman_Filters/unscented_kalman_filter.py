from __future__ import annotations
from typing import Tuple, Any
import torch
from tqdm.auto import tqdm

Tensor = torch.Tensor


class UKF:
    """
    Scenario-agnostic Unscented Kalman Filter implemented in PyTorch.
    Requires the scenario to expose `run_step` and `measure`.
    """

    def __init__(
        self,
        scenario: Any,
        Q: Tensor,
        R: Tensor,
        P0: Tensor,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        jitter: float = 1e-6,
    ):
        self.scenario = scenario
        self.state_dim = Q.shape[0]
        self.measure_dim = R.shape[0]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.jitter = float(jitter)

        self.scenario_device: torch.device = getattr(
            scenario, "device", torch.device("cpu")
        )
        self.scenario_dtype: torch.dtype = getattr(
            scenario, "dtype", torch.float32
        )

        self.Q = Q.to(self.scenario_device, self.scenario_dtype)
        self.R = R.to(self.scenario_device, self.scenario_dtype)
        self.P0 = P0.to(self.scenario_device, self.scenario_dtype)

        self.lambda_ = self.alpha ** 2 * (self.state_dim + self.kappa) - self.state_dim
        weight_mean = torch.full(
            (2 * self.state_dim + 1,),
            0.5 / (self.state_dim + self.lambda_),
            device=self.scenario_device,
            dtype=self.scenario_dtype,
        )
        weight_cov = weight_mean.clone()
        weight_mean[0] = self.lambda_ / (self.state_dim + self.lambda_)
        weight_cov[0] = weight_mean[0] + (1 - self.alpha ** 2 + self.beta)
        self.Wm = weight_mean
        self.Wc = weight_cov

    @torch.no_grad()
    def run(self, Y: Tensor, x0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            Y:  (B, T, M) measurements
            x0: (B, N) initial states
        Returns:
            Xhat:  (B, T, N)
            P_all: (B, T, N, N)
        """
        B, T, _ = Y.shape
        dev, dtp = Y.device, Y.dtype

        Y_scn = Y.to(self.scenario_device, self.scenario_dtype)
        means = x0.to(self.scenario_device, self.scenario_dtype)
        covs = self.P0.unsqueeze(0).expand(B, self.state_dim, self.state_dim).clone()

        Xhat = torch.empty(B, T, self.state_dim, device=self.scenario_device, dtype=self.scenario_dtype)
        P_all = torch.empty(B, T, self.state_dim, self.state_dim, device=self.scenario_device, dtype=self.scenario_dtype)

        for t in tqdm(range(T), desc="UKF", leave=False):
            sigma_points = self._sigma_points_batch(means, covs)
            propagated, _ = self.scenario.run_step(sigma_points.reshape(B * (2 * self.state_dim + 1), self.state_dim))
            propagated = propagated.reshape(B, 2 * self.state_dim + 1, self.state_dim)

            means, covs = self._unscented_transform_batch(propagated, self.Q)
            means, covs = self._update_batch(means, covs, propagated, Y_scn[:, t])

            Xhat[:, t] = means
            P_all[:, t] = covs

        return (
            Xhat.to(device=dev, dtype=dtp),
            P_all.to(device=dev, dtype=dtp),
        )

    def _sigma_points_batch(self, means: Tensor, covs: Tensor) -> Tensor:
        B = means.shape[0]
        mat = self._make_pd(covs)
        eigvals, eigvecs = torch.linalg.eigh(mat)
        eps = torch.finfo(mat.dtype).eps
        eigvals = torch.clamp(eigvals, min=eps)
        sqrt_mat = eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
        scale = torch.sqrt(torch.tensor(self.state_dim + self.lambda_, device=mat.device, dtype=mat.dtype))
        sqrt_mat = sqrt_mat * scale

        sigmas = torch.empty(B, 2 * self.state_dim + 1, self.state_dim, device=self.scenario_device, dtype=self.scenario_dtype)
        sigmas[:, 0] = means
        for k in range(self.state_dim):
            offset = sqrt_mat[:, :, k]
            sigmas[:, k + 1] = means + offset
            sigmas[:, self.state_dim + k + 1] = means - offset
        return sigmas

    def _unscented_transform_batch(self, sigma: Tensor, noise: Tensor) -> Tuple[Tensor, Tensor]:
        mean = torch.sum(self.Wm.view(1, -1, 1) * sigma, dim=1)
        diff = sigma - mean.unsqueeze(1)
        cov = torch.einsum("l,bln,blm->bnm", self.Wc, diff, diff)
        cov = cov + noise.unsqueeze(0)
        cov = self._make_pd(cov)
        return mean, cov

    def _update_batch(self, means: Tensor, covs: Tensor, sigma: Tensor, measurement: Tensor) -> Tuple[Tensor, Tensor]:
        B = means.shape[0]
        z_sigma = self.scenario.measure(sigma.reshape(B * (2 * self.state_dim + 1), self.state_dim))
        z_sigma = z_sigma.reshape(B, 2 * self.state_dim + 1, self.measure_dim)

        z_mean, S = self._unscented_transform_batch(z_sigma, self.R)
        diff_x = sigma - means.unsqueeze(1)
        diff_z = z_sigma - z_mean.unsqueeze(1)
        P_xz = torch.einsum("l,bln,blm->bnm", self.Wc, diff_x, diff_z)

        gain = torch.linalg.solve(S, P_xz.transpose(-1, -2)).transpose(-1, -2)
        residual = measurement - z_mean
        means = means + torch.einsum("bnm,bm->bn", gain, residual)
        KS = torch.matmul(gain, S)
        covs = covs - torch.matmul(KS, gain.transpose(-1, -2))
        covs = self._make_pd(covs)
        return means, covs

    def _make_pd(self, matrix: Tensor) -> Tensor:
        matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eps = torch.finfo(matrix.dtype).eps
        eigvals = torch.clamp(eigvals, min=eps)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
