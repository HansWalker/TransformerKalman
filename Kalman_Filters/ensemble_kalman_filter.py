from __future__ import annotations
from typing import Tuple, Any, Optional
import numpy as np
import torch
from tqdm.auto import tqdm

Tensor = torch.Tensor


class EnKF:
    """
    Scenario-agnostic Ensemble Kalman Filter implemented fully in PyTorch.
    Keeps an ensemble of size `ensemble_size` for each batch element and
    propagates them through the provided scenario.
    """

    def __init__(
        self,
        scenario: Any,
        Q: Tensor,
        R: Tensor,
        P0: Tensor,
        ensemble_size: int = 32,
        inflation: float = 1.0,
        localization_radius: int = 0,
        jitter: float = 1e-6,
    ):
        self.scenario = scenario
        self.state_dim = Q.shape[0]
        self.measure_dim = R.shape[0]
        self.ensemble_size = int(max(2, ensemble_size))
        self.inflation = float(max(1.0, inflation))
        self.jitter = float(jitter)

        # Scenario device/dtype controls how we call run_step/measure.
        self.scenario_device: torch.device = getattr(
            scenario, "device", torch.device("cpu")
        )
        self.scenario_dtype: torch.dtype = getattr(
            scenario, "dtype", torch.float32
        )

        self.Q = Q.to(self.scenario_device, self.scenario_dtype)
        self.R = R.to(self.scenario_device, self.scenario_dtype)
        self.P0 = P0.to(self.scenario_device, self.scenario_dtype)

        eye_N = torch.eye(self.state_dim, device=self.scenario_device, dtype=self.scenario_dtype)
        eye_M = torch.eye(self.measure_dim, device=self.scenario_device, dtype=self.scenario_dtype)

        self.chol_Q = torch.linalg.cholesky(self.Q + self.jitter * eye_N)
        self.chol_R = torch.linalg.cholesky(self.R + self.jitter * eye_M)
        self.chol_P0 = torch.linalg.cholesky(self.P0 + self.jitter * eye_N)

        self.localization: Optional[torch.Tensor] = None
        if localization_radius > 0:
            self.localization = self._build_localization(self.state_dim, localization_radius).to(
                self.scenario_device, self.scenario_dtype
            )

    @torch.no_grad()
    def run(self, Y: Tensor, x0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            Y:  (B, T, M) measurement sequences
            x0: (B, N) initial states
        Returns:
            (Xhat, P_all)
            Xhat:  (B, T, N) filtered means
            P_all: (B, T, N, N) filtered covariances
        """
        B, T, _ = Y.shape
        dev_out, dtype_out = Y.device, Y.dtype
        scenario_compute_dtype = getattr(self.scenario, "compute_dtype", self.scenario_dtype)
        work_dtype = torch.promote_types(dtype_out, scenario_compute_dtype)
        Y_work = Y.to(dev_out, work_dtype)
        x0_work = x0.to(dev_out, work_dtype)
        J = self.ensemble_size
        N, M = self.state_dim, self.measure_dim

        Xhat = torch.empty(B, T, N, device=dev_out, dtype=work_dtype)
        P_all = torch.empty(B, T, N, N, device=dev_out, dtype=work_dtype)

        ensemble = self._sample_initial_ensemble(x0_work)

        for t in tqdm(range(T), desc="EnKF", leave=False):
            ensemble = self._predict(ensemble, dev_out)
            if self.inflation > 1.0:
                ensemble = self._apply_inflation(ensemble)

            ensemble = self._update(
                ensemble,
                measurements=Y_work[:, t],
                dev=dev_out,
            )

            mean = ensemble.mean(dim=1)
            cov = self._cov_from_ensemble(ensemble)
            Xhat[:, t] = mean
            P_all[:, t] = cov

        return Xhat.to(dev_out, dtype_out), P_all.to(dev_out, dtype_out)

    # ---------------------------------------------------------------- helper methods
    def _sample_initial_ensemble(self, x0: Tensor) -> Tensor:
        B = x0.shape[0]
        eps = torch.randn(
            B, self.ensemble_size, self.state_dim, device=x0.device, dtype=x0.dtype
        )
        chol = self.chol_P0.to(x0.device, x0.dtype)
        return x0.unsqueeze(1) + eps @ chol.T

    def _predict(self, ensemble: Tensor, dev: torch.device) -> Tensor:
        B, J, N = ensemble.shape
        flat = ensemble.reshape(B * J, N).to(self.scenario_device, self.scenario_dtype)
        next_state, _ = self.scenario.run_step(flat)
        next_state = next_state.reshape(B, J, N).to(dev, ensemble.dtype)

        noise = torch.randn(B, J, N, device=dev, dtype=ensemble.dtype)
        noise = noise @ self.chol_Q.to(dev, ensemble.dtype).T
        return next_state + noise

    def _measure_ensemble(self, ensemble: Tensor) -> Tensor:
        B, J, N = ensemble.shape
        flat = ensemble.reshape(B * J, N).to(self.scenario_device, self.scenario_dtype)
        measurements = self.scenario.measure(flat).reshape(B, J, self.measure_dim)
        return measurements.to(ensemble.device, ensemble.dtype)

    def _apply_inflation(self, ensemble: Tensor) -> Tensor:
        mean = ensemble.mean(dim=1, keepdim=True)
        return mean + (ensemble - mean) * self.inflation

    def _cov_from_ensemble(self, ensemble: Tensor) -> Tensor:
        anomalies = ensemble - ensemble.mean(dim=1, keepdim=True)
        denom = max(self.ensemble_size - 1, 1)
        cov = torch.einsum("bjn,bjm->bnm", anomalies, anomalies) / denom
        if self.localization is not None:
            cov = cov * self.localization.to(cov.device, cov.dtype)
        return cov

    def _update(self, ensemble: Tensor, measurements: Tensor, dev: torch.device) -> Tensor:
        B, J, N = ensemble.shape
        M = self.measure_dim

        Z = self._measure_ensemble(ensemble)
        z_mean = Z.mean(dim=1, keepdim=True)
        x_mean = ensemble.mean(dim=1, keepdim=True)

        A = ensemble - x_mean
        C = Z - z_mean

        denom = max(self.ensemble_size - 1, 1)
        P_xz = torch.einsum("bjn,bjm->bnm", A, C) / denom
        P_zz = torch.einsum("bjm,bjn->bmn", C, C) / denom

        P_zz = P_zz + self.R.to(dev, ensemble.dtype)

        if self.localization is not None:
            loc = self.localization.to(dev, ensemble.dtype)
            P_xz = torch.einsum("ij,bjm->bim", loc, P_xz)

        gain = torch.linalg.solve(
            P_zz,
            P_xz.transpose(1, 2),
        ).transpose(1, 2)

        meas_noise = torch.randn(B, J, M, device=dev, dtype=ensemble.dtype)
        meas_noise = meas_noise @ self.chol_R.to(dev, ensemble.dtype).T

        innovation = measurements.unsqueeze(1) + meas_noise - Z
        delta = torch.einsum("bjm,bmn->bjn", innovation, gain)
        return ensemble + delta

    @staticmethod
    def _build_localization(N: int, radius: int) -> torch.Tensor:
        idx = np.arange(N)
        d = np.abs(idx[:, None] - idx[None, :])
        d = np.minimum(d, N - d)
        a = d / float(radius)
        w = np.zeros_like(a, dtype=float)
        m1 = a <= 1.0
        m2 = (a > 1.0) & (a <= 2.0)
        am1 = a[m1]
        am2 = a[m2]
        w[m1] = 1 - (5 / 3) * am1**2 + (5 / 8) * am1**3 + 0.5 * am1**4 - 0.25 * am1**5
        w[m2] = (
            4
            - 5 * am2
            + (5 / 3) * am2**2
            + (5 / 8) * am2**3
            - 0.5 * am2**4
            + (1 / 12) * am2**5
        ) / 12.0
        w = np.clip(w, 0.0, 1.0)
        return torch.from_numpy(w)
