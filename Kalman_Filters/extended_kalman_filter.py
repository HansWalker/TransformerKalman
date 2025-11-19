from __future__ import annotations
from typing import Tuple, Any, Callable
import torch
from torch import func as torch_func
from tqdm.auto import tqdm

Tensor = torch.Tensor


class EKF:
    """
    Scenario-agnostic Extended Kalman Filter implemented with PyTorch autograd.
    The scenario must expose `run_step` and `measure`.
    """

    def __init__(self, scenario: Any, Q: Tensor, R: Tensor, P0: Tensor, jitter: float = 1e-6):
        self.scenario = scenario
        self.state_dim = Q.shape[0]
        self.measure_dim = R.shape[0]
        self.jitter = float(jitter)

        self.device: torch.device = getattr(scenario, "device", torch.device("cpu"))
        self.dtype: torch.dtype = getattr(scenario, "dtype", torch.float32)

        self.Q = Q.to(self.device, self.dtype)
        self.R = R.to(self.device, self.dtype)
        self.P0 = P0.to(self.device, self.dtype)

        self.eye_state = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        self.eye_measure = torch.eye(self.measure_dim, device=self.device, dtype=self.dtype)


    @torch.no_grad()
    def run(self, Y: Tensor, x0: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, _ = Y.shape
        dev, dtp = Y.device, Y.dtype

        means = x0.to(self.device, self.dtype)
        covs = self.P0.unsqueeze(0).expand(B, self.state_dim, self.state_dim).clone()

        Xhat = torch.empty(B, T, self.state_dim, device=self.device, dtype=self.dtype)
        P_all = torch.empty(B, T, self.state_dim, self.state_dim, device=self.device, dtype=self.dtype)

        scenario_measure = self.scenario.measure

        dyn_jacobian = getattr(self.scenario, "state_jacobian", None)
        meas_jacobian = getattr(self.scenario, "measurement_jacobian", None)

        Q_batch = self.Q.unsqueeze(0).expand(B, -1, -1)
        R_batch = self.R.unsqueeze(0).expand(B, -1, -1)
        eye_batch = self.eye_state.unsqueeze(0).expand(B, -1, -1)
        eye_measure_batch = self.eye_measure.unsqueeze(0).expand(B, -1, -1)

        for t in tqdm(range(T), desc="EKF", leave=False):
            x_pred_batch, z_pred_batch = self.scenario.run_step(means)
            x_pred_batch = x_pred_batch.to(self.device, self.dtype).reshape(B, self.state_dim)
            z_pred_batch = z_pred_batch.to(self.device, self.dtype).reshape(B, self.measure_dim)

            if callable(dyn_jacobian):
                F_batch = dyn_jacobian(means)
            else:
                with torch.enable_grad():
                    F_batch = self._batched_jacobian(
                        lambda inp: self.scenario.run_step(inp)[0],
                        means,
                    )
            F_batch = F_batch.to(self.device, self.dtype)

            if callable(meas_jacobian):
                H_batch = meas_jacobian(x_pred_batch)
            else:
                with torch.enable_grad():
                    H_batch = self._batched_jacobian(
                        lambda inp: scenario_measure(inp),
                        x_pred_batch,
                    )
            H_batch = H_batch.to(self.device, self.dtype)

            P_pred = torch.matmul(torch.matmul(F_batch, covs), F_batch.transpose(-1, -2)) + Q_batch

            z_obs = Y[:, t].to(self.device, self.dtype)
            S = torch.matmul(torch.matmul(H_batch, P_pred), H_batch.transpose(-1, -2)) + R_batch + self.jitter * eye_measure_batch
            PHt = torch.matmul(P_pred, H_batch.transpose(-1, -2))
            K = torch.linalg.solve(S, PHt.transpose(-1, -2)).transpose(-1, -2)

            innovation = z_obs - z_pred_batch
            mean_upd = x_pred_batch + torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1)

            I_KH = eye_batch - torch.matmul(K, H_batch)
            cov_upd = torch.matmul(torch.matmul(I_KH, P_pred), I_KH.transpose(-1, -2)) + torch.matmul(
                torch.matmul(K, R_batch), K.transpose(-1, -2)
            )

            means = mean_upd
            covs = cov_upd

            Xhat[:, t] = mean_upd
            P_all[:, t] = cov_upd

        return (
            Xhat.to(device=dev, dtype=dtp).detach(),
            P_all.to(device=dev, dtype=dtp).detach(),
        )

    def _batched_jacobian(self, func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
        x = x.detach().requires_grad_(True)

        def single(inp: Tensor) -> Tensor:
            inp = inp.unsqueeze(0)
            out = func(inp)
            return out.squeeze(0)

        jac_single = torch_func.jacrev(single)
        jacobian = torch_func.vmap(jac_single)(x)
        return jacobian
