from __future__ import annotations
from typing import Tuple, Any
import torch

Tensor = torch.Tensor

class SIRPF:
    """
    Scenario-agnostic bootstrap Particle Filter (SIR resampling).

    Scenario (batched) contract (same as your other filters):
      - run_step(X:(B,N)) -> (X_next:(B,N), Z_pred:(B,M))   # ΦΔ(X), h(ΦΔ(X))
      - measure(X:(B,N))  -> Z:(B,M)                        # h(X)

    Constructor args:
      scenario : object with .run_step and .measure (batched)
      Q, R, P0 : (N,N), (M,M), (N,N) torch tensors
      J        : number of particles
      ess_frac : resample when ESS < ess_frac * J  (e.g., 0.5)
      jitter   : small diagonal added to R when Cholesky is ill-conditioned

    run(Y, x0) returns:
      Xhat  (B,T,N): filtered means   [torch: same device/dtype as Y]
      P_all (B,T,N,N): filtered covariances (sample covariance of particles)
    """

    def __init__(
        self,
        scenario: Any,
        Q: Tensor,
        R: Tensor,
        P0: Tensor,
        J: int = 256,
        ess_frac: float = 0.5,
        jitter: float = 1e-8,
    ):
        self.scn, self.Q, self.R, self.P0 = scenario, Q, R, P0
        self.N, self.M = Q.shape[0], R.shape[0]
        self.J = int(J)
        self.ess_frac = float(ess_frac)
        self.jitter = float(jitter)

        # Zero-mean Gaussians for process/prior noise (cov shared across batch)
        MVN = torch.distributions.MultivariateNormal
        self._proc = MVN(loc=torch.zeros(self.N, device=Q.device, dtype=Q.dtype),
                         covariance_matrix=Q)
        self._prior = MVN(loc=torch.zeros(self.N, device=P0.device, dtype=P0.dtype),
                          covariance_matrix=P0)

    @torch.no_grad()
    def run(self, Y: Tensor, x0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Y:  (B,T,M)
        x0: (B,N)
        """
        B, T, _ = Y.shape
        N, M, J = self.N, self.M, self.J

        dev, dtp = Y.device, Y.dtype

        # outputs
        Xhat  = torch.empty(B, T, N, device=dev, dtype=dtp)
        P_all = torch.empty(B, T, N, N, device=dev, dtype=dtp)

        # ---- initialize particles and weights ----
        # particles: (B,J,N); draw prior noise once and shift by x0
        Xp = x0[:, None, :] + self._prior.rsample((B, J)).to(device=dev, dtype=dtp)
        w  = torch.full((B, J), 1.0 / J, device=dev, dtype=dtp)

        # ---- precompute measurement-metric (Cholesky of R) for log-likelihood ----
        # Add tiny jitter for numerical safety if needed.
        R = self.R.to(device=dev, dtype=dtp)
        try:
            L = torch.linalg.cholesky(R)                        # (M,M)
        except RuntimeError:
            L = torch.linalg.cholesky(R + self.jitter * torch.eye(M, device=dev, dtype=dtp))
        log_norm_const = -0.5 * (M * torch.log(torch.tensor(2 * torch.pi, device=dev, dtype=dtp))
                                 + 2.0 * torch.log(torch.diag(L)).sum())

        def _log_gauss(y: Tensor, mu: Tensor) -> Tensor:
            """
            y:  (..., M)
            mu: (..., M)
            returns log N(y | mu, R) elementwise over leading dims
            """
            # solve L v = (y - mu)
            v = torch.cholesky_solve((y - mu).unsqueeze(-1), L).squeeze(-1)  # (..., M)
            quad = torch.sum((y - mu) * v, dim=-1)                            # (...)
            return log_norm_const - 0.5 * quad

        # ---- time loop ----
        ess_threshold = self.ess_frac * J

        for t in range(T):
            # ---------- propagate particles ----------
            # Flatten (B,J,N) -> (B*J,N) to reuse scenario.run_step
            Xp_flat = Xp.reshape(B * J, N)
            Xp_next, Zp_pred = self.scn.run_step(Xp_flat)   # (B*J,N), (B*J,M)
            Xp = Xp_next.reshape(B, J, N)
            Zp = Zp_pred.reshape(B, J, M)

            # add process noise (Brownian style; scenario uses deterministic ΦΔ)
            Xp = Xp + self._proc.rsample((B, J)).to(device=dev, dtype=dtp)

            # ---------- weight update ----------
            # y_t: (B,M) -> broadcast to (B,J,M); compute log-likelihood per particle
            y_t = Y[:, t].unsqueeze(1).expand(B, J, M)
            logw_inc = _log_gauss(y_t, Zp)                    # (B,J)

            # accumulate in log-domain, then normalize
            logw = torch.log(w + 1e-45) + logw_inc
            logw = logw - torch.amax(logw, dim=1, keepdim=True)  # avoid overflow
            w = torch.exp(logw)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-45)

            # ---------- compute estimate ----------
            # mean
            mu = torch.einsum('bj,bjn->bn', w, Xp)            # (B,N)
            Xhat[:, t] = mu

            # covariance (weighted, unbiased-ish)
            Xc = Xp - mu.unsqueeze(1)                         # (B,J,N)
            # Use weights on the left: P = Xc^T diag(w) Xc
            P = torch.einsum('bjn,bjm,bj->bnm', Xc, Xc, w)    # (B,N,N)
            P_all[:, t] = P

            # ---------- resample (systematic) if ESS low ----------
            ess = 1.0 / (w.pow(2).sum(dim=1) + 1e-45)         # (B,)
            need = ess < ess_threshold
            if need.any():
                Xp = self._systematic_resample(Xp, w, need)   # (B,J,N)
                # reset weights to uniform where resampled
                w[need] = 1.0 / J

        return Xhat, P_all

    # ---- systematic resampling (batched) ----
    @staticmethod
    def _systematic_resample(Xp: Tensor, w: Tensor, need_mask: Tensor) -> Tensor:
        """
        Xp:        (B,J,N)
        w:         (B,J)    (sums to 1 per batch)
        need_mask: (B,) boolean; only those batches get resampled
        returns new particles with equal weights (uniform)
        """
        B, J, N = Xp.shape
        device, dtype = Xp.device, Xp.dtype

        # if none need resampling, return early
        if not need_mask.any():
            return Xp

        # Work on only the selected batches to save a bit of math
        idx = torch.nonzero(need_mask, as_tuple=False).squeeze(1)  # (B_sel,)
        X_sel = Xp[idx]             # (B_sel, J, N)
        w_sel = w[idx]              # (B_sel, J)

        # CDF
        cdf = torch.cumsum(w_sel, dim=1)                          # (B_sel, J)

        # systematic uniforms u_k = u0 + k/J, u0 ~ U[0, 1/J)
        Bsel = X_sel.shape[0]
        u0 = torch.rand(Bsel, 1, device=device, dtype=dtype) / J  # (B_sel,1)
        u = u0 + (torch.arange(J, device=device, dtype=dtype).unsqueeze(0) / J)  # (B_sel,J)

        # for each u, find first index where cdf >= u
        # torch.searchsorted operates on last dimension
        idxs = torch.searchsorted(cdf, u, right=False)            # (B_sel, J)
        idxs = torch.clamp(idxs, 0, J - 1)

        # gather new particles
        arange_b = torch.arange(Bsel, device=device).unsqueeze(1).expand(Bsel, J)
        X_new = X_sel[arange_b, idxs, :]                          # (B_sel, J, N)

        # scatter back
        Xp[idx] = X_new
        return Xp