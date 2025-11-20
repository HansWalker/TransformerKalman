from __future__ import annotations
import abc
from typing import Tuple
import torch

Tensor = torch.Tensor


class Scenario(abc.ABC):
    """
    Abstract base class for a dynamical system scenario.

    Contract used across the codebase:
    - Subclasses populate Q, R, P0, H on self.device/self.dtype.
    - run_step advances a batch of states and returns the noiseless measurement.
    - measure applies the observation model to a batch of states.
    - sample_batch draws noisy rollouts for training/evaluation.
    Current implementations assume measurement_dim == state_dim via a square H.
    """

    @abc.abstractmethod
    def __init__(
        self,
        *,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        Each subclass should construct internal dynamics plus matrices Q/R/P0/H
        placed on the requested device/dtype.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def measure(self, state: Tensor) -> Tensor:
        """
        Apply the deterministic observation model h(x) for a batch of states.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_step(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Advance the process model one step (without process noise) and return:
        - next state            (B, state_dim)
        - corresponding h(next) (B, measure_dim)
        """
        raise NotImplementedError

    def _state_derivative(self, state: Tensor) -> Tensor:
        """
        Optional continuous-time dynamics f(x) used by numerical integrators
        in some scenarios.

        Expected signature:
        - state: (B, state_dim)
        - returns: (B, state_dim) time derivative at the current state

        Subclasses that implement ODE-based dynamics (e.g. RK4 integration)
        should override this method. The default implementation is provided
        only for type/contract clarity and will raise if called.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_batch(
        self,
        T: int,  # time steps
        B: int,  # batch size
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return (X_true, Y, x0) on self.device/self.dtype:
        - X_true: (B, T, state_dim) noiseless state trajectory
        - Y:      (B, T, measure_dim) noisy observation trajectory
        - x0:     (B, state_dim) initial state
        """
        raise NotImplementedError
