from __future__ import annotations
import abc
from typing import Tuple
import torch

Tensor = torch.Tensor


class Scenario(abc.ABC):
    """
    Abstract base class for a dynamical system scenario.
    Assumes state and measurement dimensions are the same (M == state_dim).
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
        Each subclass should construct internal dynamics plus matrices Q/R/P0/H.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    def world_dim(self) -> int:
        # For L96, world_dim == state_dim; other scenarios may differ.
        dim = self.state_dim
        return dim

    @abc.abstractmethod
    def sample_batch(
        self,
        T: int,  # time steps
        B: int,  # batch size
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return (X_true, Y, x0) on self.device/self.dtype.
        - X_true: (B, T, state_dim) state trajectory
        - Y:      (B, T, state_dim) observation trajectory (M == state_dim here)
        - x0:     (B, state_dim) initial state
        """
        raise NotImplementedError
