from abc import ABC, abstractmethod
from typing import Tuple
import torch

Tensor = torch.Tensor

class VerificationStrategy(ABC):
    @abstractmethod
    def verify(
        self,
        prop_buffer: Tensor,    # (1, k) drafted token IDs
        q_buffer: Tensor,       # (1, k) draft probabilities
        p_buffer: Tensor,       # (1, k) target probabilities
        tgt_ids: Tensor         # (1, k) target token IDs
    ) -> Tuple[Tensor, int]:
        """
        Returns:
          good:  BoolTensor of shape (1, k), True=accepted
          m:     number of accepted tokens in prefix order
        """
        ...

# Probability‐ratio sampling
class RatioSamplingStrategy(VerificationStrategy):
    def verify(self, prop_buffer, q_buffer, p_buffer, tgt_ids: Tensor):
        # u <= p/q (u is uniformly sampled in [0, 1])
        u = torch.rand_like(p_buffer)
        ratios = p_buffer / q_buffer
        good = u <= ratios
        m = int(good.cumprod(dim=-1).sum())
        return good, m

# Exact‐match: accept only if draft's proposal is the same as target's argmax
class ExactMatchStrategy(VerificationStrategy):
    def verify(self, prop_buffer, q_buffer, p_buffer, tgt_ids: Tensor):
        good = prop_buffer.eq(tgt_ids)
        m = int(good.cumprod(dim=-1).sum())
        return good, m
