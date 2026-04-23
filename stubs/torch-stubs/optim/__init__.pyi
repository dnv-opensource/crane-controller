# Type stubs for torch.optim
# Covers only the symbols used by this project.

from collections.abc import Callable, Iterable
from typing import Any

from .. import Tensor

class Optimizer:
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None: ...

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None: ...
