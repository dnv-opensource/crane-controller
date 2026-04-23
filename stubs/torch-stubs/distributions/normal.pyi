# Type stubs for torch.distributions.normal
# Covers only the symbols used by this project.

from .. import Tensor

class Normal:
    loc: Tensor | float
    scale: Tensor | float
    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None: ...
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor: ...
    def log_prob(self, value: Tensor) -> Tensor: ...
