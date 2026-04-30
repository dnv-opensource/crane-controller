# Type stubs for matplotlib.pyplot
# Covers only the symbols used by this project.

from collections.abc import Sequence
from typing import Any, Literal, overload

from numpy.typing import NDArray

from .axes import Axes
from .figure import Figure
from .lines import Line2D
from .text import Text

rcParams: dict[str, Any]

@overload
def subplots(
    nrows: Literal[1] = ...,
    ncols: Literal[1] = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[True] = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw: Any,
) -> tuple[Figure, Axes]: ...
@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: Literal[False],
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw: Any,
) -> tuple[Figure, NDArray[Any]]: ...
@overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | Literal["none", "all", "row", "col"] = ...,
    sharey: bool | Literal["none", "all", "row", "col"] = ...,
    squeeze: bool = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, Any] | None = ...,
    gridspec_kw: dict[str, Any] | None = ...,
    **fig_kw: Any,
) -> tuple[Figure, Any]: ...
def show(*, block: bool | None = None) -> None: ...
def ion() -> None: ...
def title(
    label: str,
    fontdict: dict[str, Any] | None = None,
    loc: Literal["left", "center", "right"] | None = None,
    pad: float | None = None,
    *,
    y: float | None = None,
    **kwargs: Any,
) -> Text: ...
def suptitle(t: str, **kwargs: Any) -> Text: ...
def plot(
    *args: Any,
    scalex: bool = True,
    scaley: bool = True,
    data: Any = None,
    **kwargs: Any,
) -> list[Line2D]: ...
def xlim(*args: Any, **kwargs: Any) -> tuple[float, float]: ...
def ylim(*args: Any, **kwargs: Any) -> tuple[float, float]: ...
def tight_layout(**kwargs: Any) -> None: ...
def pause(interval: float) -> None: ...
def legend(*args: Any, **kwargs: Any) -> Any: ...
def close(fig: Any = ...) -> None: ...
