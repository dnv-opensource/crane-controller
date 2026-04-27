# Type stubs for matplotlib.figure
# Covers only the symbols used by this project.

from typing import Any

class Figure:
    def tight_layout(self, **kwargs: Any) -> None: ...
    def savefig(self, fname: str, **kwargs: Any) -> None: ...
