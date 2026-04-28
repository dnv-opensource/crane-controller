# Type stubs for matplotlib
# Covers only the symbols used by this project.

from . import _axes as _axes
from . import axes as axes
from . import figure as figure
from . import lines as lines
from . import pyplot as pyplot
from . import text as text

def use(backend: str, *, force: bool = True) -> None: ...
