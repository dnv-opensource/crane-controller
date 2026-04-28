"""Test configuration and fixtures."""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from shutil import rmtree

import matplotlib as mpl
import pytest
from py_crane.crane import Crane

from crane_controller.crane_factory import build_crane

mpl.use("Agg")  # headless backend — must be set before any pyplot import


@pytest.fixture(scope="package", autouse=True)
def chdir():
    """
    Fixture that changes the current working directory to the 'test_working_directory' folder.
    This fixture is automatically used for the entire package.
    """
    original_cwd = Path.cwd()
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    try:
        yield
    finally:
        os.chdir(original_cwd)  # reset to original working directory after tests


@pytest.fixture(scope="package", autouse=True)
def test_dir() -> Path:
    """
    Fixture that returns the absolute path of the directory containing the current file.
    This fixture is automatically used for the entire package.
    """
    return Path(__file__).parent.absolute()

@pytest.fixture(scope="package", autouse=True)
def model_dir() -> Path:
    """
    Fixture that returns the absolute path of the directory containing the trained model files.
    This fixture is automatically used for the entire package.
    """
    return Path(__file__).parent.absolute().parent / "models"


output_dirs: list[str] = [
    "results",
    "data",
]
output_files: list[str] = [
    "*test*.pdf",
]


@pytest.fixture(autouse=True)
def default_setup_and_teardown():
    """
    Fixture that performs setup and teardown actions before and after each test function.
    It removes the output directories and files specified in 'output_dirs' and 'output_files' lists.
    """
    _remove_output_dirs_and_files()
    yield
    _remove_output_dirs_and_files()


def _remove_output_dirs_and_files() -> None:
    """
    Helper function that removes the output directories and files specified in 'output_dirs' and 'output_files' lists.
    """
    for folder in output_dirs:
        rmtree(folder, ignore_errors=True)
    for pattern in output_files:
        for file in Path.cwd().glob(pattern):
            _file = Path(file)
            if _file.is_file():
                _file.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def setup_logging(caplog: pytest.LogCaptureFixture) -> None:
    """
    Fixture that sets up logging for each test function.
    It sets the log level to 'INFO' and clears the log capture.
    """
    caplog.set_level("INFO")
    caplog.clear()


@pytest.fixture(autouse=True)
def logger() -> logging.Logger:
    """Fixture that returns the logger object."""
    return logging.getLogger()


@pytest.fixture
def crane() -> Callable[..., Crane]:
    return build_crane


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help=(
            "Command line switch to show plots during tests, and dump additional results to console. By default, False."
        ),
    )


@pytest.fixture(scope="session")
def show(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--show"))


@pytest.fixture
def v0() -> float:
    return 1.0


@pytest.fixture
def reward_limit() -> float:
    return 0.0


@pytest.fixture
def trained() -> tuple[str, bool]:
    return ("anti-pendulum.json", False)
