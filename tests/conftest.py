"""Test configuration and fixtures."""

import logging
import os
from pathlib import Path
from shutil import rmtree

import pytest
import torch
import torch.cuda
from crane_controller.crane_factory import build_crane

CUDA_AVAILABLE: bool = torch.cuda.is_available()

TORCH_DEVICES: list[str] = ["cuda", "cpu"] if CUDA_AVAILABLE else ["cpu"]


@pytest.fixture(scope="class", params=TORCH_DEVICES)
def vary_torch_default_device(request: pytest.FixtureRequest):
    torch.set_default_device(request.param)
    yield
    torch.set_default_device("cpu")  # reset to default device after test


@pytest.fixture(scope="session", autouse=True)
def chdir():
    """
    Fixture that changes the current working directory to the 'test_working_directory' folder.
    This fixture is automatically used for the entire session.
    """
    original_cwd = Path.cwd()
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    try:
        yield
    finally:
        os.chdir(original_cwd)  # reset to original working directory after tests


@pytest.fixture(scope="session", autouse=True)
def test_dir() -> Path:
    """
    Fixture that returns the absolute path of the directory containing the current file.
    This fixture is automatically used for the entire session.
    """
    return Path(__file__).parent.absolute()


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
def crane():
    return build_crane


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="Command line switch to show plots during tests, and dump additional results to console. By default, False.",
    )


@pytest.fixture(scope="session")
def show(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--show")
