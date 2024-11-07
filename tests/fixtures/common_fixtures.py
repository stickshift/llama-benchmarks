from pathlib import Path

from dotenv import load_dotenv
import pytest

__all__ = [
    "datasets_path",
    "project_env",
    "project_path",
]


@pytest.fixture
def project_path(pytestconfig) -> Path:
    return pytestconfig.rootpath


@pytest.fixture(autouse=True)
def project_env(project_path):
    """Load project environment variables."""
    load_dotenv(project_path / "project.env")


@pytest.fixture
def datasets_path(project_path: Path) -> Path:
    return project_path / ".build" / "datasets"
