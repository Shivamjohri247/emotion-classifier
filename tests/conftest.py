import pytest
import os
import shutil
from pathlib import Path
from src.utils.common import read_yaml

@pytest.fixture(scope="session")
def config():
    return read_yaml(Path("tests/config/test_config.yaml"))

@pytest.fixture(scope="session")
def cleanup_artifacts():
    yield
    if os.path.exists("tests/artifacts"):
        shutil.rmtree("tests/artifacts")

@pytest.fixture(scope="session", autouse=True)
def setup_test_env(cleanup_artifacts):
    # Create necessary directories
    os.makedirs("tests/artifacts", exist_ok=True)
    os.makedirs("tests/data", exist_ok=True)
    
    # Create test data
    from tests.data.create_test_data import create_test_data
    create_test_data() 