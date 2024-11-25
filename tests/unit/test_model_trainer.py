import pytest
from src.components.model_trainer import ModelTrainer
from src.utils.common import read_yaml
from pathlib import Path
import pandas as pd

@pytest.fixture
def config():
    return read_yaml(Path("tests/config/test_config.yaml"))

def test_model_trainer_initialization(config):
    model_trainer = ModelTrainer(config)
    assert model_trainer is not None
    assert model_trainer.device is not None

def test_model_preparation(config):
    model_trainer = ModelTrainer(config)
    train_df = pd.read_csv("tests/artifacts/data_ingestion/train.csv")
    dataset = model_trainer.prepare_dataset(train_df)
    
    assert dataset is not None
    assert len(dataset) > 0 