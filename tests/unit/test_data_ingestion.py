import pytest
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.utils.common import read_yaml
from pathlib import Path

@pytest.fixture
def config():
    return read_yaml(Path("tests/config/test_config.yaml"))

def test_data_ingestion_initialization(config):
    data_ingestion = DataIngestion(config)
    assert data_ingestion is not None
    assert data_ingestion.tokenizer is not None

def test_data_ingestion_process(config):
    data_ingestion = DataIngestion(config)
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    assert Path(train_path).exists()
    assert Path(test_path).exists()
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    assert 'input_ids' in train_df.columns
    assert 'attention_mask' in train_df.columns
    assert 'labels' in train_df.columns 