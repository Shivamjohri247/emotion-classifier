from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.utils.common import read_yaml
from pathlib import Path
import logging

class TrainingPipeline:
    def __init__(self):
        self.config = read_yaml(Path("config/config.yaml"))
        
    def start_training(self):
        try:
            # Data Ingestion
            data_ingestion = DataIngestion(self.config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            
            # Model Training
            model_trainer = ModelTrainer(self.config)
            model_path = model_trainer.train(train_data_path, test_data_path)
            logging.info("Model training completed")
            
            return model_path
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise e 