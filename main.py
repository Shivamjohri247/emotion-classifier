from src.pipeline.training_pipeline import TrainingPipeline
import mlflow
import mlflow.transformers
import logging
from pathlib import Path
import os
import shutil
from mlflow.exceptions import MlflowException
import time

logging.basicConfig(level=logging.INFO)

def clean_mlflow_artifacts():
    """Clean up MLflow artifacts and database"""
    try:
        # Remove mlruns directory
        if os.path.exists("mlruns"):
            shutil.rmtree("mlruns")
        
        # Remove mlflow.db
        if os.path.exists("mlflow_db/mlflow.db"):
            os.remove("mlflow_db/mlflow.db")
            
        logging.info("Cleaned up MLflow artifacts")
    except Exception as e:
        logging.error(f"Error cleaning MLflow artifacts: {str(e)}")

def setup_mlflow():
    """Setup MLflow environment"""
    try:
        # Clean up existing MLflow artifacts
        clean_mlflow_artifacts()
        
        # Create fresh directories
        os.makedirs("mlruns", exist_ok=True)
        os.makedirs("mlflow_db", exist_ok=True)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"sqlite:///mlflow_db/mlflow.db")
        
        # Create new experiment
        experiment_name = "emotion_classification"
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        logging.info("MLflow setup completed successfully")
        
    except Exception as e:
        logging.error(f"Error in MLflow setup: {str(e)}")
        raise e

def main():
    try:
        # Setup MLflow environment
        setup_mlflow()
        
        with mlflow.start_run(run_name="emotion_classification_training"):
            # Start training pipeline
            training_pipeline = TrainingPipeline()
            model_path = training_pipeline.start_training()
            
            # Log parameters
            config = training_pipeline.config
            mlflow.log_params({
                "model_name": config.model_trainer.model_name,
                "num_epochs": config.model_trainer.num_train_epochs,
                "batch_size": config.model_trainer.batch_size,
                "learning_rate": config.model_trainer.learning_rate
            })
            
            # Log model
            mlflow.log_artifact(model_path)
            
            logging.info(f"Training completed. Model saved at: {model_path}")
            
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 