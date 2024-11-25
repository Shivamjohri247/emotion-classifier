import os
from pathlib import Path
import logging
import shutil
import time
import sqlite3

logging.basicConfig(level=logging.INFO)

def clean_mlflow_db():
    """Clean up MLflow database"""
    try:
        db_path = Path("mlflow_db/mlflow.db")
        if db_path.exists():
            # Close any existing connections
            conn = sqlite3.connect(db_path)
            conn.close()
            # Remove the file
            os.remove(db_path)
            logging.info("Cleaned up MLflow database")
    except Exception as e:
        logging.error(f"Error cleaning MLflow database: {str(e)}")

def setup_mlflow_server():
    try:
        # Clean up existing directories and database
        for dir_path in ['mlflow_db', 'mlruns']:
            dir_path = Path(dir_path)
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(exist_ok=True)
        
        # Clean up database
        clean_mlflow_db()
        
        # Start MLflow server with specific backend store and artifact location
        cmd = (
            "mlflow server "
            f"--backend-store-uri sqlite:///mlflow_db/mlflow.db "
            f"--default-artifact-root {Path('mlruns').absolute()} "
            "--host 127.0.0.1 "
            "--port 5000"
        )
        
        logging.info("Starting MLflow server...")
        os.system(cmd)
        
    except Exception as e:
        logging.error(f"Error starting MLflow server: {str(e)}")
        raise e

if __name__ == "__main__":
    setup_mlflow_server() 