import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_name = "emotion_classifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "api.py",
    "inference.py",
    "batch_inference.py",
    "mlflow_server.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    "README.md",
    "test/__init__.py",
    "test/unit/__init__.py",
    "test/integration/__init__.py",
    "notebooks/exploration.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logger.info(f"Created directory: {filedir}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logger.info(f"Created empty file: {filepath}")

logger.info("Project structure created successfully!")