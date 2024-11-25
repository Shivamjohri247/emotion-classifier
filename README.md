# 🎭 Emotion Classification System

Hey there! 👋 Welcome to my Emotion Classification project. This is a production-ready system that can understand emotions in text using state-of-the-art transformer models.

## 🚀 What Can This Do?

This system can:
- Classify text into 15 different emotions
- Process single or multiple texts
- Serve predictions through an API
- Track experiments with MLflow
- Run on both CPU and GPU

## 🛠️ Getting Started

### Prerequisites
Before we dive in, make sure you have:
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)
- Basic knowledge of Python and ML

### Installation

1. First, let's set up our environment:
   conda create -n emotion-env python=3.8
   conda activate emotion-env

2. Install dependencies:
   pip install -e .

USAGE
-----
1. Training:
   - Start MLflow: python mlflow_server.py
   - Run training: python main.py

2. Prediction:
   a) Single text:
      python inference.py
      
   b) Batch processing:
      python batch_inference.py
      
   c) API:
      python api.py
      Access: http://127.0.0.1:8000/docs

FILE STRUCTURE
-------------
api.py              - FastAPI implementation
inference.py        - Single prediction
batch_inference.py  - Batch prediction
main.py            - Training script
mlflow_server.py    - MLflow setup
config/config.yaml  - Configuration

API ENDPOINTS
------------
GET  /              - Health check
GET  /model-info    - Model information
POST /predict       - Single prediction
POST /predict/batch - Batch prediction

EMOTION LABELS
-------------
0: Joy
1: Sadness
2: Anger
3: Fear
4: Love
5: Surprise
6: Hate
7: Neutral
8: Worry
9: Relief
10: Happiness
11: Fun
12: Empty
13: Enthusiasm
14: Boredom

REQUIREMENTS
-----------
- Python 3.8+
- PyTorch
- Transformers
- MLflow
- FastAPI
- pandas
- scikit-learn

QUICK COMMANDS
-------------
Start MLflow:
mlflow server --backend-store-uri sqlite:///mlflow_db/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

Start API:
uvicorn api:app --host 127.0.0.1 --port 8000

Example Prediction:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text":"I am feeling happy!"}'

AUTHOR
------
Maintained by: ShivamJohri

For detailed documentation, visit: https://github.com/ShivamJohri/emotion_classifier

## 🧪 Testing

### Test Structure