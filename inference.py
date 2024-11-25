from src.pipeline.prediction_pipeline import PredictionPipeline
import logging

logging.basicConfig(level=logging.INFO)

def predict_emotion(text: str, model_path: str = "artifacts/model_trainer/best_model"):
    try:
        # Initialize prediction pipeline
        predictor = PredictionPipeline(model_path)
        
        # Make prediction
        result = predictor.predict(text)
        
        # Print results
        print(f"\nText: {result['text']}")
        print(f"Predicted Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        raise e

if __name__ == "__main__":
    # Example usage
    texts = [
        "I am so happy today!",
        "This is really frustrating",
        "I'm feeling quite anxious about tomorrow",
        "What a wonderful surprise!",
        "I love spending time with my family"
    ]
    
    print("\n=== Emotion Prediction Results ===")
    for text in texts:
        predict_emotion(text)
        print("-" * 50) 