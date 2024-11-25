from src.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def batch_predict(texts: list, model_path: str = "artifacts/model_trainer/best_model"):
    try:
        # Initialize prediction pipeline
        predictor = PredictionPipeline(model_path)
        
        # Make predictions
        results = []
        for text in texts:
            result = predictor.predict(text)
            results.append(result)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        return df_results
        
    except Exception as e:
        logging.error(f"Error during batch inference: {str(e)}")
        raise e

if __name__ == "__main__":
    # Example batch prediction
    texts = [
        "I am so happy today!",
        "This is really frustrating",
        "I'm feeling quite anxious about tomorrow",
        "What a wonderful surprise!",
        "I love spending time with my family"
    ]
    
    results_df = batch_predict(texts)
    print("\n=== Batch Prediction Results ===")
    print(results_df)
    
    # Save results
    results_df.to_csv("prediction_results.csv", index=False) 