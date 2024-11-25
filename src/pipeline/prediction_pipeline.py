from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

class PredictionPipeline:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Emotion label mapping
        self.emotion_mapping = {
            0: 'joy', 1: 'sadness', 2: 'anger', 
            3: 'fear', 4: 'love', 5: 'surprise',
            6: 'hate', 7: 'neutral', 8: 'worry',
            9: 'relief', 10: 'happiness', 11: 'fun',
            12: 'empty', 13: 'enthusiasm', 14: 'boredom'
        }
        
    def predict(self, text: str):
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                
            # Get emotion label and confidence
            emotion = self.emotion_mapping[predicted_class]
            confidence = predictions[0][predicted_class].item()
                
            return {
                "emotion": emotion,
                "confidence": confidence,
                "text": text
            }
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise e 