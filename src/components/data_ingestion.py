import os
from src.utils.common import *
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataIngestion:
    def __init__(self, config: ConfigBox):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_trainer.model_name)

    def tokenize_data(self, df):
        """Tokenize the text data"""
        try:
            logging.info("Tokenizing text data...")
            # Tokenize the texts
            tokenized = self.tokenizer(
                df['text'].tolist(),
                padding=True,
                truncation=True,
                max_length=self.config.model_trainer.max_length,
                return_tensors="pt"
            )
            
            # Convert emotions to numerical labels
            emotion_mapping = {
                'joy': 0, 'sadness': 1, 'anger': 2, 
                'fear': 3, 'love': 4, 'surprise': 5,
                'hate': 6, 'neutral': 7, 'worry': 8,
                'relief': 9, 'happiness': 10, 'fun': 11,
                'empty': 12, 'enthusiasm': 13, 'boredom': 14
            }
            
            df['labels'] = df['Emotion'].str.lower().map(emotion_mapping)
            
            # Create processed dataframe
            processed_df = pd.DataFrame({
                'input_ids': tokenized['input_ids'].tolist(),
                'attention_mask': tokenized['attention_mask'].tolist(),
                'labels': df['labels'].tolist()
            })
            
            return processed_df
            
        except Exception as e:
            logging.error(f"Error in tokenization: {str(e)}")
            raise e

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Read the data using the path from config
            data_path = self.config.data_ingestion.local_data_file
            logging.info(f"Reading data from: {data_path}")
            
            # Read the data with nrows=2000
            df = pd.read_csv(data_path, nrows=2000)
            logging.info(f"Initial dataset shape: {df.shape}")
            
            # Tokenize and process the data
            processed_df = self.tokenize_data(df)
            logging.info(f"Processed dataset shape: {processed_df.shape}")
            
            # Create train-test split
            train_set, test_set = train_test_split(
                processed_df, test_size=0.2, random_state=42
            )
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

            # Create directories and save splits
            os.makedirs(self.config.data_ingestion.root_dir, exist_ok=True)
            
            train_data_path = os.path.join(self.config.data_ingestion.root_dir, "train.csv")
            test_data_path = os.path.join(self.config.data_ingestion.root_dir, "test.csv")
            
            train_set.to_csv(train_data_path, index=False)
            test_set.to_csv(test_data_path, index=False)

            logging.info(f"Ingestion of data completed. Files saved at {self.config.data_ingestion.root_dir}")
            return train_data_path, test_data_path

        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise e