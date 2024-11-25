from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    IntervalStrategy
)
import torch
from datasets import Dataset
import pandas as pd
import numpy as np
from src.utils.common import *
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, config: ConfigBox):
        self.config = config
        
        # CUDA setup
        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")
        
        # Log device information
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"Using GPU: {gpu_name}")
            logging.info(f"GPU Memory: {gpu_memory:.2f} GB")
            torch.cuda.set_device(0)
        else:
            logging.warning("CUDA not available. Using CPU for training.")
        
        logging.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_trainer.model_name)
        
    def prepare_dataset(self, df):
        try:
            logging.info("Preparing dataset...")
            # Convert string representations of lists back to tensors
            df['input_ids'] = df['input_ids'].apply(eval)
            df['attention_mask'] = df['attention_mask'].apply(eval)
            
            # Convert to tensors and move to device
            input_ids = [torch.tensor(x, device=self.device) for x in tqdm(df['input_ids'], desc="Processing input_ids")]
            attention_masks = [torch.tensor(x, device=self.device) for x in tqdm(df['attention_mask'], desc="Processing attention_masks")]
            
            # Ensure labels are clean integers
            df['labels'] = pd.to_numeric(df['labels'], errors='coerce')
            df = df.dropna(subset=['labels'])
            labels = torch.tensor(df['labels'].astype(int).tolist(), device=self.device)
            
            # Convert to HF dataset
            dataset = Dataset.from_dict({
                'input_ids': [x.cpu().numpy() for x in input_ids],
                'attention_mask': [x.cpu().numpy() for x in attention_masks],
                'labels': labels.cpu().numpy()
            })
            
            logging.info(f"Dataset prepared with {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            logging.error(f"Error in prepare_dataset: {str(e)}")
            raise e

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
        
    def train(self, train_data_path, test_data_path):
        try:
            # Load data
            logging.info("Loading training and test data...")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Testing data shape: {test_df.shape}")
            
            # Prepare datasets
            train_dataset = self.prepare_dataset(train_df)
            test_dataset = self.prepare_dataset(test_df)
            
            logging.info("Datasets prepared successfully")
            
            # Load model and move to GPU
            logging.info("Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_trainer.model_name,
                num_labels=int(self.config.model_trainer.num_labels)
            )
            model.to(self.device)
            
            logging.info(f"Model loaded and moved to {self.device}")
            
            # Training arguments with GPU optimization
            training_args = TrainingArguments(
                output_dir=self.config.model_trainer.root_dir,
                num_train_epochs=float(self.config.model_trainer.num_train_epochs),
                per_device_train_batch_size=int(self.config.model_trainer.batch_size),
                per_device_eval_batch_size=int(self.config.model_trainer.batch_size),
                learning_rate=float(self.config.model_trainer.learning_rate),
                weight_decay=float(self.config.model_trainer.weight_decay),
                evaluation_strategy="steps",
                eval_steps=100,
                logging_steps=50,
                save_strategy="steps",
                save_steps=100,
                load_best_model_at_end=True,
                remove_unused_columns=False,
                no_cuda=False,
                fp16=True if self.device.type == "cuda" else False,
                gradient_accumulation_steps=4 if self.device.type == "cpu" else 1,
                dataloader_num_workers=4,
                optim="adamw_torch",
                report_to="mlflow"
            )
            
            logging.info("Starting training...")
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(self.tokenizer),
                compute_metrics=self.compute_metrics
            )
            
            # Train model
            trainer.train()
            
            # Final evaluation
            eval_results = trainer.evaluate()
            logging.info(f"Final evaluation results: {eval_results}")
            
            # Save model
            model_path = os.path.join(self.config.model_trainer.root_dir, "best_model")
            trainer.save_model(model_path)
            logging.info(f"Model saved to {model_path}")
            
            return model_path
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise e