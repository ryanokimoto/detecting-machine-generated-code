import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import argparse
import warnings
warnings.filterwarnings("ignore")

# %%
# Wrapper class to handle data loading, model initialization, training, and evaluation for UnixCoder
class UnixcoderTrainer:
    def __init__(self, max_length=512, model_name="microsoft/unixcoder-base"):
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.num_labels = None

    # Loads Parquet data from Hugging Face, removes nulls, and calculates label stats
    def load_and_prepare_data(self):
        try:
            splits = {'train': 'task_a/task_a_training_set_1.parquet', 'validation': 'task_a/task_a_validation_set.parquet', 'test': 'task_a/task_a_test_set_sample.parquet'}
            df = pd.read_parquet("hf://datasets/DaniilOr/SemEval-2026-Task13/" + splits["train"])
            print(f"Dataset columns: {df.columns.tolist()}")
            print(f"Sample data:\n{df.head()}")
            print
            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'code' and 'label' columns")

            df = df.dropna(subset=['code', 'label'])

            df['label'] = df['label'].astype(int)
            self.num_labels = df['label'].nunique()

            print(f"Number of unique labels: {self.num_labels}")
            print(f"Label range: {df['label'].min()} to {df['label'].max()}")
            print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

            val_df = pd.read_parquet("hf://datasets/DaniilOr/SemEval-2026-Task13/" + splits["validation"])
            print(f"Train samples: {len(df)}, Validation samples: {len(val_df)}")

            return df, val_df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    # Initializes the UnixCoder model and tokenizer binary classification
    def initialize_model_and_tokenizer(self):
        print(f"Initializing {self.model_name} model and tokenizer...")

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        ).to('cuda')

        print(f"Model initialized with {self.num_labels} labels")

    # Applies tokenization with truncation and padding to input code
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['code'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    # Converts DataFrames to HF Datasets and maps the tokenizer over them
    def prepare_datasets(self, train_df, val_df):
        print("Preparing datasets for training...")

        train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code']
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code']
        )

        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')

        return train_dataset, val_dataset

    # Computes standard classification metrics: Accuracy, F1, Precision, Recall
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Configures hyperparameters and executes the training loop with early stopping
    def train(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        print("Starting training...")
        print(self.model)
        print(self.model.device)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            save_total_limit=2,
            report_to=[],
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        print(f"Start training")
        trainer.train()

        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"Training completed. Model saved to {output_dir}")

        return trainer

    # Generates predictions on the validation set and prints a classification report
    def evaluate_model(self, trainer, val_dataset):
        print("Evaluating model...")

        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        print("Classification Report:")
        print(classification_report(y_true, y_pred))

        return predictions

    # Orchestrates the full workflow: load -> init -> prep -> train -> evaluate
    def run_full_pipeline(self, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        try:
            train_df, val_df = self.load_and_prepare_data()

            self.initialize_model_and_tokenizer()

            train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)

            trainer = self.train(
                train_dataset, val_dataset,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            self.evaluate_model(trainer, val_dataset)

            print("Pipeline completed successfully!")
            return trainer

        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise


# Instantiates the trainer and runs the pipeline with specific parameters
OUTPUT_DIR = "taskA-unixcoder-model"

trainer_obj = UnixcoderTrainer(
    max_length=256,
)

t = trainer_obj.run_full_pipeline(
    output_dir=OUTPUT_DIR,
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
)
# %%
import torch
import logging
from itertools import chain
from datasets import load_dataset
from tqdm import tqdm


# Runs streaming inference on large Parquet files to save memory and writes CSV
@torch.no_grad()
def predict_with_trainer(trainer_obj, parquet_path, output_path, max_length=512, batch_size=16, device=None):
    """
    Uses trainer_obj.model and trainer_obj.tokenizer to run streaming inference
    over a parquet file with columns ['ID','code'] and writes 'ID,prediction' CSV.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extracts model and tokenizer from the existing trainer object
    model = trainer_obj.model
    tokenizer = trainer_obj.tokenizer if hasattr(trainer_obj, "tokenizer") else trainer_obj.args._setup_devices and None
    if tokenizer is None and hasattr(trainer_obj, "tokenizer"):
        tokenizer = trainer_obj.tokenizer
    if tokenizer is None:
        raise ValueError("trainer_obj must have a tokenizer (e.g., provided when creating the Trainer).")

    model.to(device)
    model.eval()

    # Loads dataset in streaming mode to handle files larger than RAM
    ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)

    # Validates schema (checking ID/code) and re-chains the first row back
    it = iter(ds)
    first = next(it)
    if not {"ID", "code"}.issubset(first.keys()):
        raise ValueError("Parquet file must contain 'ID' and 'code' columns")
    stream = chain([first], it)

    # Generator function to yield data in chunks
    def batcher(iterator, bs):
        buf = []
        for ex in iterator:
            buf.append(ex)
            if len(buf) == bs:
                yield buf
                buf = []
        if buf:
            yield buf

    # Iterates through batches, predicts, and writes results to file immediately
    with open(output_path, "w") as f:
        f.write("ID,prediction\n")

        for batch in tqdm(batcher(stream, batch_size), desc="Predicting"):
            codes = [row["code"] for row in batch]
            ids   = [row["ID"] for row in batch]

            enc = tokenizer(
                codes,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_labels = logits.argmax(dim=-1).cpu().tolist()

            for ex_id, pred in zip(ids, pred_labels):
                f.write(f"{ex_id},{pred}\n")

    print(f"Predictions saved to {output_path}")

# Defines test set paths and executes the prediction function
splits = {'train': 'task_a/task_a_training_set_1.parquet', 'validation': 'task_a/task_a_validation_set.parquet', 'test': 'task_a/task_a_test_set_sample.parquet'}
TEST_PARQUET = "hf://datasets/DaniilOr/SemEval-2026-Task13/" + splits["test"]
OUT_CSV = "unixcoder_submission.csv"
predict_with_trainer(
    trainer_obj=t,
    parquet_path=TEST_PARQUET,
    output_path=OUT_CSV,
    max_length=256,
    batch_size=32,
    device="cuda"
)

print("Wrote:", OUT_CSV)

