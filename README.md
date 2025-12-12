# detecting-machine-generated-code

## Binary Classifiers for SemEval-2026 Task 13 Task A

binaryclassifiers contains fine-tuned binary classification models for SemEval-2026 Task 13: Detecting AI-Generated Code. The models are designed to distinguish between human-written and machine-generated code using the Task A dataset.

## Models

We utilize two pre-trained Transformer models fine-tuned for sequence classification:

- **CodeBERT** (microsoft/codebert-base)
- **UnixCoder** (microsoft/unixcoder-base)

Both models use a Roberta-based architecture and are trained to classify code snippets into two labels:

- `0`: Human-written
- `1`: Machine-generated

## Dataset

The project automatically streams data from the Hugging Face Hub:

- **Dataset**: DaniilOr/SemEval-2026-Task13
- **Input Format**: Parquet files

## Prerequisites

Install the required Python packages before running the scripts:

```bash
pip install torch transformers datasets pandas scikit-learn numpy tqdm
```

## Project Structure

- `codebert.py`: Script for training, evaluating, and generating predictions using CodeBERT.
- `unixcoder.py`: Script for training, evaluating, and generating predictions using UnixCoder.

## Usage

### 1. Training & Inference

The scripts are designed to run the full pipeline automatically. This includes:
- Loading and cleaning the data.
- Fine-tuning the model (default: 3 epochs).
- Evaluating on the validation set.
- Running streaming inference on the test set.
- Saving the model artifacts and submission CSV.

To run the CodeBERT pipeline:

```bash
python codebert.py
```

To run the UnixCoder pipeline:

```bash
python unixcoder.py
```

### 2. Output

After execution, you will find the submission files in the output directory:

- `codebert_submission.csv`
- `unixcoder_submission.csv`

The format of the output CSV is:

```csv
ID,label
12345,1
67890,0
...
```

## Configuration

Hyperparameters can be adjusted directly in the `run_full_pipeline` call within the scripts:

```python
# Example configuration in the script

t = trainer_obj.run_full_pipeline(
    output_dir="taskA-codebert-model",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
)
```
