import os
import time
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import numpy as np
from tqdm import tqdm
import warnings
from config import Config, get_config
from data_loader import create_dataloaders, load_semeval_data
from features import extract_features_batch
from model import create_model, CodeDetectionModel
from evaluate import evaluate_model, calculate_metrics


class Trainer:
    
    def __init__(
        self,
        model: CodeDetectionModel,
        config: Config,
        train_loader,
        val_loader,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.training.num_epochs
        if config.training.max_steps:
            total_steps = min(total_steps, config.training.max_steps)
        
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.scaler = GradScaler() if config.training.fp16 else None
        
        self.global_step = 0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        os.makedirs(config.training.output_dir, exist_ok=True)
        os.makedirs(config.training.logging_dir, exist_ok=True)
    
    def _create_optimizer(self) -> AdamW:
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            if self.config.training.max_steps and self.global_step >= self.config.training.max_steps:
                break
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.config.training.fp16:
                with autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        structural_features=batch.get("structural_features"),
                        labels=batch["labels"]
                    )
                    loss = outputs["loss"] / self.config.training.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    structural_features=batch.get("structural_features"),
                    labels=batch["labels"]
                )
                loss = outputs["loss"] / self.config.training.gradient_accumulation_steps
            
            if self.config.training.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.fp16:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if self.global_step % self.config.training.logging_steps == 0:
                self._log_training_step(total_loss / num_batches)
            
            if self.global_step % self.config.training.eval_steps == 0:
                val_metrics = self.evaluate()
                self._log_validation(val_metrics)
                
                if self._check_early_stopping(val_metrics):
                    break
                
                self.model.train()
        
        return {"loss": total_loss / max(num_batches, 1)}
    
    def evaluate(self) -> Dict[str, float]:
        return evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            self.config.training.fp16
        )
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        current_f1 = val_metrics.get("macro_f1", 0)
        
        if current_f1 > self.best_val_f1 + self.config.training.early_stopping_threshold:
            self.best_val_f1 = current_f1
            self.patience_counter = 0
            self._save_checkpoint("best")
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.training.early_stopping_patience:
                return True
        
        return False
    
    def _log_training_step(self, loss: float):
        log_entry = {
            "step": self.global_step,
            "loss": loss,
            "lr": self.scheduler.get_last_lr()[0],
            "timestamp": datetime.now().isoformat()
        }
        self.training_history.append(log_entry)
    
    def _log_validation(self, metrics: Dict[str, float]):
        print(f"      Accuracy: {metrics['accuracy']:.4f}")
        print(f"      Macro F1: {metrics['macro_f1']:.4f}")
        print(f"      Best F1:  {self.best_val_f1:.4f}")
    
    def _save_checkpoint(self, name: str):
        checkpoint_path = os.path.join(
            self.config.training.output_dir,
            f"checkpoint_{name}.pt"
        )
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_f1": self.best_val_f1,
            "config": self.config
        }, checkpoint_path)
        
    
    def train(self) -> Dict[str, float]:
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.config.training.num_epochs}")
        print(f"   Batch size: {self.config.training.effective_batch_size}")
        print(f"   Mixed precision: {self.config.training.fp16}")
        
        
        for epoch in range(self.config.training.num_epochs):
            
            train_metrics = self.train_epoch(epoch)
            
            if self.config.training.max_steps and self.global_step >= self.config.training.max_steps:
                break
        final_metrics = self.evaluate()
        
        self._save_checkpoint("final")

        history_path = os.path.join(self.config.training.logging_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"   Final Macro F1: {final_metrics['macro_f1']:.4f}")
        print(f"   Best Macro F1:  {self.best_val_f1:.4f}")
        
        return final_metrics


def train_model(config: Optional[Config] = None) -> Tuple[CodeDetectionModel, Dict]:
    if config is None:
        config = get_config()
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    train_samples, train_meta = load_semeval_data(config.data, "train")
    val_samples, val_meta = load_semeval_data(config.data, "validation")
    
    train_features = None
    val_features = None
    
    if config.model.use_structural_features:
        train_codes = [s.code for s in train_samples]
        train_langs = [s.language for s in train_samples]
        train_features = extract_features_batch(train_codes, train_langs)
        
        val_codes = [s.code for s in val_samples]
        val_langs = [s.language for s in val_samples]
        val_features = extract_features_batch(val_codes, val_langs)
    
    from data_loader import CodeDetectionDataset
    from torch.utils.data import DataLoader
    
    train_dataset = CodeDetectionDataset(
        train_samples, tokenizer, config.data.max_length, train_features
    )
    val_dataset = CodeDetectionDataset(
        val_samples, tokenizer, config.data.max_length, val_features
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=True
    )
    
    model = create_model(config)
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device
    )
    
    final_metrics = trainer.train()
    
    return model, final_metrics


if __name__ == "__main__":
    
    model, metrics = train_model(config)
