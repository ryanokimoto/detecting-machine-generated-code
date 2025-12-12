import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple
import math


class CodeDetectionModel(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 2,
        num_structural_features: int = 15,
        dropout: float = 0.1,
        use_structural_features: bool = True,
        freeze_embeddings: bool = True,
        freeze_n_layers: int = 6
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.use_structural_features = use_structural_features
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        if freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        
        if freeze_n_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layer[:freeze_n_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
        if use_structural_features:
            self.feature_projection = nn.Sequential(
                nn.Linear(num_structural_features, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 128),
            )
            classifier_input_size = hidden_size + 128
        else:
            self.feature_projection = None
            classifier_input_size = hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        if self.feature_projection is not None:
            for layer in self.feature_projection:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        if self.use_structural_features and structural_features is not None:
            feature_embedding = self.feature_projection(structural_features)
            combined = torch.cat([pooled_output, feature_embedding], dim=-1)
        else:
            combined = pooled_output
        
        logits = self.classifier(combined)
        probabilities = F.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probabilities": probabilities
        }
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LightweightCodeDetector(nn.Module):
    def __init__(
        self,
        num_features: int = 15,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        num_labels: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        in_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_labels))
        
        self.model = nn.Sequential(*layers)
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        logits = self.model(features)
        probabilities = F.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probabilities": probabilities
        }
        
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        
        return result


class EnsembleDetector(nn.Module):
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structural_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        all_probs = []
        all_logits = []
        
        for model in self.models:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                structural_features=structural_features
            )
            all_probs.append(outputs["probabilities"])
            all_logits.append(outputs["logits"])
        
        # Weighted average of probabilities
        weights = self.weights.to(all_probs[0].device)
        stacked_probs = torch.stack(all_probs, dim=0)
        avg_probs = (stacked_probs * weights.view(-1, 1, 1)).sum(dim=0)
        
        stacked_logits = torch.stack(all_logits, dim=0)
        avg_logits = (stacked_logits * weights.view(-1, 1, 1)).sum(dim=0)
        
        result = {
            "logits": avg_logits,
            "probabilities": avg_probs
        }
        
        if labels is not None:
            result["loss"] = F.cross_entropy(avg_logits, labels)
        
        return result


def create_model(config) -> CodeDetectionModel:
    model = CodeDetectionModel(
        model_name=config.model.model_name,
        num_labels=config.model.num_labels,
        num_structural_features=config.model.num_structural_features,
        dropout=config.model.dropout,
        use_structural_features=config.model.use_structural_features,
        freeze_embeddings=config.model.freeze_embeddings,
        freeze_n_layers=config.model.freeze_n_layers
    )
    return model


if __name__ == "__main__":
    from config import get_debug_config
    
    config = get_debug_config()
    model = create_model(config)
    
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    structural_features = torch.randn(batch_size, 15)
    labels = torch.randint(0, 2, (batch_size,))
    
    outputs = model(input_ids, attention_mask, structural_features, labels)
    
    print(f"\nTest forward pass:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Probabilities shape: {outputs['probabilities'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
