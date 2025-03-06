# src/temporal_verification/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
from sklearn.metrics import classification_report

from src.models.temporal_verifier import TemporalVerificationModel

class TemporalClaimDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        
        # Map pandemic phases to IDs
        self.phase_to_id = {
            "unknown": 0,
            "early_outbreak": 1,
            "first_wave": 2,
            "vaccination_phase": 3,
            "variant_phase": 4
        }
        
        # Map labels to IDs
        self.label_to_id = {
            "SUPPORTED": 0,
            "REFUTED": 1,
            "TEMPORAL_CONFLICT": 2
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        claim = item["claim"]
        evidence = " ".join(item["evidence"])
        input_text = f"{claim} [SEP] {evidence}"
        
        # Get label - for training data this would include temporal conflicts
        label = item.get("temporal_label", item["label"])
        label_id = self.label_to_id.get(label, 0)
        
        # Get temporal phase
        phase = "unknown"
        if "temporal_info" in item and item["temporal_info"]["pandemic_phase"]:
            phase = item["temporal_info"]["pandemic_phase"]
        phase_id = self.phase_to_id.get(phase, 0)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding.input_ids.flatten(),
            "attention_mask": encoding.attention_mask.flatten(),
            "temporal_phase": torch.tensor(phase_id),
            "label": torch.tensor(label_id)
        }

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = TemporalVerificationModel()
    model.to(device)
    
    # Load data
    train_dataset = TemporalClaimDataset(
        "../data/processed/temporal_train.jsonl",
        tokenizer
    )
    val_dataset = TemporalClaimDataset(
        "../data/processed/temporal_val.jsonl",
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Set up optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            temporal_phase = batch["temporal_phase"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, temporal_phase)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                temporal_phase = batch["temporal_phase"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids, attention_mask, temporal_phase)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(classification_report(all_labels, all_preds, target_names=["SUPPORTED", "REFUTED", "TEMPORAL_CONFLICT"]))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "../models/temporal_verifier.pt")
            print("Saved best model!")

if __name__ == "__main__":
    train()