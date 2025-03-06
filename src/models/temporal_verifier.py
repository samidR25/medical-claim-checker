# src/models/temporal_verifier.py

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import json
import datetime
import numpy as np

class TemporalVerificationModel(nn.Module):
    def __init__(self, pretrained_model_name="roberta-large"):
        super(TemporalVerificationModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 3)  # 3 classes: SUPPORTED, REFUTED, TEMPORAL_CONFLICT
        self.temporal_embedding = nn.Embedding(5, 50)  # 5 pandemic phases, 50-dim embedding
        self.fusion_layer = nn.Linear(self.roberta.config.hidden_size + 50, self.roberta.config.hidden_size)
        
    def forward(self, input_ids, attention_mask, temporal_phase):
        # Get RoBERTa embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Get temporal embeddings
        temporal_emb = self.temporal_embedding(temporal_phase)
        
        # Fuse the embeddings
        combined = torch.cat([pooled_output, temporal_emb], dim=1)
        fused = self.fusion_layer(combined)
        
        # Final classification
        logits = self.classifier(fused)
        
        return logits

class TemporalVerifier:
    def __init__(self, model_path=None, kb_path="../data/processed/temporal_kb.json"):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
        # Load model
        if model_path:
            self.model = TemporalVerificationModel()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model = TemporalVerificationModel()
        
        self.model.eval()
        
        # Load knowledge base
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)
        
        # Map pandemic phases to IDs
        self.phase_to_id = {
            "unknown": 0,
            "early_outbreak": 1,
            "first_wave": 2,
            "vaccination_phase": 3,
            "variant_phase": 4
        }
    
    def verify(self, claim, evidence, publication_date=None, pandemic_phase=None):
        """Verify a claim considering its temporal context"""
        
        # Process inputs
        if isinstance(evidence, list):
            evidence = " ".join(evidence)
        
        input_text = f"{claim} [SEP] {evidence}"
        encoded_input = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Get temporal phase ID
        if pandemic_phase:
            phase_id = self.phase_to_id.get(pandemic_phase, 0)
        else:
            phase_id = 0
        
        phase_tensor = torch.tensor([phase_id])
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(encoded_input.input_ids, encoded_input.attention_mask, phase_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
        
        # Check if related to known temporal facts
        relevant_facts = []
        if publication_date:
            pub_date = datetime.datetime.fromisoformat(publication_date)
            
            # Check the knowledge base for facts relevant to this time period
            for fact in self.kb:
                for statement in fact["statements"]:
                    start = datetime.datetime.fromisoformat(statement["start_date"])
                    end = datetime.datetime.fromisoformat(statement["end_date"])
                    
                    if start <= pub_date <= end:
                        relevant_facts.append({
                            "fact_id": fact["fact"],
                            "consensus": statement["consensus"],
                            "description": statement["description"]
                        })
        
        # Map prediction to label
        labels = ["SUPPORTED", "REFUTED", "TEMPORAL_CONFLICT"]
        
        result = {
            "prediction": labels[prediction],
            "confidence": probs[0][prediction].item(),
            "temporal_context": {
                "publication_date": publication_date,
                "pandemic_phase": pandemic_phase,
                "relevant_facts": relevant_facts
            }
        }
        
        return result

# Training code would be added here