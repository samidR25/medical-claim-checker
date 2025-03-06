# src/data_processing/prepare_training_data.py

import json
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm

def prepare_training_data():
    """Prepare training data with temporal conflicts"""
    
    # Load temporally-enriched dataset
    with open('../data/processed/temporal_covidfact.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Load temporal knowledge base
    with open('../data/processed/temporal_kb.json', 'r') as f:
        kb = json.load(f)
    
    # Create mapping of facts to their temporal validity
    fact_timelines = {}
    for fact in kb:
        fact_timelines[fact["fact"]] = fact["statements"]
    
    # Keywords for matching claims to facts
    fact_keywords = {
        "masks_prevent_spread": ["mask", "face covering", "prevent", "spread"],
        "hydroxychloroquine_treatment": ["hydroxychloroquine", "hcq", "treatment", "cure"],
        # Add more mappings
    }
    
    # Process each claim
    processed_data = []
    for item in tqdm(data):
        # Skip items without temporal info
        if "temporal_info" not in item or not item["temporal_info"]["publication_date"]:
            processed_data.append(item)
            continue
        
        # Get publication date
        pub_date = datetime.fromisoformat(item["temporal_info"]["publication_date"])
        
        # Match claim to KB facts
        claim_lower = item["claim"].lower()
        matched_facts = []
        
        for fact_id, keywords in fact_keywords.items():
            if any(keyword in claim_lower for keyword in keywords):
                matched_facts.append(fact_id)
        
        # If claim matches a fact, check if it's a temporal conflict
        is_temporal_conflict = False
        conflict_reason = ""
        
        for fact_id in matched_facts:
            for statement in fact_timelines.get(fact_id, []):
                start = datetime.fromisoformat(statement["start_date"])
                end = datetime.fromisoformat(statement["end_date"])
                
                if start <= pub_date <= end:
                    # If claim is SUPPORTED but consensus is false, or vice versa
                    if (item["label"] == "SUPPORTED" and statement["consensus"] == "false") or \
                       (item["label"] == "REFUTED" and statement["consensus"] == "true"):
                        is_temporal_conflict = True
                        conflict_reason = statement["description"]
                        break
        
        # Add temporal conflict label if needed
        if is_temporal_conflict:
            item["temporal_label"] = "TEMPORAL_CONFLICT"
            item["conflict_reason"] = conflict_reason
        else:
            item["temporal_label"] = item["label"]
        
        processed_data.append(item)
    
    # Generate synthetic temporal conflicts if needed
    if sum(1 for item in processed_data if item.get("temporal_label") == "TEMPORAL_CONFLICT") < 100:
        # Create synthetic examples by manipulating dates
        for item in random.sample([i for i in processed_data if i["label"] == "SUPPORTED"], 100):
            # Copy the item and modify it
            new_item = item.copy()
            new_item["temporal_label"] = "TEMPORAL_CONFLICT"
            new_item["conflict_reason"] = "Synthetic temporal conflict for training"
            
            # Modify the date to a different pandemic phase
            current_phase = item["temporal_info"]["pandemic_phase"]
            new_phase = random.choice([p for p in ["early_outbreak", "first_wave", "vaccination_phase", "variant_phase"] if p != current_phase])
            
            new_item["temporal_info"]["pandemic_phase"] = new_phase
            
            # Add to dataset
            processed_data.append(new_item)
    
    # Split into train, validation, test
    random.shuffle(processed_data)
    train_size = int(0.8 * len(processed_data))
    val_size = int(0.1 * len(processed_data))
    
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:train_size+val_size]
    test_data = processed_data[train_size+val_size:]
    
    # Save splits
    with open('../data/processed/temporal_train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open('../data/processed/temporal_val.jsonl', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    with open('../data/processed/temporal_test.jsonl', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    prepare_training_data()