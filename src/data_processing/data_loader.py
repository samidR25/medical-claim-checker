"""
Module for loading and preprocessing the COVID-Fact dataset,
separating original claims from counter-claims.
"""

import json
import hashlib
from typing import List, Dict, Any, Tuple

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_original_claims(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract only the supported (original) claims."""
    return [item for item in data if item["label"] == "SUPPORTED"]

def extract_original_counter_claims(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract only the refuted (counter) claims."""
    return [item for item in data if item["label"] == "REFUTED"]

def add_identifiers(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add unique identifiers to claims for tracking relationships."""
    for claim in claims:
        # Create a hash based on the claim text
        claim_id = hashlib.md5(claim["claim"].encode()).hexdigest()
        claim["id"] = claim_id
    return claims

def match_counter_claims(
    original_claims: List[Dict[str, Any]],
    counter_claims: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a mapping between original claims and their counter-claims.
    Uses text similarity to match them.
    """
    from sentence_transformers import SentenceTransformer, util
    import torch
    
    # Load sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Get embeddings for all claims
    original_texts = [claim["claim"] for claim in original_claims]
    counter_texts = [claim["claim"] for claim in counter_claims]
    
    original_embeddings = model.encode(original_texts, convert_to_tensor=True)
    counter_embeddings = model.encode(counter_texts, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(counter_embeddings, original_embeddings)
    
    # Match counter-claims to original claims
    matches = {}
    for counter_idx, counter_claim in enumerate(counter_claims):
        # Get similarity scores for this counter-claim against all original claims
        scores = cosine_scores[counter_idx]
        
        # Get the most similar original claim
        max_score_idx = torch.argmax(scores).item()
        max_score = scores[max_score_idx].item()
        
        # If similarity is high enough, consider it a match
        if max_score > 0.5:  # Threshold can be adjusted
            original_id = original_claims[max_score_idx]["id"]
            
            if original_id not in matches:
                matches[original_id] = []
                
            counter_copy = counter_claim.copy()
            counter_copy["similarity_score"] = max_score
            counter_copy["original_claim_id"] = original_id
            counter_copy["generation_method"] = "original"
            
            matches[original_id].append(counter_copy)
    
    return matches