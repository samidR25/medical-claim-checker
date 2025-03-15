"""
Module for comparing original and enhanced counter-claims.
"""

import json
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def save_processed_dataset(
    original_claims: List[Dict[str, Any]],
    original_counter_claims: Dict[str, List[Dict[str, Any]]],
    enhanced_counter_claims: List[Dict[str, Any]],
    output_file: str
):
    """
    Save processed dataset with both original and enhanced counter-claims.
    
    Args:
        original_claims: List of original claims
        original_counter_claims: Dictionary mapping original claim IDs to their counter-claims
        enhanced_counter_claims: List of enhanced counter-claims
        output_file: Output file path
    """
    result = []
    
    for orig_claim in original_claims:
        claim_id = orig_claim["id"]
        
        # Get original counter-claims for this claim
        orig_counters = original_counter_claims.get(claim_id, [])
        
        # Get enhanced counter-claims for this claim
        enh_counters = [c for c in enhanced_counter_claims if c["original_claim_id"] == claim_id]
        
        # Add to result
        result.append({
            "original_claim": orig_claim,
            "original_counter_claims": orig_counters,
            "enhanced_counter_claims": enh_counters
        })
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        
    return result

def create_comparison_stats(processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create statistics comparing original and enhanced counter-claims.
    
    Args:
        processed_data: Processed data with original and enhanced counter-claims
        
    Returns:
        Dictionary with comparison statistics
    """
    stats = {
        "total_original_claims": len(processed_data),
        "total_original_counter_claims": sum(len(item["original_counter_claims"]) for item in processed_data),
        "total_enhanced_counter_claims": sum(len(item["enhanced_counter_claims"]) for item in processed_data),
        "avg_original_counter_claims_per_claim": sum(len(item["original_counter_claims"]) for item in processed_data) / len(processed_data) if processed_data else 0,
        "avg_enhanced_counter_claims_per_claim": sum(len(item["enhanced_counter_claims"]) for item in processed_data) / len(processed_data) if processed_data else 0,
        "strategy_distribution": {}
    }
    
    # Count strategy distribution for enhanced counter-claims
    strategy_counts = {}
    for item in processed_data:
        for counter in item["enhanced_counter_claims"]:
            strategy = counter.get("strategy", "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    stats["strategy_distribution"] = strategy_counts
    
    return stats