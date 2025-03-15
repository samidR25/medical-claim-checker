"""
Script to evaluate the performance of the enhanced counter-claim generation system.
This compares the original approach with the new enhanced approach.
"""

import sys
import os
import json
import pandas as pd
from typing import List, Dict, Any
import argparse

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.counter_claim_generation.generator import EnhancedCounterClaimGenerator
from src.data_processing.data_loader import (
    load_dataset, extract_original_claims, extract_original_counter_claims,
    add_identifiers, match_counter_claims
)
from src.evaluation.data_comparison import save_processed_dataset, create_comparison_stats

def generate_enhanced_counter_claims(
    original_claims: List[Dict[str, Any]],
    generator: EnhancedCounterClaimGenerator,
    num_candidates: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate enhanced counter-claims for original claims.
    
    Args:
        original_claims: List of original claims
        generator: Counter-claim generator
        num_candidates: Number of counter-claims to generate per claim
        
    Returns:
        List of enhanced counter-claims
    """
    enhanced_claims = []
    
    for claim_data in original_claims:
        counter_claims = generator.generate_counter_claims(
            claim_data["claim"], num_candidates=num_candidates)
        
        for counter_claim in counter_claims:
            # Create a new entry for this counter-claim
            enhanced_entry = {
                "claim": counter_claim["counter_claim"],
                "label": "REFUTED",
                "evidence": claim_data.get("evidence", []),
                "gold_source": claim_data.get("gold_source", ""),
                "flair": claim_data.get("flair", ""),
                "original_claim_id": claim_data["id"],
                "original_claim": claim_data["claim"],
                "generation_method": "enhanced",
                "strategy": counter_claim["strategy"],
                "contradiction_score": counter_claim.get("contradiction_score", 0)
            }
            
            enhanced_claims.append(enhanced_entry)
    
    return enhanced_claims

def main():
    parser = argparse.ArgumentParser(description="Evaluate counter-claim generation")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with claims")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--num-claims", type=int, default=None, help="Number of claims to process")
    parser.add_argument("--num-candidates", type=int, default=3, help="Number of counter-claims to generate per claim")
    args = parser.parse_args()
    
    # Initialize the generator
    generator = EnhancedCounterClaimGenerator(
        use_srl=True,
        use_entity_sub=True,
        use_quantity_mod=True,
        use_negation=True
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.input)
    
    # Extract original claims and counter-claims
    print("Separating original claims and counter-claims...")
    original_claims = extract_original_claims(dataset)
    original_counter_claims = extract_original_counter_claims(dataset)
    
    # Limit number of claims if specified
    if args.num_claims:
        original_claims = original_claims[:args.num_claims]
    
    # Add identifiers to original claims
    print("Adding identifiers...")
    original_claims = add_identifiers(original_claims)
    
    # Match original counter-claims to original claims
    print("Matching original counter-claims...")
    original_counter_map = match_counter_claims(original_claims, original_counter_claims)
    
    # Generate enhanced counter-claims
    print("Generating enhanced counter-claims...")
    enhanced_counter_claims = generate_enhanced_counter_claims(
        original_claims, generator, args.num_candidates)
    
    # Save processed dataset
    print("Saving processed dataset...")
    processed_data = save_processed_dataset(
        original_claims,
        original_counter_map,
        enhanced_counter_claims,
        args.output
    )
    
    # Create comparison statistics
    print("Creating comparison statistics...")
    stats = create_comparison_stats(processed_data)
    
    # Print statistics
    print("\nComparison Statistics:")
    print(f"Total original claims: {stats['total_original_claims']}")
    print(f"Total original counter-claims: {stats['total_original_counter_claims']}")
    print(f"Total enhanced counter-claims: {stats['total_enhanced_counter_claims']}")
    print(f"Average original counter-claims per claim: {stats['avg_original_counter_claims_per_claim']:.2f}")
    print(f"Average enhanced counter-claims per claim: {stats['avg_enhanced_counter_claims_per_claim']:.2f}")
    
    print("\nStrategy distribution for enhanced counter-claims:")
    for strategy, count in stats["strategy_distribution"].items():
        print(f"  {strategy}: {count} ({count/stats['total_enhanced_counter_claims']*100:.1f}%)")
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()