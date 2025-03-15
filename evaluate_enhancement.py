"""
Script to run the full evaluation of the enhanced counter-claim generation system.
"""

import argparse
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.counter_claim_generation.generator import EnhancedCounterClaimGenerator
from src.data_processing.data_loader import (
    load_dataset, extract_original_claims, extract_original_counter_claims,
    add_identifiers, match_counter_claims
)
from src.evaluation.data_comparison import save_processed_dataset, create_comparison_stats
from src.evaluation.metrics_visualization import CounterClaimEvaluator

def run_full_evaluation(input_file, output_dir, num_claims=None, num_candidates=3):
    """
    Run the full evaluation pipeline.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        num_claims: Number of claims to process (None for all)
        num_candidates: Number of counter-claims to generate per claim
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed_data_file = os.path.join(output_dir, 'processed_data.json')
    metrics_output_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(metrics_output_dir, exist_ok=True)
    
    # Step 1: Process the dataset and generate enhanced counter-claims
    if not os.path.exists(processed_data_file):
        print("Processing dataset and generating enhanced counter-claims...")
        
        # Initialize the generator
        generator = EnhancedCounterClaimGenerator(
            use_srl=True,
            use_entity_sub=True,
            use_quantity_mod=True,
            use_negation=True
        )
        
        # Load dataset
        dataset = load_dataset(input_file)
        
        # Extract original claims and counter-claims
        original_claims = extract_original_claims(dataset)
        original_counter_claims = extract_original_counter_claims(dataset)
        
        # Limit number of claims if specified
        if num_claims:
            original_claims = original_claims[:num_claims]
        
        # Add identifiers to original claims
        original_claims = add_identifiers(original_claims)
        
        # Match original counter-claims to original claims
        original_counter_map = match_counter_claims(original_claims, original_counter_claims)
        
        # Generate enhanced counter-claims
        from src.evaluation.evaluate_counter_claims import generate_enhanced_counter_claims
        enhanced_counter_claims = generate_enhanced_counter_claims(
            original_claims, generator, num_candidates)
        
        # Save processed dataset
        save_processed_dataset(
            original_claims,
            original_counter_map,
            enhanced_counter_claims,
            processed_data_file
        )
        
        print(f"Processed data saved to {processed_data_file}")
    else:
        print(f"Using existing processed data from {processed_data_file}")
    
    # Step 2: Calculate metrics and create visualizations
    print("Calculating metrics and creating visualizations...")
    evaluator = CounterClaimEvaluator(processed_data_file)
    metrics = evaluator.calculate_metrics()
    
    # Save metrics to JSON
    import json
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    evaluator.visualize_metrics(metrics, metrics_output_dir)
    
    print(f"Metrics saved to {os.path.join(output_dir, 'metrics.json')}")
    print(f"Visualizations saved to {metrics_output_dir}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total original claims: {metrics['count']['total_original_claims']}")
    print(f"Original counter-claims: {metrics['count']['total_original_counter_claims']}")
    print(f"Enhanced counter-claims: {metrics['count']['total_enhanced_counter_claims']}")
    print(f"Coverage - Original: {metrics['count']['coverage_original']*100:.1f}%")
    print(f"Coverage - Enhanced: {metrics['count']['coverage_enhanced']*100:.1f}%")
    print(f"Avg counter-claims per claim - Original: {metrics['count']['avg_original_per_claim']:.2f}")
    print(f"Avg counter-claims per claim - Enhanced: {metrics['count']['avg_enhanced_per_claim']:.2f}")
    print(f"Lexical diversity - Original: {metrics['diversity']['lexical_diversity_original']:.3f}")
    print(f"Lexical diversity - Enhanced: {metrics['diversity']['lexical_diversity_enhanced']:.3f}")
    print(f"Avg contradiction score (Enhanced): {metrics['quality']['avg_contradiction_score']:.3f}")
    
    print("\nTop generation strategies:")
    for strategy, data in sorted(
        metrics['strategy']['strategy_distribution'].items(), 
        key=lambda x: x[1]['count'], 
        reverse=True
    )[:5]:
        print(f"  {strategy}: {data['count']} ({data['percentage']*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Run full evaluation of enhanced counter-claim generation")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with claims")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-claims", type=int, default=None, help="Number of claims to process")
    parser.add_argument("--num-candidates", type=int, default=3, help="Number of counter-claims to generate per claim")
    args = parser.parse_args()
    
    run_full_evaluation(
        args.input,
        args.output_dir,
        args.num_claims,
        args.num_candidates
    )

if __name__ == "__main__":
    main()