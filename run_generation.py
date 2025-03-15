"""
Script to run the enhanced counter-claim generation system.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.counter_claim_generation.generator import EnhancedCounterClaimGenerator

def generate_counter_claims_for_text(input_text: str, generator: EnhancedCounterClaimGenerator,
                                   num_candidates: int = 3) -> Dict[str, Any]:
    """
    Generate counter-claims for a single input text.
    
    Args:
        input_text: The claim text
        generator: Counter-claim generator
        num_candidates: Number of counter-claims to generate
        
    Returns:
        Dictionary with original claim and generated counter-claims
    """
    counter_claims = generator.generate_counter_claims(
        input_text, num_candidates=num_candidates)
    
    return {
        "original_claim": input_text,
        "generated_counter_claims": counter_claims
    }

def main():
    parser = argparse.ArgumentParser(description="Generate counter-claims")
    parser.add_argument("--text", type=str, help="Input text to generate counter-claims for")
    parser.add_argument("--input-file", type=str, help="Input JSON(L) file with claims")
    parser.add_argument("--output", type=str, default="counter_claims_output.json", 
                      help="Output file path")
    parser.add_argument("--num-candidates", type=int, default=3, 
                      help="Number of counter-claims to generate per claim")
    parser.add_argument("--strategies", type=str, default="all", 
                      help="Comma-separated list of strategies to use (srl,entity,quantity,negation)")
    args = parser.parse_args()
    
    # Parse strategies
    if args.strategies == "all":
        use_srl = use_entity = use_quantity = use_negation = True
    else:
        strategies = args.strategies.split(",")
        use_srl = "srl" in strategies
        use_entity = "entity" in strategies
        use_quantity = "quantity" in strategies
        use_negation = "negation" in strategies
    
    # Initialize the generator
    generator = EnhancedCounterClaimGenerator(
        use_srl=use_srl,
        use_entity_sub=use_entity,
        use_quantity_mod=use_quantity,
        use_negation=use_negation
    )
    
    # Process input
    if args.text:
        # Generate for a single text
        result = generate_counter_claims_for_text(
            args.text, generator, args.num_candidates)
        results = [result]
    elif args.input_file:
        # Generate for multiple claims from a file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            # Check if it's a JSONL file
            if args.input_file.endswith('.jsonl'):
                claims = [json.loads(line.strip())["claim"] for line in f]
            else:
                # Assume regular JSON
                data = json.load(f)
                claims = [item["claim"] for item in data]
        
        results = []
        for claim in claims:
            result = generate_counter_claims_for_text(
                claim, generator, args.num_candidates)
            results.append(result)
    else:
        parser.error("Either --text or --input-file must be provided")
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} claims")
    print(f"Generated {sum(len(r['generated_counter_claims']) for r in results)} counter-claims")
    print(f"Results saved to {args.output}")
    
    # Print sample results
    if results:
        sample = results[0]
        print("\nSample result:")
        print(f"Original claim: {sample['original_claim']}")
        print("Generated counter-claims:")
        for i, cc in enumerate(sample['generated_counter_claims']):
            print(f"{i+1}. {cc['counter_claim']} (Strategy: {cc['strategy']})")

if __name__ == "__main__":
    main()