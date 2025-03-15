"""
Module for modifying numerical quantities in claims.
This creates counter-claims by changing numbers, percentages, and other quantifiable expressions.
"""

import re
import random
from typing import List, Dict, Any
import spacy
from spacy.tokens import Doc

def modify_quantities(claim: str, doc=None) -> List[Dict[str, Any]]:
    """
    Generate counter-claims by modifying numerical quantities.
    
    Args:
        claim: The original claim
        doc: Optional pre-processed spaCy doc
        
    Returns:
        List of counter-claim candidates with metadata
    """
    if doc is None:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(claim)
    
    candidates = []
    
    # Find all numbers in the claim
    quantities = []
    for token in doc:
        if token.like_num:
            quantities.append({
                "text": token.text,
                "index": token.i,
                "is_percent": any(t.text in ["%", "percent", "percentage"] 
                                 for t in token.rights)
            })
    
    # Also find numbers with regex (for more complex cases)
    regex_quantities = find_quantities_with_regex(claim)
    for q in regex_quantities:
        if not any(q["text"] == existing["text"] for existing in quantities):
            quantities.append(q)
    
    # Generate counter-claims for each quantity
    for quantity in quantities:
        # Determine if it's a percentage or an absolute number
        if quantity["is_percent"]:
            counter_claims = modify_percentage(claim, quantity)
        else:
            counter_claims = modify_number(claim, quantity)
        
        candidates.extend(counter_claims)
    
    return candidates

def find_quantities_with_regex(text: str) -> List[Dict[str, Any]]:
    """
    Find numerical quantities using regex patterns.
    
    Args:
        text: The text to search in
        
    Returns:
        List of quantity dictionaries with text and position
    """
    quantities = []
    
    # Pattern for numbers (including decimals)
    num_pattern = r'\b\d+(?:\.\d+)?\b'
    
    # Pattern for percentages
    percent_pattern = r'\b\d+(?:\.\d+)?(?:\s*%)?\s*(?:percent|percentage)\b'
    
    # Find all percentages
    for match in re.finditer(percent_pattern, text, re.IGNORECASE):
        quantities.append({
            "text": match.group(0),
            "index": match.start(),
            "is_percent": True
        })
    
    # Find all other numbers
    for match in re.finditer(num_pattern, text):
        # Skip if this number is part of a percentage we already found
        if not any(q["index"] <= match.start() < q["index"] + len(q["text"]) 
                 for q in quantities):
            quantities.append({
                "text": match.group(0),
                "index": match.start(),
                "is_percent": False
            })
    
    return quantities

def modify_percentage(claim: str, quantity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate counter-claims by modifying a percentage.
    
    Args:
        claim: The original claim
        quantity: Dictionary with the percentage information
        
    Returns:
        List of counter-claim candidates
    """
    candidates = []
    original_text = quantity["text"]
    
    # Extract the numeric value
    numeric_match = re.search(r'\d+(?:\.\d+)?', original_text)
    if not numeric_match:
        return candidates
    
    original_value = float(numeric_match.group(0))
    
    # Different modifications based on the value
    modifications = []
    
    # Inversion (less than 50% -> more than 50%, or vice versa)
    if original_value < 50:
        modifications.append({
            "strategy": "percentage_inversion",
            "new_value": random.randint(50, 95)
        })
    elif original_value > 50:
        modifications.append({
            "strategy": "percentage_inversion",
            "new_value": random.randint(5, 49)
        })
    
    # Significant increase/decrease
    if original_value < 80:
        modifications.append({
            "strategy": "percentage_increase",
            "new_value": min(99, original_value + random.randint(20, 80))
        })
    if original_value > 20:
        modifications.append({
            "strategy": "percentage_decrease",
            "new_value": max(1, original_value - random.randint(20, 80))
        })
    
    # Apply modifications
    for mod in modifications:
        # Format as the original (preserve decimal places if present)
        if '.' in numeric_match.group(0):
            new_text = f"{mod['new_value']:.1f}"
        else:
            new_text = str(int(mod['new_value']))
        
        # Replace the numeric part while keeping the "percent" text
        new_quantity = original_text.replace(numeric_match.group(0), new_text)
        counter_claim = claim.replace(original_text, new_quantity)
        
        candidates.append({
            "counter_claim": counter_claim,
            "strategy": mod["strategy"],
            "original_quantity": original_text,
            "modified_quantity": new_quantity
        })
    
    return candidates

def modify_number(claim: str, quantity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate counter-claims by modifying a number.
    
    Args:
        claim: The original claim
        quantity: Dictionary with the number information
        
    Returns:
        List of counter-claim candidates
    """
    candidates = []
    original_text = quantity["text"]
    
    # Extract the numeric value
    try:
        original_value = float(original_text)
    except ValueError:
        return candidates  # Skip if we can't convert to a number
    
    # Different modifications based on the value
    modifications = []
    
    # For very large numbers (e.g., COVID-19 case counts)
    if original_value >= 1000:
        # Significant decrease
        modifications.append({
            "strategy": "large_number_decrease",
            "new_value": original_value / random.randint(5, 20)
        })
        # Significant increase
        modifications.append({
            "strategy": "large_number_increase",
            "new_value": original_value * random.randint(2, 10)
        })
    # For smaller numbers
    else:
        # Add/subtract a significant amount
        factor = max(2, int(original_value / 2))
        modifications.append({
            "strategy": "number_increase",
            "new_value": original_value + random.randint(factor, factor*3)
        })
        if original_value > 2:
            modifications.append({
                "strategy": "number_decrease",
                "new_value": max(0, original_value - random.randint(1, factor))
            })
    
    # Apply modifications
    for mod in modifications:
        # Format as the original (preserve decimal places if present)
        if original_value == int(original_value):
            new_text = str(int(mod['new_value']))
        else:
            decimals = len(original_text.split('.')[-1]) if '.' in original_text else 1
            new_text = f"{mod['new_value']:.{decimals}f}"
        
        counter_claim = claim.replace(original_text, new_text)
        
        candidates.append({
            "counter_claim": counter_claim,
            "strategy": mod["strategy"],
            "original_quantity": original_text,
            "modified_quantity": new_text
        })
    
    return candidates