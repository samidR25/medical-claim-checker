"""
Module for checking if a generated counter-claim actually contradicts the original claim.
This helps filter out generated candidates that aren't truly contradictory.
"""

from typing import Dict, Any, Union, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ContradictionClassifier:
    """A classifier to determine if a counter-claim contradicts an original claim."""
    
    def __init__(self, model_name: str = "roberta-large-mnli"):
        """
        Initialize the contradiction classifier.
        
        Args:
            model_name: Name of the pre-trained NLI model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def check_contradiction(self, 
                           claim: str, 
                           counter_claim: str) -> Dict[str, float]:
        """
        Check if the counter-claim contradicts the original claim.
        
        Args:
            claim: The original claim
            counter_claim: The generated counter-claim
            
        Returns:
            Dictionary with probabilities for contradiction, entailment, and neutral
        """
        # Tokenize
        inputs = self.tokenizer(claim, counter_claim, return_tensors="pt", 
                              truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
        
        # Return probabilities for each class
        # Most NLI models have the labels in order: entailment (0), neutral (1), contradiction (2)
        return {
            "entailment": probs[0],
            "neutral": probs[1],
            "contradiction": probs[2]
        }

# Initialize the classifier (lazy loading)
_classifier = None

def check_contradiction(claim: str, counter_claim: str) -> float:
    """
    Check if the counter-claim contradicts the original claim.
    
    Args:
        claim: The original claim
        counter_claim: The generated counter-claim
        
    Returns:
        Contradiction score (higher means more likely to be contradictory)
    """
    global _classifier
    
    # Lazy load the classifier
    if _classifier is None:
        try:
            _classifier = ContradictionClassifier()
        except Exception as e:
            print(f"Error loading contradiction classifier: {e}")
            # Fallback to a simple heuristic
            return _heuristic_contradiction_score(claim, counter_claim)
    
    try:
        result = _classifier.check_contradiction(claim, counter_claim)
        return result["contradiction"]
    except Exception as e:
        print(f"Error checking contradiction: {e}")
        return _heuristic_contradiction_score(claim, counter_claim)

def _heuristic_contradiction_score(claim: str, counter_claim: str) -> float:
    """
    Calculate a heuristic contradiction score when the model is unavailable.
    
    Args:
        claim: The original claim
        counter_claim: The generated counter-claim
        
    Returns:
        Contradiction score based on simple heuristics
    """
    # Check for negation differences
    neg_words = ["not", "no", "never", "none", "nothing", "isn't", "aren't", "wasn't", 
                "weren't", "doesn't", "don't", "didn't", "cannot", "can't", "won't"]
    
    # Count negation words in each claim
    claim_neg_count = sum(1 for word in neg_words if f" {word} " in f" {claim} ")
    counter_neg_count = sum(1 for word in neg_words if f" {word} " in f" {counter_claim} ")
    
    # If one has negation and the other doesn't, likely a contradiction
    if claim_neg_count != counter_neg_count:
        return 0.8
    
    # Check for opposite qualifier words
    opposite_pairs = [
        ("effective", "ineffective"),
        ("safe", "unsafe"),
        ("proven", "unproven"),
        ("protective", "harmful"),
        ("increase", "decrease"),
        ("more", "less"),
        ("high", "low"),
        ("many", "few"),
        ("most", "least"),
        ("positive", "negative")
    ]
    
    # Check if any opposite pairs appear in the claims
    for word1, word2 in opposite_pairs:
        if (word1 in claim.lower() and word2 in counter_claim.lower()) or \
           (word2 in claim.lower() and word1 in counter_claim.lower()):
            return 0.9
    
    # If the claims are very different, assume medium contradiction
    if len(set(claim.lower().split()) & set(counter_claim.lower().split())) < len(claim.split()) / 2:
        return 0.6
    
    # Default: low contradiction score
    return 0.3