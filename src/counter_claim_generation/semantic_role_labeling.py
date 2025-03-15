"""
Module for extracting semantic roles from claims to understand their structure.
This helps generate more meaningful counter-claims by identifying key components.
"""

import spacy
from spacy.tokens import Doc
from typing import Dict, Any, Optional

# Try to use a pre-trained SRL model if available
try:
    import allennlp
    from allennlp.predictors.predictor import Predictor
    HAS_ALLENNLP = True
except ImportError:
    HAS_ALLENNLP = False

def extract_semantic_roles(claim: str) -> Dict[str, Any]:
    """
    Extract semantic roles from a claim using AllenNLP's SRL model
    or a rule-based approach as fallback.
    
    Args:
        claim: The claim to analyze
        
    Returns:
        Dictionary with semantic roles as keys and text spans as values
    """
    if HAS_ALLENNLP:
        return extract_with_allennlp(claim)
    else:
        return extract_with_rules(claim)
    
def extract_with_allennlp(claim: str) -> Dict[str, Any]:
    """
    Use AllenNLP's SRL model to extract semantic roles.
    
    Args:
        claim: The claim to analyze
        
    Returns:
        Dictionary with semantic roles
    """
    try:
        # Load the SRL predictor (download if needed)
        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        )
        
        # Get predictions
        prediction = predictor.predict(sentence=claim)
        
        # Extract verb-argument structures
        verbs = prediction.get("verbs", [])
        
        # If no verbs found, return empty
        if not verbs:
            return {}
            
        # Take the main verb (usually the one with most arguments)
        main_verb = max(verbs, key=lambda x: len(x.get("tags", [])))
        
        # Extract arguments
        roles = {}
        for arg_tag, arg_text in extract_arguments(main_verb):
            roles[arg_tag] = arg_text
            
        # Add the main verb
        if "V" in roles:
            roles["action"] = roles["V"]
            
        return roles
        
    except Exception as e:
        print(f"Error in AllenNLP SRL: {e}")
        return {}
    
def extract_arguments(verb_structure):
    """
    Extract argument spans from AllenNLP output format.
    
    Args:
        verb_structure: Output from AllenNLP SRL
        
    Returns:
        List of (argument_tag, argument_text) tuples
    """
    tags = verb_structure.get("tags", [])
    words = verb_structure.get("words", [])
    
    arguments = []
    current_tag = None
    current_span = []
    
    for tag, word in zip(tags, words):
        # Format is "B-ARG0", "I-ARG0", etc. or "O" for non-arguments
        if tag.startswith("B-"):
            # A new argument starts
            if current_tag:
                # Store the previous argument
                arguments.append((current_tag, " ".join(current_span)))
            current_tag = tag[2:]  # Remove "B-"
            current_span = [word]
        elif tag.startswith("I-"):
            # Continuation of current argument
            if current_tag and current_tag == tag[2:]:
                current_span.append(word)
        elif tag == "O":
            # Outside any argument
            if current_tag:
                # Store the previous argument
                arguments.append((current_tag, " ".join(current_span)))
                current_tag = None
                current_span = []
    
    # Don't forget the last argument
    if current_tag:
        arguments.append((current_tag, " ".join(current_span)))
    
    return arguments
    
def extract_with_rules(claim: str) -> Dict[str, Any]:
    """
    Use rule-based approach to extract semantic-like roles.
    This is a simplified approach when AllenNLP is not available.
    
    Args:
        claim: The claim to analyze
        
    Returns:
        Dictionary with semantic-like roles
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(claim)
    
    roles = {}
    
    # Find the main verb
    main_verb = None
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
            main_verb = token
            break
    
    if not main_verb:
        return roles
    
    # Add the main verb as the action
    roles["action"] = main_verb.text
    
    # Find the subject (agent)
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"] and token.head == main_verb:
            # Get the full noun phrase
            start = token.i
            while start > 0 and doc[start-1].dep_ in ["compound", "amod", "det"]:
                start -= 1
            end = token.i + 1
            while end < len(doc) and doc[end].dep_ in ["prep", "pobj", "compound"]:
                end += 1
            roles["agent"] = doc[start:end].text
    
    # Find the object
    for token in doc:
        if token.dep_ in ["dobj", "pobj"] and token.head == main_verb:
            # Get the full noun phrase
            start = token.i
            while start > 0 and doc[start-1].dep_ in ["compound", "amod", "det"]:
                start -= 1
            end = token.i + 1
            roles["object"] = doc[start:end].text
    
    return roles