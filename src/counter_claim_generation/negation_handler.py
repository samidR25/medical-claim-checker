"""
Module for handling negation in claims.
This creates counter-claims by introducing or removing negations.
"""

import spacy
from typing import List, Dict, Any
import re

def flip_negation(claim: str, doc=None) -> List[Dict[str, Any]]:
    """
    Generate counter-claims by flipping negations.
    
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
    
    # Strategy 1: Add negation to main verb if not negated
    negated_verb = add_verb_negation(claim, doc)
    if negated_verb and negated_verb != claim:
        candidates.append({
            "counter_claim": negated_verb,
            "strategy": "add_verb_negation",
            "modified_component": "main verb"
        })
    
    # Strategy 2: Remove negation from main verb if negated
    unnegated_verb = remove_verb_negation(claim, doc)
    if unnegated_verb and unnegated_verb != claim:
        candidates.append({
            "counter_claim": unnegated_verb,
            "strategy": "remove_verb_negation",
            "modified_component": "main verb"
        })
    
    # Strategy 3: Flip "no evidence" to "evidence" and vice versa
    flipped_evidence = flip_evidence_phrase(claim)
    if flipped_evidence and flipped_evidence != claim:
        candidates.append({
            "counter_claim": flipped_evidence,
            "strategy": "flip_evidence_phrase",
            "modified_component": "evidence phrase"
        })
    
    # Strategy 4: Handle specific COVID-19 negation patterns
    covid_negations = flip_covid_specific_negations(claim)
    candidates.extend(covid_negations)
    
    return candidates

def add_verb_negation(claim: str, doc) -> str:
    """Add negation to the main verb if it's not already negated"""
    # Find the main verb
    main_verb = None
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
            # Check if it's already negated
            if any(child.dep_ == "neg" for child in token.children):
                return None  # Already negated
            main_verb = token
            break
    
    if not main_verb:
        return None
    
    # Add negation based on the verb form
    if main_verb.tag_ in ["VBP", "VBZ"]:  # Present tense
        if main_verb.text.lower() in ["is", "are", "was", "were", "have", "has", "had"]:
            # For be verbs and auxiliaries
            negation = main_verb.text + " not"
        else:
            # For other verbs
            if main_verb.tag_ == "VBZ":  # 3rd person singular
                negation = "does not " + main_verb.lemma_
            else:
                negation = "do not " + main_verb.lemma_
    elif main_verb.tag_ == "VBD":  # Past tense
        if main_verb.text.lower() in ["was", "were", "had"]:
            # For be verbs and auxiliaries
            negation = main_verb.text + " not"
        else:
            # For other verbs
            negation = "did not " + main_verb.lemma_
    elif main_verb.tag_ == "MD":  # Modal
        negation = main_verb.text + " not"
    else:
        # For other cases, simple approach
        negation = "not " + main_verb.text
    
    # Replace the verb with its negated form
    return claim.replace(main_verb.text, negation)

def remove_verb_negation(claim: str, doc) -> str:
    """Remove negation from the main verb if it's negated"""
    # Patterns to match common negations
    patterns = [
        (r"(do|does|did|is|are|was|were|have|has|had|can|could|will|would|should|might|may) not", r"\1"),
        (r"(do|does|did|is|are|was|were|have|has|had|can|could|will|would|should|might|may)n't", r"\1"),
        (r"(cannot|can't)", "can"),
        (r"(won't)", "will"),
        (r"(shouldn't)", "should"),
        (r"not ", "")
    ]
    
    # Try to find and remove negations
    for pattern, replacement in patterns:
        if re.search(pattern, claim, re.IGNORECASE):
            return re.sub(pattern, replacement, claim, flags=re.IGNORECASE)
    
    return None

def flip_evidence_phrase(claim: str) -> str:
    """Flip phrases about evidence"""
    # Patterns to match and their replacements
    patterns = [
        (r"no evidence (that|for|of)", "evidence that"),
        (r"no proof (that|for|of)", "proof that"),
        (r"not proven (that|to)", "proven that"),
        (r"evidence (that|for|of)", "no evidence that"),
        (r"proof (that|for|of)", "no proof that"),
        (r"proven (that|to)", "not proven that")
    ]
    
    # Try to flip evidence phrases
    for pattern, replacement in patterns:
        if re.search(pattern, claim, re.IGNORECASE):
            return re.sub(pattern, replacement, claim, flags=re.IGNORECASE)
    
    return None

def flip_covid_specific_negations(claim: str) -> List[Dict[str, Any]]:
   """Handle specific COVID-19 negation patterns"""
   candidates = []
   
   # COVID-specific patterns to match and their replacements
   patterns = [
       # Transmission
       (r"(is|are|can be) transmitted", "cannot be transmitted"),
       (r"(is|are|can) not( be)? transmitted", "can be transmitted"),
       
       # Protection
       (r"(protect|prevents|stops|blocks)", "does not protect against"),
       (r"(does not|doesn't|do not|don't) protect", "protects"),
       
       # Effectiveness
       (r"(is|are) effective", "is not effective"),
       (r"(is|are) not effective", "is effective"),
       
       # Safety
       (r"(is|are) safe", "is not safe"),
       (r"(is|are) not safe", "is safe"),
       
       # Symptoms
       (r"(cause|causes|causing)( the)? symptoms", "does not cause symptoms"),
       (r"(does not|doesn't) cause symptoms", "causes symptoms"),
       
       # Risk
       (r"(increase|increases|increased)( the)? risk", "does not increase risk"),
       (r"(does not|doesn't) increase risk", "increases risk"),
       
       # Immunity
       (r"(provide|provides|provided)( the)? immunity", "does not provide immunity"),
       (r"(does not|doesn't) provide immunity", "provides immunity")
   ]
   
   # Try to apply each pattern
   for pattern, replacement in patterns:
       match = re.search(pattern, claim, re.IGNORECASE)
       if match:
           # Create a counter-claim
           counter_claim = re.sub(pattern, replacement, claim, flags=re.IGNORECASE)
           candidates.append({
               "counter_claim": counter_claim,
               "strategy": "covid_specific_negation",
               "modified_component": match.group(0),
               "replacement": replacement
           })
   
   return candidates