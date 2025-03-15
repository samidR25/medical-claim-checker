"""
Module for substituting medical entities in claims with contradictory ones.
This creates counter-claims by replacing key entities with opposites or alternatives.
"""

import spacy
from typing import List, Dict, Any
import random

# COVID-19 specific entity substitution mappings
ENTITY_SUBSTITUTIONS = {
    # Transmission-related
    "airborne": ["not airborne", "waterborne", "foodborne"],
    "droplets": ["aerosols", "fomites", "direct contact"],
    
    # Effectiveness-related
    "effective": ["ineffective", "harmful", "useless"],
    "ineffective": ["effective", "beneficial", "helpful"],
    
    # Treatment-related
    "treatment": ["no treatment", "prevention", "diagnostic"],
    "cure": ["no cure", "palliative care", "supportive treatment"],
    
    # Risk-related
    "high risk": ["low risk", "no risk", "minimal risk"],
    "low risk": ["high risk", "extreme risk", "severe risk"],
    
    # Population-related
    "children": ["adults", "elderly", "pregnant women"],
    "elderly": ["children", "young adults", "teenagers"],
    
    # Symptom-related
    "symptom": ["not a symptom", "side effect", "unrelated condition"],
    "asymptomatic": ["symptomatic", "severely symptomatic"],
    
    # Specific COVID terms
    "coronavirus": ["influenza virus", "common cold virus", "rhinovirus"],
    "COVID-19": ["influenza", "common cold", "seasonal flu"],
    "SARS-CoV-2": ["influenza virus", "rhinovirus", "adenovirus"],
    
    # Vaccines
    "vaccine": ["placebo", "saline injection", "vitamin supplement"],
    "vaccinated": ["unvaccinated", "unexposed", "immune"],
    "immunity": ["no immunity", "susceptibility", "vulnerability"],
    
    # Public health measures
    "masks": ["face shields", "no protection", "hand washing"],
    "social distancing": ["crowding", "close contact", "normal interaction"],
    "lockdown": ["reopening", "normal operation", "unrestricted movement"]
}

def substitute_entities(claim: str, doc=None) -> List[Dict[str, Any]]:
    """
    Generate counter-claims by substituting medical entities.
    
    Args:
        claim: The original claim
        doc: Optional pre-processed spaCy doc
        
    Returns:
        List of counter-claim candidates with metadata
    """
    if doc is None:
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(claim)
    
    candidates = []
    
    # Method 1: Look for exact matches in our substitution dictionary
    for original, replacements in ENTITY_SUBSTITUTIONS.items():
        if original.lower() in claim.lower():
            # Create a counter-claim for each possible replacement
            for replacement in replacements:
                # Use regex to replace while preserving case
                import re
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                counter_claim = pattern.sub(replacement, claim)
                
                candidates.append({
                    "counter_claim": counter_claim,
                    "strategy": "entity_substitution",
                    "original_entity": original,
                    "replacement_entity": replacement
                })
    
    # Method 2: Use spaCy's entity recognition for medical terms
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE", "PERSON", "NORP"]:
            # For organizations, products, locations, persons, and nationalities
            # we can create a counter-claim by replacing with a generic alternative
            replacements = []
            
            if ent.label_ == "ORG" and any(term in ent.text.lower() for term in ["cdc", "who", "fda", "nih"]):
                # For health organizations
                replacements = ["researchers", "scientists", "doctors", "experts"]
            elif ent.label_ == "PRODUCT" and any(term in ent.text.lower() for term in ["vaccine", "treatment", "drug"]):
                # For medical products
                replacements = ["placebo", "alternative treatment", "untested remedy"]
            elif ent.label_ == "GPE" and len(ent.text.split()) == 1:  # Only single-word locations
                # For countries/locations
                replacements = ["other countries", "different regions", "most places"]
            
            if replacements:
                replacement = random.choice(replacements)
                counter_claim = claim.replace(ent.text, replacement)
                
                candidates.append({
                    "counter_claim": counter_claim,
                    "strategy": "entity_replacement",
                    "original_entity": ent.text,
                    "replacement_entity": replacement,
                    "entity_type": ent.label_
                })
    
    return candidates