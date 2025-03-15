"""
Enhanced counter-claim generator that coordinates different strategies
for generating counter-claims based on the semantic structure of the original claim.
"""

import spacy
import random
from typing import List, Dict, Any, Tuple

from .semantic_role_labeling import extract_semantic_roles
from .entity_substitution import substitute_entities
from .quantity_modifier import modify_quantities
from .negation_handler import flip_negation
from .contradiction_classifier import check_contradiction

class EnhancedCounterClaimGenerator:
    def __init__(self, use_srl=True, use_entity_sub=True, 
                 use_quantity_mod=True, use_negation=True):
        """
        Initialize the enhanced counter-claim generator with selected strategies.
        
        Args:
            use_srl: Whether to use semantic role labeling
            use_entity_sub: Whether to use entity substitution
            use_quantity_mod: Whether to use quantity modification
            use_negation: Whether to use negation flipping
        """
        self.use_srl = use_srl
        self.use_entity_sub = use_entity_sub
        self.use_quantity_mod = use_quantity_mod
        self.use_negation = use_negation
        
        # Initialize spaCy for text processing
        self.nlp = spacy.load("en_core_web_lg")
        
    def generate_counter_claims(self, claim: str, 
                               num_candidates: int = 5,
                               min_contradiction_score: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate multiple counter-claims for a given claim using various strategies.
        
        Args:
            claim: The original claim
            num_candidates: Number of counter-claim candidates to generate
            min_contradiction_score: Minimum contradiction score for a valid counter-claim
            
        Returns:
            List of dictionaries containing counter-claims and their generation metadata
        """
        doc = self.nlp(claim)
        candidates = []
        
        # Apply different strategies based on the claim structure
        if self.use_srl:
            roles = extract_semantic_roles(claim)
            if roles:
                srl_candidates = self._generate_from_srl(claim, roles, doc)
                candidates.extend(srl_candidates)
        
        if self.use_entity_sub:
            entity_candidates = substitute_entities(claim, doc)
            candidates.extend(entity_candidates)
            
        if self.use_quantity_mod:
            quantity_candidates = modify_quantities(claim, doc)
            candidates.extend(quantity_candidates)
            
        if self.use_negation:
            negation_candidates = flip_negation(claim, doc)
            candidates.extend(negation_candidates)
            
        # Filter candidates by contradiction score
        filtered_candidates = []
        for candidate in candidates:
            contradiction_score = check_contradiction(claim, candidate["counter_claim"])
            if contradiction_score >= min_contradiction_score:
                candidate["contradiction_score"] = contradiction_score
                filtered_candidates.append(candidate)
                
        # Sort by contradiction score and return top candidates
        filtered_candidates.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return filtered_candidates[:num_candidates]
    
    def _generate_from_srl(self, claim: str, roles: Dict, doc) -> List[Dict[str, Any]]:
        """Generate counter-claims using semantic role information"""
        candidates = []
        
        # Example: modify the agent of an action
        if "agent" in roles and "action" in roles:
            agent = roles["agent"]
            action = roles["action"]
            
            # Create counter-claim by replacing agent
            # (Implementation would depend on your SRL structure)
            # This is simplified for illustration
            candidates.append({
                "counter_claim": claim.replace(agent, f"No one"),
                "strategy": "agent_removal",
                "modified_component": agent
            })
            
        # Example: modify the object of an action
        if "object" in roles and "action" in roles:
            obj = roles["object"]
            action = roles["action"]
            
            # Create counter-claim by modifying object
            # This is simplified for illustration
            candidates.append({
                "counter_claim": claim.replace(obj, f"not {obj}"),
                "strategy": "object_negation",
                "modified_component": obj
            })
            
        return candidates