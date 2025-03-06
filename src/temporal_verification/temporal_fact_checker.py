# src/temporal_verification/temporal_fact_checker.py

import json
import torch
import argparse
from datetime import datetime
from src.models.temporal_verifier import TemporalVerifier
from src.evidence_retrieval.evidence_retriever import EvidenceRetriever

class TemporalFactChecker:
    def __init__(self, model_path="../models/temporal_verifier.pt", kb_path="../data/processed/temporal_kb.json"):
        self.verifier = TemporalVerifier(model_path, kb_path)
        self.retriever = EvidenceRetriever()  # You would implement this based on the COVID-Fact evidence retrieval
    
    def check_claim(self, claim, publication_date=None):
        """Check a claim considering its temporal context"""
        
        # Extract temporal information
        if publication_date:
            try:
                pub_date = datetime.fromisoformat(publication_date)
            except:
                pub_date = datetime.now()
                publication_date = pub_date.isoformat()
        else:
            pub_date = datetime.now()
            publication_date = pub_date.isoformat()
        
        # Determine pandemic phase
        phases = [
            (datetime(2019, 12, 1), datetime(2020, 3, 10), "early_outbreak"),
            (datetime(2020, 3, 11), datetime(2020, 12, 31), "first_wave"),
            (datetime(2021, 1, 1), datetime(2021, 6, 30), "vaccination_phase"),
            (datetime(2021, 7, 1), datetime(2022, 12, 31), "variant_phase")
        ]
        
        pandemic_phase = "unknown"
        for start, end, phase in phases:
            if start <= pub_date <= end:
                pandemic_phase = phase
                break
        
        # Retrieve evidence
        evidence = self.retriever.retrieve_evidence(claim)
        
        # Verify the claim
        result = self.verifier.verify(
            claim, 
            evidence, 
            publication_date=publication_date,
            pandemic_phase=pandemic_phase
        )
        
        # Add additional temporal context
        result["claim"] = claim
        result["evidence"] = evidence
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Temporal-aware fact-checking for COVID-19 claims")
    parser.add_argument("--claim", type=str, required=True, help="The claim to verify")
    parser.add_argument("--date", type=str, help="Publication date in ISO format (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    checker = TemporalFactChecker()
    result = checker.check_claim(args.claim, args.date)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()