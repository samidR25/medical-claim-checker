# src/evaluation/evaluate.py

import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.models.temporal_verifier import TemporalVerifier

def evaluate_temporal_verification():
    """Evaluate the temporal verification model on the test set"""
    
    # Load test data
    with open('../data/processed/temporal_test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Load model
    verifier = TemporalVerifier(model_path="../models/temporal_verifier.pt")
    
    # Evaluate
    predictions = []
    true_labels = []
    
    for item in tqdm(test_data):
        # Get true label
        true_label = item.get("temporal_label", item["label"])
        
        # Get prediction
        claim = item["claim"]
        evidence = item["evidence"]
        pub_date = item["temporal_info"].get("publication_date")
        phase = item["temporal_info"].get("pandemic_phase")
        
        result = verifier.verify(claim, evidence, pub_date, phase)
        prediction = result["prediction"]
        
        predictions.append(prediction)
        true_labels.append(true_label)
    
    # Calculate metrics
    labels = ["SUPPORTED", "REFUTED", "TEMPORAL_CONFLICT"]
    report = classification_report(true_labels, predictions, target_names=labels)
    conf_matrix = confusion_matrix(
        [labels.index(l) for l in true_labels], 
        [labels.index(p) for p in predictions]
    )
    
    # Print results
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('../experiments/confusion_matrix.png')
    
    # Analyze temporal conflicts
    temporal_conflicts = [item for item in test_data if item.get("temporal_label") == "TEMPORAL_CONFLICT"]
    
    # Group by pandemic phase
    phase_counts = {}
    for item in temporal_conflicts:
        phase = item["temporal_info"].get("pandemic_phase", "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    # Plot phase distribution
    plt.figure(figsize=(10, 6))
    plt.bar(phase_counts.keys(), phase_counts.values())
    plt.xlabel('Pandemic Phase')
    plt.ylabel('Count')
    plt.title('Temporal Conflicts by Pandemic Phase')
    plt.savefig('../experiments/temporal_conflicts_by_phase.png')
    
    # Save results
    results = {
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "temporal_conflicts_by_phase": phase_counts
    }
    
    with open('../experiments/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate_temporal_verification()