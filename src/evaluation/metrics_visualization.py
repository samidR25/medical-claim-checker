"""
Module for evaluating and visualizing the performance of counter-claim generation methods.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import spacy
from collections import Counter

class CounterClaimEvaluator:
    """Class for evaluating counter-claim generation methods."""
    
    def __init__(self, data_file: str):
        """
        Initialize the evaluator.
        
        Args:
            data_file: Path to the processed data file
        """
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize NLP tools
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.nlp = spacy.load('en_core_web_sm')
        self.rouge = Rouge()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate various evaluation metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "count": self._calculate_count_metrics(),
            "diversity": self._calculate_diversity_metrics(),
            "quality": self._calculate_quality_metrics(),
            "strategy": self._calculate_strategy_metrics(),
            "semantic": self._calculate_semantic_metrics()
        }
        
        return metrics
    
    def _calculate_count_metrics(self) -> Dict[str, Any]:
        """Calculate count-based metrics."""
        total_original_claims = len(self.data)
        total_original_counter_claims = sum(len(item["original_counter_claims"]) for item in self.data)
        total_enhanced_counter_claims = sum(len(item["enhanced_counter_claims"]) for item in self.data)
        
        # Calculate per-claim averages
        claims_with_original_counters = sum(1 for item in self.data if item["original_counter_claims"])
        claims_with_enhanced_counters = sum(1 for item in self.data if item["enhanced_counter_claims"])
        
        avg_original_per_claim = total_original_counter_claims / claims_with_original_counters if claims_with_original_counters else 0
        avg_enhanced_per_claim = total_enhanced_counter_claims / claims_with_enhanced_counters if claims_with_enhanced_counters else 0
        
        return {
            "total_original_claims": total_original_claims,
            "total_original_counter_claims": total_original_counter_claims,
            "total_enhanced_counter_claims": total_enhanced_counter_claims,
            "claims_with_original_counters": claims_with_original_counters,
            "claims_with_enhanced_counters": claims_with_enhanced_counters,
            "avg_original_per_claim": avg_original_per_claim,
            "avg_enhanced_per_claim": avg_enhanced_per_claim,
            "coverage_original": claims_with_original_counters / total_original_claims if total_original_claims else 0,
            "coverage_enhanced": claims_with_enhanced_counters / total_original_claims if total_original_claims else 0
        }
    
    def _calculate_diversity_metrics(self) -> Dict[str, Any]:
        """Calculate diversity metrics."""
        # Extract counter-claims
        original_counters = []
        enhanced_counters = []
        
        for item in self.data:
            for counter in item["original_counter_claims"]:
                original_counters.append(counter["claim"])
            
            for counter in item["enhanced_counter_claims"]:
                enhanced_counters.append(counter["claim"])
        
        # Calculate lexical diversity (unique tokens / total tokens)
        original_tokens = set()
        original_total = 0
        for claim in original_counters:
            tokens = [token.text.lower() for token in self.nlp(claim) if not token.is_punct and not token.is_stop]
            original_tokens.update(tokens)
            original_total += len(tokens)
        
        enhanced_tokens = set()
        enhanced_total = 0
        for claim in enhanced_counters:
            tokens = [token.text.lower() for token in self.nlp(claim) if not token.is_punct and not token.is_stop]
            enhanced_tokens.update(tokens)
            enhanced_total += len(tokens)
        
        original_diversity = len(original_tokens) / original_total if original_total else 0
        enhanced_diversity = len(enhanced_tokens) / enhanced_total if enhanced_total else 0
        
        # Calculate average length
        original_lengths = [len(claim.split()) for claim in original_counters]
        enhanced_lengths = [len(claim.split()) for claim in enhanced_counters]
        
        avg_original_length = sum(original_lengths) / len(original_lengths) if original_lengths else 0
        avg_enhanced_length = sum(enhanced_lengths) / len(enhanced_lengths) if enhanced_lengths else 0
        
        return {
            "lexical_diversity_original": original_diversity,
            "lexical_diversity_enhanced": enhanced_diversity,
            "avg_length_original": avg_original_length,
            "avg_length_enhanced": avg_enhanced_length,
            "unique_words_original": len(original_tokens),
            "unique_words_enhanced": len(enhanced_tokens)
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics based on available scores."""
        contradiction_scores = []
        for item in self.data:
            for counter in item["enhanced_counter_claims"]:
                if "contradiction_score" in counter:
                    contradiction_scores.append(counter["contradiction_score"])
        
        if not contradiction_scores:
            return {"avg_contradiction_score": 0, "median_contradiction_score": 0}
        
        return {
            "avg_contradiction_score": sum(contradiction_scores) / len(contradiction_scores),
            "median_contradiction_score": sorted(contradiction_scores)[len(contradiction_scores) // 2]
        }
    
    def _calculate_strategy_metrics(self) -> Dict[str, Any]:
        """Calculate strategy distribution metrics."""
        strategy_counts = Counter()
        for item in self.data:
            for counter in item["enhanced_counter_claims"]:
                strategy = counter.get("strategy", "unknown")
                strategy_counts[strategy] += 1
        
        total = sum(strategy_counts.values())
        
        strategy_distribution = {
            strategy: {
                "count": count,
                "percentage": count / total if total else 0
            } for strategy, count in strategy_counts.items()
        }
        
        return {
            "strategy_counts": dict(strategy_counts),
            "strategy_distribution": strategy_distribution
        }
    
    def _calculate_semantic_metrics(self) -> Dict[str, Any]:
        """Calculate semantic similarity metrics between original claims and counter-claims."""
        from sentence_transformers import SentenceTransformer, util
        
        # Load model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        original_similarities = []
        enhanced_similarities = []
        
        for item in self.data:
            original_claim = item["original_claim"]["claim"]
            original_embedding = model.encode([original_claim])[0]
            
            # Calculate similarities for original counter-claims
            for counter in item["original_counter_claims"]:
                counter_embedding = model.encode([counter["claim"]])[0]
                similarity = util.cos_sim(original_embedding, counter_embedding).item()
                original_similarities.append(similarity)
            
            # Calculate similarities for enhanced counter-claims
            for counter in item["enhanced_counter_claims"]:
                counter_embedding = model.encode([counter["claim"]])[0]
                similarity = util.cos_sim(original_embedding, counter_embedding).item()
                enhanced_similarities.append(similarity)
        
        return {
            "avg_similarity_original": sum(original_similarities) / len(original_similarities) if original_similarities else 0,
            "avg_similarity_enhanced": sum(enhanced_similarities) / len(enhanced_similarities) if enhanced_similarities else 0,
            "median_similarity_original": sorted(original_similarities)[len(original_similarities) // 2] if original_similarities else 0,
            "median_similarity_enhanced": sorted(enhanced_similarities)[len(enhanced_similarities) // 2] if enhanced_similarities else 0
        }
    
    def visualize_metrics(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """
        Create visualizations for the metrics.
        
        Args:
            metrics: Metrics dictionary
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Coverage and count comparison
        self._plot_coverage_comparison(metrics, output_dir)
        
        # 2. Diversity comparison
        self._plot_diversity_comparison(metrics, output_dir)
        
        # 3. Strategy distribution
        self._plot_strategy_distribution(metrics, output_dir)
        
        # 4. Similarity comparison
        self._plot_similarity_comparison(metrics, output_dir)
        
        # 5. Overall comparison summary
        self._plot_summary_comparison(metrics, output_dir)
    
    def _plot_coverage_comparison(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Plot coverage comparison."""
        count_metrics = metrics["count"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coverage plot
        labels = ['Original', 'Enhanced']
        coverage = [count_metrics["coverage_original"] * 100, count_metrics["coverage_enhanced"] * 100]
        
        ax1.bar(labels, coverage, color=['#1f77b4', '#ff7f0e'])
        ax1.set_ylabel('Coverage (%)')
        ax1.set_title('Claim Coverage Comparison')
        ax1.set_ylim(0, 100)
        
        for i, v in enumerate(coverage):
            ax1.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        # Counter-claims per claim
        avg_counters = [count_metrics["avg_original_per_claim"], count_metrics["avg_enhanced_per_claim"]]
        
        ax2.bar(labels, avg_counters, color=['#1f77b4', '#ff7f0e'])
        ax2.set_ylabel('Average counter-claims per claim')
        ax2.set_title('Counter-Claims per Claim')
        
        for i, v in enumerate(avg_counters):
            ax2.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_diversity_comparison(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Plot diversity comparison."""
        diversity_metrics = metrics["diversity"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Lexical diversity
        labels = ['Original', 'Enhanced']
        diversity = [diversity_metrics["lexical_diversity_original"], diversity_metrics["lexical_diversity_enhanced"]]
        
        ax1.bar(labels, diversity, color=['#1f77b4', '#ff7f0e'])
        ax1.set_ylabel('Lexical Diversity')
        ax1.set_title('Lexical Diversity Comparison')
        
        for i, v in enumerate(diversity):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Average length
        avg_lengths = [diversity_metrics["avg_length_original"], diversity_metrics["avg_length_enhanced"]]
        
        ax2.bar(labels, avg_lengths, color=['#1f77b4', '#ff7f0e'])
        ax2.set_ylabel('Average Length (words)')
        ax2.set_title('Average Counter-Claim Length')
        
        for i, v in enumerate(avg_lengths):
            ax2.text(i, v + 0.1, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diversity_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_strategy_distribution(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Plot strategy distribution."""
        strategy_metrics = metrics["strategy"]
        
        # Create pie chart of strategy distribution
        plt.figure(figsize=(10, 7))
        
        strategy_counts = strategy_metrics["strategy_counts"]
        labels = list(strategy_counts.keys())
        sizes = list(strategy_counts.values())
        
        # Sort by size
        sorted_data = sorted(zip(labels, sizes), key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in sorted_data]
        sizes = [x[1] for x in sorted_data]
        
        # If there are many strategies, limit to top N and group the rest
        if len(labels) > 7:
            top_n = 6
            other_count = sum(sizes[top_n:])
            labels = labels[:top_n] + ['Other']
            sizes = sizes[:top_n] + [other_count]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Enhanced Counter-Claim Generation Strategies')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_similarity_comparison(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Plot similarity comparison."""
        semantic_metrics = metrics["semantic"]
        
        plt.figure(figsize=(8, 6))
        
        labels = ['Original', 'Enhanced']
        similarities = [semantic_metrics["avg_similarity_original"], semantic_metrics["avg_similarity_enhanced"]]
        
        plt.bar(labels, similarities, color=['#1f77b4', '#ff7f0e'])
        plt.ylabel('Average Semantic Similarity')
        plt.title('Semantic Similarity to Original Claims')
        plt.ylim(0, 1)
        
        for i, v in enumerate(similarities):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_summary_comparison(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Plot overall summary comparison."""
        # Create radar chart comparing multiple metrics
        categories = ['Coverage', 'Counter-claims per claim', 'Lexical diversity', 'Contradiction score']
        
        count_metrics = metrics["count"]
        diversity_metrics = metrics["diversity"]
        quality_metrics = metrics["quality"]
        
        # Normalize values for radar chart
        original_values = [
            count_metrics["coverage_original"],
            count_metrics["avg_original_per_claim"] / 5,  # Normalize to 0-1 range assuming max 5 claims
            diversity_metrics["lexical_diversity_original"],
            0.5  # Baseline value for original (no contradiction score available)
        ]
        
        enhanced_values = [
            count_metrics["coverage_enhanced"],
            count_metrics["avg_enhanced_per_claim"] / 5,  # Normalize to 0-1 range
            diversity_metrics["lexical_diversity_enhanced"],
            quality_metrics["avg_contradiction_score"]
        ]
        
        # Create radar chart
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot values
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        original_values += original_values[:1]
        enhanced_values += enhanced_values[:1]
        
        ax.plot(angles, original_values, 'b-', linewidth=2, label='Original')
        ax.fill(angles, original_values, 'b', alpha=0.1)
        
        ax.plot(angles, enhanced_values, 'r-', linewidth=2, label='Enhanced')
        ax.fill(angles, enhanced_values, 'r', alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        plt.legend(loc='upper right')
        
        plt.title('Overall Comparison of Counter-Claim Generation Methods')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=300)
        plt.close()