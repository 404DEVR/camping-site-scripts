#!/usr/bin/env python3
"""
Comprehensive Statistical Significance Testing for Machine Learning Models
Fixed version with proper error handling
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, friedmanchisquare
from statsmodels.stats.contingency_tables import mcnemar
import scikit_posthocs as sp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import itertools

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelComparison:
    """
    Comprehensive statistical comparison of machine learning models.
    
    Supports multiple statistical tests and generates publication-ready outputs.
    """
    
    def __init__(self, predictions_file: str, cv_scores_file: Optional[str] = None, 
                 alpha: float = 0.05, output_dir: str = "results"):
        """
        Initialize the model comparison framework.
        
        Args:
            predictions_file: CSV with columns y_true, pred_model1, pred_model2, etc.
            cv_scores_file: Optional CSV with k-fold CV scores for each model
            alpha: Significance level (default: 0.05)
            output_dir: Directory to save results
        """
        self.predictions_file = predictions_file
        self.cv_scores_file = cv_scores_file
        self.alpha = alpha
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.predictions_df = None
        self.cv_scores_df = None
        self.model_names = []
        self.results = {}
        
        self._load_data()
        self._validate_data()
        
    def _load_data(self):
        """Load prediction and CV score data."""
        print("Loading data...")
        
        # Load predictions
        try:
            self.predictions_df = pd.read_csv(self.predictions_file)
            print(f"✓ Loaded predictions: {self.predictions_df.shape}")
        except Exception as e:
            raise ValueError(f"Error loading predictions file: {e}")
        
        # Extract model names (columns starting with 'pred_')
        pred_cols = [col for col in self.predictions_df.columns if col.startswith('pred_')]
        self.model_names = [col.replace('pred_', '') for col in pred_cols]
        print(f"✓ Found {len(self.model_names)} models: {self.model_names}")
        
        # Load CV scores if provided
        if self.cv_scores_file:
            try:
                self.cv_scores_df = pd.read_csv(self.cv_scores_file)
                print(f"✓ Loaded CV scores: {self.cv_scores_df.shape}")
            except Exception as e:
                print(f"Warning: Could not load CV scores file: {e}")
                self.cv_scores_df = None
    
    def _validate_data(self):
        """Validate input data format and completeness."""
        print("Validating data...")
        
        # Check required columns
        if 'y_true' not in self.predictions_df.columns:
            raise ValueError("Missing 'y_true' column in predictions file")
        
        # Check for missing values
        missing_preds = self.predictions_df.isnull().sum().sum()
        if missing_preds > 0:
            print(f"Warning: Found {missing_preds} missing values in predictions")
        
        # Validate CV scores format if provided
        if self.cv_scores_df is not None:
            expected_cols = set(self.model_names)
            actual_cols = set(self.cv_scores_df.columns)
            if not expected_cols.issubset(actual_cols):
                missing = expected_cols - actual_cols
                print(f"Warning: Missing CV score columns: {missing}")
        
        print("✓ Data validation complete")
    
    def mcnemar_test(self, model_a: str, model_b: str) -> Tuple[float, float]:
        """
        Perform McNemar's test for comparing two models' predictions.
        
        Args:
            model_a: Name of first model
            model_b: Name of second model
            
        Returns:
            Tuple of (test_statistic, p_value)
        """
        y_true = self.predictions_df['y_true']
        pred_a = self.predictions_df[f'pred_{model_a}']
        pred_b = self.predictions_df[f'pred_{model_b}']
        
        # Create contingency table
        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)
        
        # McNemar's table: [both_correct, a_correct_b_wrong, a_wrong_b_correct, both_wrong]
        both_correct = np.sum(correct_a & correct_b)
        a_correct_b_wrong = np.sum(correct_a & ~correct_b)
        a_wrong_b_correct = np.sum(~correct_a & correct_b)
        both_wrong = np.sum(~correct_a & ~correct_b)
        
        # Create 2x2 contingency table for McNemar's test
        table = np.array([[both_correct, a_correct_b_wrong],
                         [a_wrong_b_correct, both_wrong]])
        
        try:
            result = mcnemar(table, exact=False, correction=True)
            return result.statistic, result.pvalue
        except Exception as e:
            print(f"Warning: McNemar's test failed for {model_a} vs {model_b}: {e}")
            return np.nan, np.nan
    
    def run_all_tests(self) -> Dict:
        """
        Run all statistical tests and compile results.
        
        Returns:
            Dictionary containing all test results
        """
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*60)
        
        results = {
            'mcnemar_tests': {},
            'model_metrics': {},
            'bootstrap_cis': {},
            'bonferroni_correction': {}
        }
        
        # 1. Calculate basic metrics for each model
        print("\n1. Calculating model performance metrics...")
        for model in self.model_names:
            y_true = self.predictions_df['y_true']
            y_pred = self.predictions_df[f'pred_{model}']
            
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            
            results['model_metrics'][model] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            print(f"  ✓ {model}: Acc={accuracy:.4f}, F1={f1:.4f}")
        
        # 2. McNemar's tests (pairwise)
        print("\n2. Running McNemar's tests (pairwise predictions comparison)...")
        model_pairs = list(itertools.combinations(self.model_names, 2))
        
        for model_a, model_b in model_pairs:
            stat, p_val = self.mcnemar_test(model_a, model_b)
            results['mcnemar_tests'][f"{model_a}_vs_{model_b}"] = {
                'statistic': stat,
                'p_value': p_val
            }
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  ✓ {model_a} vs {model_b}: p={p_val:.4f} {significance}")
        
        # 3. Bootstrap confidence intervals
        print("\n3. Calculating bootstrap confidence intervals...")
        for model in self.model_names:
            ci_lower, ci_upper = self.bootstrap_ci(model)
            results['bootstrap_cis'][model] = {
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            print(f"  ✓ {model}: 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # 4. Bonferroni correction
        print("\n4. Applying Bonferroni correction...")
        n_comparisons = len(model_pairs)
        bonferroni_alpha = self.alpha / n_comparisons
        results['bonferroni_correction'] = {
            'original_alpha': self.alpha,
            'corrected_alpha': bonferroni_alpha,
            'n_comparisons': n_comparisons
        }
        print(f"  ✓ Corrected α = {bonferroni_alpha:.4f} (original α = {self.alpha})")
        
        self.results = results
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*60)
        
        return results
    
    def bootstrap_ci(self, model: str, n_iterations: int = 1000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for model accuracy.
        
        Args:
            model: Model name
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Tuple of (ci_lower, ci_upper)
        """
        try:
            y_true = self.predictions_df['y_true']
            y_pred = self.predictions_df[f'pred_{model}']
            
            n_samples = len(y_true)
            bootstrap_scores = []
            
            np.random.seed(42)  # For reproducibility
            for _ in range(n_iterations):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                y_true_boot = y_true.iloc[indices]
                y_pred_boot = y_pred.iloc[indices]
                
                # Calculate accuracy
                accuracy = accuracy_score(y_true_boot, y_pred_boot)
                bootstrap_scores.append(accuracy)
            
            # Calculate confidence interval
            ci_lower = np.percentile(bootstrap_scores, (self.alpha / 2) * 100)
            ci_upper = np.percentile(bootstrap_scores, (1 - self.alpha / 2) * 100)
            
            return ci_lower, ci_upper
        except Exception as e:
            print(f"Warning: Bootstrap CI failed for {model}: {e}")
            return np.nan, np.nan
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create comprehensive summary table of all results.
        
        Returns:
            DataFrame with model comparison results
        """
        print("Creating summary table...")
        
        summary_data = []
        
        # Find best model (highest accuracy)
        best_model = max(self.model_names, 
                        key=lambda m: self.results['model_metrics'][m]['accuracy'])
        
        for model in self.model_names:
            metrics = self.results['model_metrics'][model]
            bootstrap_ci = self.results['bootstrap_cis'][model]
            
            # Get p-value vs best model (if not the best model itself)
            p_value_vs_best = "-"
            significance = "-"
            
            if model != best_model:
                # Look for McNemar's test result
                key1 = f"{model}_vs_{best_model}"
                key2 = f"{best_model}_vs_{model}"
                
                if key1 in self.results['mcnemar_tests']:
                    p_val = self.results['mcnemar_tests'][key1]['p_value']
                elif key2 in self.results['mcnemar_tests']:
                    p_val = self.results['mcnemar_tests'][key2]['p_value']
                else:
                    p_val = np.nan
                
                if not np.isnan(p_val):
                    # Apply Bonferroni correction
                    corrected_alpha = self.results['bonferroni_correction']['corrected_alpha']
                    
                    if p_val < 0.001:
                        significance = "***"
                    elif p_val < 0.01:
                        significance = "**"
                    elif p_val < corrected_alpha:
                        significance = "*"
                    else:
                        significance = "ns"
                    
                    p_value_vs_best = f"{p_val:.4f}"
            
            # Format confidence interval
            ci_str = f"[{bootstrap_ci['ci_lower']:.4f}, {bootstrap_ci['ci_upper']:.4f}]"
            
            summary_data.append({
                'Model': model,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                '95% CI': ci_str,
                'p-value vs Best': p_value_vs_best,
                'Significance': significance
            })
        
        # Sort by accuracy (descending)
        summary_df = pd.DataFrame(summary_data)
        summary_df['Accuracy_float'] = summary_df['Accuracy'].astype(float)
        summary_df = summary_df.sort_values('Accuracy_float', ascending=False)
        summary_df = summary_df.drop('Accuracy_float', axis=1)
        
        return summary_df
    
    def export_results(self, output_dir: Optional[str] = None):
        """
        Export all results to files.
        
        Args:
            output_dir: Optional custom output directory
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        
        print(f"\nExporting results to {self.output_dir}...")
        
        # 1. Export summary table (CSV and formatted text)
        summary_df = self.create_summary_table()
        summary_df.to_csv(self.output_dir / 'model_comparison_summary.csv', index=False)
        
        # Create formatted text table
        with open(self.output_dir / 'model_comparison_summary.txt', 'w', encoding='utf-8') as f:
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n")
            f.write(f"Bonferroni corrected alpha = {self.results['bonferroni_correction']['corrected_alpha']:.4f}\n")
        
        # 2. Export detailed statistical results (JSON)
        with open(self.output_dir / 'detailed_statistical_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print("✓ Results exported successfully")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Statistical Significance Testing for ML Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python model_significance_fixed.py --predictions predictions.csv --output results/
  python model_significance_fixed.py --predictions predictions.csv --cv_scores cv_scores.csv --alpha 0.01
        """
    )
    
    parser.add_argument('--predictions', required=True,
                       help='CSV file with y_true and pred_model1, pred_model2, etc.')
    parser.add_argument('--cv_scores', 
                       help='Optional CSV file with k-fold CV scores for each model')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--output', default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    try:
        # Initialize comparison framework
        comparison = ModelComparison(
            predictions_file=args.predictions,
            cv_scores_file=args.cv_scores,
            alpha=args.alpha,
            output_dir=args.output
        )
        
        # Run all statistical tests
        results = comparison.run_all_tests()
        
        # Export results
        comparison.export_results()
        
        print(f"\n🎉 Analysis complete! Results saved to: {args.output}")
        print("\nKey files generated:")
        print("  📊 model_comparison_summary.csv - Main results table")
        print("  📝 model_comparison_summary.txt - Formatted text report")
        print("  📄 detailed_statistical_results.json - Complete results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())