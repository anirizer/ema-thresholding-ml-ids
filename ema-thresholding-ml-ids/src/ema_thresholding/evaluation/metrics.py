"""
Evaluation metrics and statistical tests for threshold adaptation experiments
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IDSMetrics:
    """Metrics calculator for intrusion detection systems"""

    def __init__(self):
        self.metrics_history = []

    def calculate_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics

        Parameters:
        -----------
        y_true : array
            True labels
        y_pred : array  
            Predicted labels
        y_proba : array, optional
            Predicted probabilities

        Returns:
        --------
        metrics : dict
            Dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        # Metrics that require probabilities
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics.update({
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })

        return metrics

    def calculate_alert_rate(self, y_pred: np.ndarray, per_samples: int = 1000) -> float:
        """
        Calculate alert rate (alerts per X samples)

        Parameters:
        -----------
        y_pred : array
            Predicted labels
        per_samples : int
            Number of samples to normalize by

        Returns:
        --------
        alert_rate : float
            Number of alerts per per_samples
        """
        alert_count = np.sum(y_pred)
        total_samples = len(y_pred)
        return (alert_count / total_samples) * per_samples

    def evaluate_threshold_method(
        self,
        method_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        thresholds: Optional[np.ndarray] = None,
        processing_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a threshold adaptation method

        Parameters:
        -----------
        method_name : str
            Name of the method
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        y_proba : array, optional
            Predicted probabilities
        thresholds : array, optional
            Threshold values used
        processing_time : float, optional
            Processing time in seconds

        Returns:
        --------
        results : dict
            Complete evaluation results
        """
        # Basic metrics
        metrics = self.calculate_basic_metrics(y_true, y_pred, y_proba)

        # Alert rate
        metrics['alert_rate_per_1k'] = self.calculate_alert_rate(y_pred, 1000)

        # Threshold statistics
        if thresholds is not None:
            metrics.update({
                'threshold_mean': np.mean(thresholds),
                'threshold_std': np.std(thresholds),
                'threshold_min': np.min(thresholds),
                'threshold_max': np.max(thresholds)
            })

        # Processing time
        if processing_time is not None:
            metrics['processing_time_ms'] = processing_time * 1000
            metrics['latency_ms_per_sample'] = (processing_time / len(y_pred)) * 1000

        # Store for later analysis
        result = {
            'method': method_name,
            'timestamp': pd.Timestamp.now(),
            **metrics
        }
        self.metrics_history.append(result)

        return metrics

    def compare_methods(
        self,
        results: Dict[str, Dict[str, float]],
        primary_metric: str = 'f1'
    ) -> pd.DataFrame:
        """
        Compare multiple threshold methods

        Parameters:
        -----------
        results : dict
            Dictionary mapping method names to their metrics
        primary_metric : str
            Primary metric for ranking

        Returns:
        --------
        comparison : DataFrame
            Comparison table sorted by primary metric
        """
        df = pd.DataFrame(results).T

        # Sort by primary metric (descending)
        df = df.sort_values(primary_metric, ascending=False)

        logger.info(f"Method comparison (ranked by {primary_metric}):")
        for idx, (method, row) in enumerate(df.iterrows(), 1):
            logger.info(f"  {idx}. {method}: {row[primary_metric]:.4f}")

        return df

    def get_metrics_history(self) -> pd.DataFrame:
        """Get history of all calculated metrics"""
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame(self.metrics_history)

class StatisticalTests:
    """Statistical tests for comparing threshold adaptation methods"""

    @staticmethod
    def paired_bootstrap_test(
        metric1: np.ndarray,
        metric2: np.ndarray,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Paired bootstrap test for comparing two methods

        Parameters:
        -----------
        metric1, metric2 : arrays
            Metrics from two methods (paired samples)
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)

        Returns:
        --------
        test_results : dict
            Statistical test results
        """
        if len(metric1) != len(metric2):
            raise ValueError("Metrics arrays must have same length")

        # Calculate observed difference
        observed_diff = np.mean(metric1) - np.mean(metric2)

        # Bootstrap resampling
        n_samples = len(metric1)
        bootstrap_diffs = []

        for _ in range(n_bootstrap):
            # Resample indices
            indices = np.random.choice(n_samples, n_samples, replace=True)

            # Calculate difference for bootstrap sample
            boot_diff = np.mean(metric1[indices]) - np.mean(metric2[indices])
            bootstrap_diffs.append(boot_diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

        # P-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= 0),
            np.mean(bootstrap_diffs <= 0)
        )

        return {
            'observed_difference': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'significant': p_value < (1 - confidence_level),
            'bootstrap_samples': n_bootstrap,
            'confidence_level': confidence_level
        }

    @staticmethod
    def wilcoxon_signed_rank_test(
        metric1: np.ndarray,
        metric2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for paired samples

        Parameters:
        -----------
        metric1, metric2 : arrays
            Metrics from two methods (paired samples)
        alternative : str
            'two-sided', 'greater', or 'less'

        Returns:
        --------
        test_results : dict
            Statistical test results
        """
        if len(metric1) != len(metric2):
            raise ValueError("Metrics arrays must have same length")

        # Perform test
        statistic, p_value = stats.wilcoxon(
            metric1, metric2, alternative=alternative, zero_method='pratt'
        )

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'alternative': alternative,
            'effect_size': np.median(metric1 - metric2)  # Hodges-Lehmann estimator
        }

    @staticmethod
    def multiple_comparisons_correction(
        p_values: List[float],
        method: str = 'bonferroni',
        alpha: float = 0.05
    ) -> Dict[str, List[float]]:
        """
        Multiple comparisons correction

        Parameters:
        -----------
        p_values : list
            List of p-values
        method : str
            Correction method ('bonferroni', 'holm', 'benjamini_hochberg')
        alpha : float
            Family-wise error rate

        Returns:
        --------
        corrected_results : dict
            Corrected p-values and significance decisions
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)

        if method == 'bonferroni':
            corrected_p = p_values * n_tests
            corrected_p = np.clip(corrected_p, 0, 1)

        elif method == 'holm':
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * (n_tests - i)

            # Ensure monotonicity
            sorted_corrected = corrected_p[sorted_indices]
            for i in range(1, n_tests):
                if sorted_corrected[i] < sorted_corrected[i-1]:
                    sorted_corrected[i] = sorted_corrected[i-1]

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = sorted_corrected[i]

            corrected_p = np.clip(corrected_p, 0, 1)

        elif method == 'benjamini_hochberg':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * n_tests / (i + 1)

            # Ensure monotonicity (reverse)
            sorted_corrected = corrected_p[sorted_indices]
            for i in range(n_tests - 2, -1, -1):
                if sorted_corrected[i] > sorted_corrected[i+1]:
                    sorted_corrected[i] = sorted_corrected[i+1]

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = sorted_corrected[i]

            corrected_p = np.clip(corrected_p, 0, 1)

        else:
            raise ValueError(f"Unknown correction method: {method}")

        return {
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected_p.tolist(),
            'significant': (corrected_p < alpha).tolist(),
            'method': method,
            'alpha': alpha
        }

def run_statistical_comparison(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    primary_metric: str = 'f1',
    n_bootstrap: int = 10000
) -> Dict[str, Dict[str, float]]:
    """
    Run comprehensive statistical comparison of methods

    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to arrays of metrics
    primary_metric : str
        Primary metric to compare
    n_bootstrap : int
        Number of bootstrap samples

    Returns:
    --------
    comparison_results : dict
        Statistical comparison results
    """
    methods = list(results_dict.keys())
    comparison_results = {}

    # Pairwise comparisons
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            metric1 = results_dict[method1][primary_metric]
            metric2 = results_dict[method2][primary_metric]

            # Bootstrap test
            bootstrap_result = StatisticalTests.paired_bootstrap_test(
                metric1, metric2, n_bootstrap=n_bootstrap
            )

            # Wilcoxon test
            wilcoxon_result = StatisticalTests.wilcoxon_signed_rank_test(
                metric1, metric2
            )

            comparison_key = f"{method1}_vs_{method2}"
            comparison_results[comparison_key] = {
                **bootstrap_result,
                'wilcoxon_p_value': wilcoxon_result['p_value'],
                'wilcoxon_significant': wilcoxon_result['significant']
            }

    return comparison_results
