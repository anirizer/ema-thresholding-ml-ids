"""
EMA-thresholding algorithm for ML-IDS
Lightweight online adaptation of decision thresholds using exponential weighted moving average
"""
import numpy as np
from typing import Union, Optional, Tuple
import logging

class EMAThreshold:
    """
    EMA-based adaptive threshold for ML-IDS

    Implementation of the lightweight online threshold adaptation algorithm
    that uses exponential weighted moving average of predicted probabilities.

    Parameters:
    -----------
    alpha : float, default=0.8
        EMA smoothing parameter (0 < alpha < 1)
        Higher values = more weight to recent predictions

    initial_threshold : float, default=0.5
        Initial threshold value

    warmup_samples : int, default=100
        Number of samples to use for warmup period

    min_threshold : float, default=0.1
        Minimum allowed threshold value

    max_threshold : float, default=0.9
        Maximum allowed threshold value
    """

    def __init__(
        self, 
        alpha: float = 0.8,
        initial_threshold: float = 0.5,
        warmup_samples: int = 100,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9
    ):
        self.alpha = alpha
        self.initial_threshold = initial_threshold
        self.warmup_samples = warmup_samples
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Internal state
        self.current_threshold = initial_threshold
        self.sample_count = 0
        self.is_warmed_up = False

        # Logging
        self.logger = logging.getLogger(__name__)

    def predict(self, probabilities: Union[np.ndarray, float]) -> Union[np.ndarray, int]:
        """
        Make predictions using current threshold

        Parameters:
        -----------
        probabilities : array-like or float
            Predicted probabilities from ML model

        Returns:
        --------
        predictions : array-like or int
            Binary predictions (0 = benign, 1 = attack)
        """
        if isinstance(probabilities, (int, float)):
            return int(probabilities >= self.current_threshold)

        return (np.array(probabilities) >= self.current_threshold).astype(int)

    def update(self, probability: float) -> float:
        """
        Update threshold using EMA and return current threshold

        Parameters:
        -----------
        probability : float
            Single predicted probability to update threshold with

        Returns:
        --------
        threshold : float
            Current threshold after update
        """
        self.sample_count += 1

        # Warmup period - don't update threshold
        if self.sample_count <= self.warmup_samples:
            if self.sample_count == self.warmup_samples:
                self.is_warmed_up = True
                self.logger.info(f"EMA warmup completed after {self.warmup_samples} samples")
            return self.current_threshold

        # EMA update: θ_i = α * θ_{i-1} + (1-α) * p_i
        old_threshold = self.current_threshold
        self.current_threshold = self.alpha * self.current_threshold + (1 - self.alpha) * probability

        # Clamp threshold to valid range
        self.current_threshold = np.clip(self.current_threshold, self.min_threshold, self.max_threshold)

        # Log significant threshold changes
        if abs(self.current_threshold - old_threshold) > 0.05:
            self.logger.debug(
                f"Threshold change: {old_threshold:.3f} -> {self.current_threshold:.3f} "
                f"(sample {self.sample_count}, p={probability:.3f})"
            )

        return self.current_threshold

    def online_predict_and_update(
        self, 
        probabilities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Online prediction with threshold updates

        For each sample i:
        1. Make decision using threshold_{i-1}
        2. Update threshold using probability_i

        Parameters:
        -----------
        probabilities : np.ndarray
            Array of predicted probabilities

        Returns:
        --------
        predictions : np.ndarray
            Binary predictions
        thresholds : np.ndarray
            Threshold values used for each prediction
        """
        n_samples = len(probabilities)
        predictions = np.zeros(n_samples, dtype=int)
        thresholds = np.zeros(n_samples)

        for i in range(n_samples):
            # Step 1: Predict using current threshold
            predictions[i] = self.predict(probabilities[i])
            thresholds[i] = self.current_threshold

            # Step 2: Update threshold with current probability
            self.update(probabilities[i])

        return predictions, thresholds

    def reset(self) -> None:
        """Reset the algorithm to initial state"""
        self.current_threshold = self.initial_threshold
        self.sample_count = 0
        self.is_warmed_up = False
        self.logger.info("EMA threshold algorithm reset to initial state")

    def get_state(self) -> dict:
        """Get current algorithm state for logging/debugging"""
        return {
            'current_threshold': self.current_threshold,
            'sample_count': self.sample_count,
            'is_warmed_up': self.is_warmed_up,
            'alpha': self.alpha
        }

def create_ema_threshold(
    alpha: float = 0.8,
    warmup: int = 100,
    **kwargs
) -> EMAThreshold:
    """
    Factory function to create EMA threshold with common configurations

    Parameters:
    -----------
    alpha : float
        EMA parameter (recommended: 0.7, 0.8, 0.9)
    warmup : int
        Warmup period (recommended: 50-500)
    **kwargs : dict
        Additional parameters for EMAThreshold

    Returns:
    --------
    ema_threshold : EMAThreshold
        Configured EMA threshold instance
    """
    return EMAThreshold(
        alpha=alpha,
        warmup_samples=warmup,
        **kwargs
    )
