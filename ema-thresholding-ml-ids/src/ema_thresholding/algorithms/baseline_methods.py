"""
Baseline threshold adaptation methods for comparison
"""
import numpy as np
from typing import Union, Optional, Tuple
from collections import deque
import logging

class StaticThreshold:
    """Static threshold baseline"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, probabilities: Union[np.ndarray, float]) -> Union[np.ndarray, int]:
        if isinstance(probabilities, (int, float)):
            return int(probabilities >= self.threshold)
        return (np.array(probabilities) >= self.threshold).astype(int)

    def update(self, probability: float) -> float:
        return self.threshold  # No update for static

    def online_predict_and_update(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.predict(probabilities)
        thresholds = np.full(len(probabilities), self.threshold)
        return predictions, thresholds

    def reset(self):
        pass  # Nothing to reset

class SlidingMeanThreshold:
    """Sliding window mean threshold"""

    def __init__(self, window_size: int = 50, initial_threshold: float = 0.5):
        self.window_size = window_size
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.window = deque(maxlen=window_size)

    def predict(self, probabilities: Union[np.ndarray, float]) -> Union[np.ndarray, int]:
        if isinstance(probabilities, (int, float)):
            return int(probabilities >= self.current_threshold)
        return (np.array(probabilities) >= self.current_threshold).astype(int)

    def update(self, probability: float) -> float:
        self.window.append(probability)
        if len(self.window) >= self.window_size:
            self.current_threshold = np.mean(self.window)
        return self.current_threshold

    def online_predict_and_update(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(probabilities)
        predictions = np.zeros(n_samples, dtype=int)
        thresholds = np.zeros(n_samples)

        for i in range(n_samples):
            predictions[i] = self.predict(probabilities[i])
            thresholds[i] = self.current_threshold
            self.update(probabilities[i])

        return predictions, thresholds

    def reset(self):
        self.current_threshold = self.initial_threshold
        self.window.clear()

class EWMALogitsThreshold:
    """EWMA on logits (before sigmoid) threshold"""

    def __init__(self, alpha: float = 0.8, initial_threshold: float = 0.5):
        self.alpha = alpha
        self.initial_threshold = initial_threshold
        self.current_logit = np.log(initial_threshold / (1 - initial_threshold))

    def predict(self, probabilities: Union[np.ndarray, float]) -> Union[np.ndarray, int]:
        current_prob = 1 / (1 + np.exp(-self.current_logit))
        if isinstance(probabilities, (int, float)):
            return int(probabilities >= current_prob)
        return (np.array(probabilities) >= current_prob).astype(int)

    def update(self, probability: float) -> float:
        # Convert probability to logit
        prob_clipped = np.clip(probability, 1e-7, 1-1e-7)
        logit = np.log(prob_clipped / (1 - prob_clipped))

        # EMA update on logits
        self.current_logit = self.alpha * self.current_logit + (1 - self.alpha) * logit

        # Convert back to probability
        return 1 / (1 + np.exp(-self.current_logit))

    def online_predict_and_update(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(probabilities)
        predictions = np.zeros(n_samples, dtype=int)
        thresholds = np.zeros(n_samples)

        for i in range(n_samples):
            current_prob = 1 / (1 + np.exp(-self.current_logit))
            predictions[i] = int(probabilities[i] >= current_prob)
            thresholds[i] = current_prob
            self.update(probabilities[i])

        return predictions, thresholds

    def reset(self):
        self.current_logit = np.log(self.initial_threshold / (1 - self.initial_threshold))

class FPRControlThreshold:
    """Simple FPR control threshold (simplified version)"""

    def __init__(self, target_fpr: float = 0.05, window_size: int = 100):
        self.target_fpr = target_fpr
        self.window_size = window_size
        self.current_threshold = 0.5
        self.predictions_window = deque(maxlen=window_size)
        self.labels_window = deque(maxlen=window_size)  # Would need true labels in practice

    def predict(self, probabilities: Union[np.ndarray, float]) -> Union[np.ndarray, int]:
        if isinstance(probabilities, (int, float)):
            return int(probabilities >= self.current_threshold)
        return (np.array(probabilities) >= self.current_threshold).astype(int)

    def update(self, probability: float, true_label: Optional[int] = None) -> float:
        prediction = int(probability >= self.current_threshold)

        # In practice, would need true labels for FPR control
        # This is a simplified version
        if true_label is not None:
            self.predictions_window.append(prediction)
            self.labels_window.append(true_label)

            if len(self.predictions_window) >= self.window_size:
                # Calculate current FPR
                fp = sum((p == 1 and l == 0) for p, l in zip(self.predictions_window, self.labels_window))
                tn = sum((p == 0 and l == 0) for p, l in zip(self.predictions_window, self.labels_window))

                if (fp + tn) > 0:
                    current_fpr = fp / (fp + tn)

                    # Adjust threshold based on FPR
                    if current_fpr > self.target_fpr:
                        self.current_threshold += 0.01  # Increase threshold to reduce FPR
                    elif current_fpr < self.target_fpr * 0.8:
                        self.current_threshold -= 0.01  # Decrease threshold

                    self.current_threshold = np.clip(self.current_threshold, 0.1, 0.9)

        return self.current_threshold

    def online_predict_and_update(self, probabilities: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(probabilities)
        predictions = np.zeros(n_samples, dtype=int)
        thresholds = np.zeros(n_samples)

        for i in range(n_samples):
            predictions[i] = self.predict(probabilities[i])
            thresholds[i] = self.current_threshold
            label = true_labels[i] if true_labels is not None else None
            self.update(probabilities[i], label)

        return predictions, thresholds

    def reset(self):
        self.current_threshold = 0.5
        self.predictions_window.clear()
        self.labels_window.clear()

# Factory functions for easy baseline creation
def create_baseline_methods():
    """Create all baseline methods with default parameters"""
    return {
        'static_0.5': StaticThreshold(0.5),
        'static_opt': StaticThreshold(0.5),  # Will be set to optimal threshold during evaluation  
        'sliding_mean_50': SlidingMeanThreshold(50),
        'ewma_logits_0.8': EWMALogitsThreshold(0.8),
        'fpr_control_0.05': FPRControlThreshold(0.05),
    }

def get_baseline_names():
    """Get list of all baseline method names"""
    return ['static_0.5', 'static_opt', 'sliding_mean_50', 'ewma_logits_0.8', 'fpr_control_0.05']
