# EMA-thresholding for ML-IDS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Lightweight Online Adaptation of Decision Thresholds for Machine Learning-based Intrusion Detection Systems**

This repository contains the implementation of the EMA-thresholding algorithm described in our paper:

> *EMA-thresholding for ML-IDS: Lightweight Online Adaptation of Decision Thresholds*  
> Authors: Amir Askhat et al.

## 🔍 Overview

Static thresholds in ML-IDS often lead to excessive false positives when network traffic changes. We present a lightweight online threshold adaptation algorithm based on Exponential Moving Average (EMA) of predicted probabilities.

**Key Features:**
- **O(1) memory complexity** - only current threshold state needed
- **Model-agnostic** - works with any ML classifier outputting probabilities  
- **Streaming-ready** - easy integration into real-time systems
- **Simple update rule**: `θᵢ = α·θᵢ₋₁ + (1-α)·pᵢ`

**Results:** ~12-14% precision improvement on NSL-KDD and CICIDS2017 with <5% processing overhead.

## 📊 Algorithm

The EMA-thresholding algorithm works as follows:

1. **Initialization**: Set initial threshold θ₀ (typically 0.5)
2. **For each incoming sample i**:
   - Make decision using current threshold θᵢ₋₁  
   - Update threshold: θᵢ = α·θᵢ₋₁ + (1-α)·pᵢ
   - Where α ∈ [0,1] is the EMA smoothing parameter

**Recommended parameters:**
- α = 0.8 (balances responsiveness vs stability)
- Warmup period = 100 samples
- Threshold bounds = [0.1, 0.9]

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ema-thresholding-ml-ids.git
cd ema-thresholding-ml-ids

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from ema_thresholding.algorithms.ema_threshold import EMAThreshold
import numpy as np

# Initialize EMA threshold adapter
ema = EMAThreshold(alpha=0.8, warmup_samples=100)

# Example: online prediction and threshold adaptation
probabilities = np.array([0.3, 0.7, 0.2, 0.9, 0.4])
predictions, thresholds = ema.online_predict_and_update(probabilities)

print(f"Predictions: {predictions}")
print(f"Thresholds: {thresholds}")
```

### Running Experiments

1. **Download datasets** (NSL-KDD downloads automatically):
```bash
python experiments/scripts/download_datasets.py --dataset nsl_kdd
# For CICIDS2017, manual download from Kaggle required
```

2. **Prepare data splits**:
```bash  
python experiments/scripts/prepare_splits.py --dataset nsl_kdd --temporal
```

3. **Run experiments**:
```bash
python experiments/scripts/run_experiment.py \
    --config experiments/configs/nsl_kdd.yaml \
    --model random_forest \
    --alpha 0.8 \
    --warmup 100 \
    --seed 42
```

4. **Analyze results**:
```bash
python experiments/scripts/analyze_results.py --results-dir results/
```

## 📁 Project Structure

```
ema-thresholding-ml-ids/
├── src/ema_thresholding/           # Main package
│   ├── algorithms/                 # Threshold adaptation algorithms
│   │   ├── ema_threshold.py       # Main EMA algorithm
│   │   └── baseline_methods.py    # Baseline comparisons
│   ├── data/                      # Data loading and preprocessing
│   ├── models/                    # ML models (RF, XGBoost)
│   ├── evaluation/               # Metrics and statistical tests
│   └── utils/                    # Configuration and logging
├── experiments/                   # Experiment scripts and configs
│   ├── scripts/                  # Executable scripts
│   └── configs/                  # YAML configuration files
├── data/                         # Dataset storage
│   ├── raw/                     # Raw downloaded datasets
│   └── processed/               # Preprocessed data
├── results/                      # Experiment results
├── tests/                       # Unit tests
└── README.md
```

## 🔧 Configuration

Experiments are configured via YAML files. Example `nsl_kdd.yaml`:

```yaml
experiment:
  name: "ema_nsl_kdd_experiment"
  random_state: 42
  n_seeds: 5

dataset:
  name: "nsl_kdd"
  data_dir: "data/raw"
  download: true

models:
  random_forest:
    enabled: true
    calibrate: true
    hyperparams:
      n_estimators: 100
      max_depth: 20

thresholds:
  ema:
    enabled: true
    params:
      alpha: 0.8
      warmup_samples: 100
  baselines:
    static_0.5:
      enabled: true
    static_opt:
      enabled: true
```

## 📈 Baseline Methods

The repository includes implementations of several baseline threshold adaptation methods:

- **Static(0.5)**: Fixed threshold at 0.5
- **Static(opt)**: Optimal static threshold from validation set
- **SlidingMean(w)**: Sliding window mean of probabilities  
- **EWMA-logits**: EMA applied to logits instead of probabilities
- **FPR-control**: Simple FPR-based threshold control

## 📊 Evaluation Metrics

**Primary metrics:**
- Precision, Recall, F1-score
- PR-AUC (Area Under Precision-Recall Curve)
- Alert rate (alerts per 1000 samples)
- Processing latency (ms/sample)

**Statistical testing:**
- Paired block bootstrap (10,000 samples)
- Multiple runs with different seeds
- Confidence intervals and p-values

## 🔬 Experimental Results

**NSL-KDD Dataset:**
- Random Forest: 84.0±0.2% → 95.0±0.1% precision (+13.1%)
- XGBoost: Similar improvements

**CICIDS2017 Dataset:**  
- Random Forest: 83.0±0.2% → 94.0±0.1% precision (+13.3%)
- Processing overhead: <5% latency increase

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{askhat2024ema,
  title={EMA-thresholding for ML-IDS: Lightweight Online Adaptation of Decision Thresholds},
  author={Askhat, Amir and others},
  journal={Conference Proceedings},
  year={2024}
}
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
flake8 src/ tests/
```

### Adding New Methods

1. Implement your method in `src/ema_thresholding/algorithms/`
2. Add configuration in experiment YAML files
3. Update `run_experiment.py` to include your method
4. Add unit tests in `tests/`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Development guidelines:**
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Use meaningful commit messages

## 🐛 Issues

If you encounter any problems or have questions, please open an issue on GitHub with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment details (Python version, OS, etc.)

## 🙏 Acknowledgments

- NSL-KDD dataset: University of New Brunswick
- CICIDS2017 dataset: Canadian Institute for Cybersecurity
- Scikit-learn and XGBoost communities
- Contributors and reviewers

---

**Project Status**: Active development  
**Maintenance**: This project is actively maintained  
**Support**: Community-driven support via GitHub issues
