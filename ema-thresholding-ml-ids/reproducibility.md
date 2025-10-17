# Reproducibility Guide

## Hardware Requirements
- CPU: Intel i7-12700H or equivalent
- RAM: 16GB minimum 
- Storage: 5GB for datasets and results

## Software Environment
- Python 3.8+
- Ubuntu 22.04 (tested), other Linux/macOS should work
- Required packages: see requirements.txt

## Reproducing Paper Results

### 1. Setup Environment
```bash
git clone <repository-url>
cd ema-thresholding-ml-ids
pip install -r requirements.txt
pip install -e .
```

### 2. Download Datasets
```bash
# NSL-KDD (automatic)
python experiments/scripts/download_datasets.py --dataset nsl_kdd

# CICIDS2017 (manual download required)
# Download from: https://www.kaggle.com/datasets/cicdataset/cicids2017
# Place CSV files in data/raw/
```

### 3. Run Experiments
```bash
# NSL-KDD experiments (5 seeds)
for seed in {42..46}; do
  python experiments/scripts/run_experiment.py \
    --config experiments/configs/nsl_kdd.yaml \
    --seed $seed
done

# CICIDS2017 experiments (5 seeds)  
for seed in {42..46}; do
  python experiments/scripts/run_experiment.py \
    --config experiments/configs/cicids2017.yaml \
    --seed $seed
done
```

### 4. Expected Results
- NSL-KDD Random Forest: precision 84.0±0.2% → 95.0±0.1% 
- CICIDS2017 XGBoost: precision 83.0±0.2% → 94.0±0.1%
- Processing overhead: <5% latency increase

## Notes
- Results may vary slightly due to different hardware/software versions
- Statistical significance tested with paired block bootstrap (p<0.05)
- All experiments use temporal splits for realistic evaluation
