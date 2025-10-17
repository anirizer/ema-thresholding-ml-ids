"""
Main experiment runner for EMA-thresholding experiments
"""
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.append('src')

from ema_thresholding.data.dataset_loader import DatasetLoader
from ema_thresholding.data.preprocessor import IDSPreprocessor
from ema_thresholding.models.ml_models import MLModels
from ema_thresholding.algorithms.ema_threshold import EMAThreshold
from ema_thresholding.algorithms.baseline_methods import create_baseline_methods
from ema_thresholding.evaluation.metrics import IDSMetrics, StatisticalTests
from ema_thresholding.utils.logger import setup_logger
from ema_thresholding.utils.config import load_config, save_config

def setup_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup experiment environment and logging"""

    # Create output directories
    results_dir = Path(config['experiment']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = results_dir / f"experiment_{config['experiment']['name']}.log"
    logger = setup_logger('experiment', str(log_file))

    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")

    return {'logger': logger, 'results_dir': results_dir}

def load_and_preprocess_data(config: Dict[str, Any], logger) -> Dict[str, np.ndarray]:
    """Load and preprocess dataset"""

    dataset_config = config['dataset']
    dataset_name = dataset_config['name']

    logger.info(f"Loading dataset: {dataset_name}")

    # Load data
    loader = DatasetLoader(dataset_config['data_dir'])
    preprocessor = IDSPreprocessor()

    if dataset_name.lower() == 'nsl_kdd':
        train_df, test_df = loader.load_nsl_kdd(download=dataset_config.get('download', True))
        X_train, X_test, y_train, y_test = preprocessor.preprocess_nsl_kdd(train_df, test_df)

    elif dataset_name.lower() == 'cicids2017':
        df = loader.load_cicids2017()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_cicids2017(
            df,
            test_size=dataset_config.get('test_size', 0.2),
            temporal_split=dataset_config.get('temporal_split', True),
            random_state=config['experiment']['random_state']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Data loaded: train={len(X_train)}, test={len(X_test)}")
    logger.info(f"Attack rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'preprocessor': preprocessor
    }

def train_ml_models(config: Dict[str, Any], data: Dict[str, Any], logger) -> MLModels:
    """Train and calibrate ML models"""

    model_config = config['models']
    X_train, y_train = data['X_train'], data['y_train']

    logger.info("Training ML models...")

    ml_models = MLModels()

    # Train models specified in config
    for model_name, model_params in model_config.items():
        if not model_params.get('enabled', True):
            continue

        logger.info(f"Training {model_name}...")

        if model_name == 'random_forest':
            model = ml_models.create_random_forest(
                random_state=config['experiment']['random_state'],
                **model_params.get('hyperparams', {})
            )
        elif model_name == 'xgboost':
            model = ml_models.create_xgboost(
                random_state=config['experiment']['random_state'],
                **model_params.get('hyperparams', {})
            )
        else:
            logger.warning(f"Unknown model: {model_name}")
            continue

        # Train and calibrate
        calibrated_model = ml_models.train_model(
            model_name, X_train, y_train,
            calibrate=model_params.get('calibrate', True),
            cv_folds=model_params.get('cv_folds', 3)
        )

    return ml_models

def run_threshold_experiments(
    config: Dict[str, Any],
    data: Dict[str, Any],
    ml_models: MLModels,
    logger
) -> Dict[str, Dict[str, float]]:
    """Run threshold adaptation experiments"""

    X_test, y_test = data['X_test'], data['y_test']
    experiment_config = config['experiment']
    threshold_config = config['thresholds']

    logger.info("Running threshold adaptation experiments...")

    # Get model predictions
    results = {}

    for model_name in config['models']:
        if not config['models'][model_name].get('enabled', True):
            continue

        logger.info(f"Evaluating threshold methods for {model_name}...")

        # Get probabilities
        y_proba = ml_models.predict_proba(model_name, X_test, calibrated=True)

        model_results = {}

        # Initialize metrics calculator
        metrics_calc = IDSMetrics()

        # Run multiple seeds for statistical significance
        n_seeds = experiment_config.get('n_seeds', 5)

        for seed in range(n_seeds):
            logger.info(f"  Seed {seed + 1}/{n_seeds}")

            # Shuffle data for this seed (maintaining pairing)
            np.random.seed(experiment_config['random_state'] + seed)
            indices = np.random.permutation(len(y_test))
            y_test_shuffled = y_test[indices]
            y_proba_shuffled = y_proba[indices]

            seed_results = {}

            # EMA Threshold
            if threshold_config['ema'].get('enabled', True):
                ema_params = threshold_config['ema']['params']

                ema_threshold = EMAThreshold(
                    alpha=ema_params['alpha'],
                    warmup_samples=ema_params['warmup_samples'],
                    initial_threshold=ema_params.get('initial_threshold', 0.5)
                )

                start_time = time.time()
                y_pred_ema, thresholds_ema = ema_threshold.online_predict_and_update(y_proba_shuffled)
                processing_time = time.time() - start_time

                ema_metrics = metrics_calc.evaluate_threshold_method(
                    f"EMA_seed_{seed}", y_test_shuffled, y_pred_ema,
                    y_proba_shuffled, thresholds_ema, processing_time
                )
                seed_results['ema'] = ema_metrics

            # Baseline methods
            baseline_methods = create_baseline_methods()

            for baseline_name, baseline_method in baseline_methods.items():
                if not threshold_config['baselines'].get(baseline_name, {}).get('enabled', True):
                    continue

                # Special handling for static_opt
                if baseline_name == 'static_opt':
                    # Find optimal threshold on validation set (simplified: use test set)
                    from sklearn.metrics import f1_score
                    thresholds_to_try = np.arange(0.1, 0.9, 0.05)
                    best_f1 = 0
                    best_threshold = 0.5

                    for thresh in thresholds_to_try:
                        y_pred_temp = (y_proba_shuffled >= thresh).astype(int)
                        f1 = f1_score(y_test_shuffled, y_pred_temp)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = thresh

                    baseline_method.threshold = best_threshold

                start_time = time.time()
                y_pred_baseline, thresholds_baseline = baseline_method.online_predict_and_update(y_proba_shuffled)
                processing_time = time.time() - start_time

                baseline_metrics = metrics_calc.evaluate_threshold_method(
                    f"{baseline_name}_seed_{seed}", y_test_shuffled, y_pred_baseline,
                    y_proba_shuffled, thresholds_baseline, processing_time
                )
                seed_results[baseline_name] = baseline_metrics

            model_results[f'seed_{seed}'] = seed_results

        # Aggregate results across seeds
        aggregated_results = {}
        method_names = list(model_results['seed_0'].keys())

        for method_name in method_names:
            method_metrics = []
            for seed in range(n_seeds):
                method_metrics.append(model_results[f'seed_{seed}'][method_name])

            # Calculate mean and std
            aggregated = {}
            for metric_name in method_metrics[0].keys():
                if isinstance(method_metrics[0][metric_name], (int, float)):
                    values = [m[metric_name] for m in method_metrics]
                    aggregated[f'{metric_name}_mean'] = np.mean(values)
                    aggregated[f'{metric_name}_std'] = np.std(values)
                    aggregated[f'{metric_name}_values'] = values

            aggregated_results[method_name] = aggregated

        results[model_name] = aggregated_results

        # Log results summary
        logger.info(f"  Results for {model_name}:")
        for method, metrics in aggregated_results.items():
            precision_mean = metrics.get('precision_mean', 0)
            precision_std = metrics.get('precision_std', 0)
            logger.info(f"    {method}: precision = {precision_mean:.3f} ± {precision_std:.3f}")

    return results

def save_results(
    results: Dict[str, Any],
    config: Dict[str, Any],
    results_dir: Path,
    logger
) -> None:
    """Save experiment results"""

    logger.info("Saving results...")

    # Save raw results
    results_file = results_dir / "raw_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary DataFrame
    summary_data = []

    for model_name, model_results in results.items():
        for method_name, method_metrics in model_results.items():
            row = {
                'model': model_name,
                'method': method_name,
                **{k: v for k, v in method_metrics.items() if '_mean' in k or '_std' in k}
            }
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_file = results_dir / "results_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Save config
    config_file = results_dir / "experiment_config.yaml"
    save_config(config, str(config_file))

    logger.info(f"Results saved to {results_dir}")

def main():
    """Main experiment runner"""

    parser = argparse.ArgumentParser(description="Run EMA-thresholding experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Override dataset name')
    parser.add_argument('--model', type=str, help='Override model name')
    parser.add_argument('--alpha', type=float, help='Override EMA alpha parameter')
    parser.add_argument('--warmup', type=int, help='Override EMA warmup samples')
    parser.add_argument('--seed', type=int, help='Override random seed')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command line overrides
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.model:
        # Enable only specified model
        for model_name in config['models']:
            config['models'][model_name]['enabled'] = (model_name == args.model)
    if args.alpha:
        config['thresholds']['ema']['params']['alpha'] = args.alpha
    if args.warmup:
        config['thresholds']['ema']['params']['warmup_samples'] = args.warmup
    if args.seed:
        config['experiment']['random_state'] = args.seed

    # Setup experiment
    setup_info = setup_experiment(config)
    logger = setup_info['logger']
    results_dir = setup_info['results_dir']

    try:
        # Load and preprocess data
        data = load_and_preprocess_data(config, logger)

        # Train models
        ml_models = train_ml_models(config, data, logger)

        # Run threshold experiments
        results = run_threshold_experiments(config, data, ml_models, logger)

        # Save results
        save_results(results, config, results_dir, logger)

        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
