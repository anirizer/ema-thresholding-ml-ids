"""
Configuration management utilities
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Parameters:
    -----------
    config_path : str
        Path to configuration file

    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

    logger.info(f"Saved configuration to {config_path}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure

    Parameters:
    -----------
    config : dict
        Configuration to validate

    Returns:
    --------
    valid : bool
        Whether configuration is valid
    """
    required_sections = ['experiment', 'dataset', 'models', 'thresholds']

    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False

    # Validate experiment section
    exp_config = config['experiment']
    if 'name' not in exp_config or 'random_state' not in exp_config:
        logger.error("Experiment section missing required fields: name, random_state")
        return False

    # Validate dataset section
    dataset_config = config['dataset']
    if 'name' not in dataset_config:
        logger.error("Dataset section missing required field: name")
        return False

    # Validate models section
    models_config = config['models']
    if not any(models_config.get(model, {}).get('enabled', False) for model in models_config):
        logger.error("No models enabled in configuration")
        return False

    logger.info("Configuration validation passed")
    return True

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence

    Parameters:
    -----------
    base_config : dict
        Base configuration
    override_config : dict
        Override configuration

    Returns:
    --------
    merged_config : dict
        Merged configuration
    """
    import copy

    merged = copy.deepcopy(base_config)

    def recursive_merge(base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                recursive_merge(base[key], value)
            else:
                base[key] = value

    recursive_merge(merged, override_config)
    return merged
