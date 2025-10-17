"""
Dataset loader for NSL-KDD and CICIDS2017 datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io
import logging
from typing import Tuple, Optional, Union
from urllib.parse import urlparse
import os

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Universal dataset loader for intrusion detection datasets"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Dataset URLs and info
        self.dataset_info = {
            'nsl_kdd': {
                'train_url': 'https://github.com/jmnwong/NSL-KDD-Dataset/raw/master/KDDTrain%2B.txt',
                'test_url': 'https://github.com/jmnwong/NSL-KDD-Dataset/raw/master/KDDTest%2B.txt',
                'train_file': 'KDDTrain+.txt',
                'test_file': 'KDDTest+.txt',
                'feature_names': self._get_nsl_kdd_features(),
                'label_col': 'class',
                'attack_types': ['normal', 'dos', 'probe', 'r2l', 'u2r']
            },
            'cicids2017': {
                'description': 'CICIDS2017 dataset - manual download required',
                'kaggle_url': 'https://www.kaggle.com/datasets/cicdataset/cicids2017',
                'files': [
                    'Monday-WorkingHours.pcap_ISCX.csv',
                    'Tuesday-WorkingHours.pcap_ISCX.csv',
                    'Wednesday-workingHours.pcap_ISCX.csv',
                    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
                ],
                'label_col': 'Label'
            }
        }

    def _get_nsl_kdd_features(self) -> list:
        """NSL-KDD feature names"""
        return [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
        ]

    def download_file(self, url: str, filename: str) -> bool:
        """Download file from URL"""
        filepath = self.data_dir / filename

        if filepath.exists():
            logger.info(f"File {filename} already exists, skipping download")
            return True

        try:
            logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Successfully downloaded {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False

    def load_nsl_kdd(self, download: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load NSL-KDD dataset

        Parameters:
        -----------
        download : bool, default=True
            Whether to download dataset if not present

        Returns:
        --------
        train_df, test_df : tuple of DataFrames
            Training and testing datasets
        """
        info = self.dataset_info['nsl_kdd']
        train_file = self.data_dir / info['train_file']
        test_file = self.data_dir / info['test_file']

        # Download if needed
        if download and not train_file.exists():
            self.download_file(info['train_url'], info['train_file'])
        if download and not test_file.exists():
            self.download_file(info['test_url'], info['test_file'])

        # Load data
        try:
            train_df = pd.read_csv(train_file, names=info['feature_names'][:-1])  # Exclude difficulty
            test_df = pd.read_csv(test_file, names=info['feature_names'])

            logger.info(f"Loaded NSL-KDD: train={len(train_df)}, test={len(test_df)}")
            return train_df, test_df

        except Exception as e:
            logger.error(f"Failed to load NSL-KDD: {e}")
            raise

    def load_cicids2017(self) -> pd.DataFrame:
        """
        Load CICIDS2017 dataset

        Note: Files must be manually downloaded from Kaggle or official source
        and placed in data/raw directory

        Returns:
        --------
        df : DataFrame
            Combined CICIDS2017 dataset
        """
        info = self.dataset_info['cicids2017']
        dataframes = []

        logger.info("Loading CICIDS2017 dataset...")

        for filename in info['files']:
            filepath = self.data_dir / filename

            if not filepath.exists():
                logger.warning(f"File {filename} not found. Please download manually from:")
                logger.warning(f"{info['kaggle_url']}")
                continue

            try:
                df = pd.read_csv(filepath)
                # Clean column names
                df.columns = df.columns.str.strip()
                dataframes.append(df)
                logger.info(f"Loaded {filename}: {len(df)} samples")

            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

        if not dataframes:
            raise FileNotFoundError(
                "No CICIDS2017 files found. Please download them manually and place in data/raw/"
            )

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined CICIDS2017 dataset: {len(combined_df)} samples")

        return combined_df

    def load_dataset(self, dataset_name: str, **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generic dataset loader

        Parameters:
        -----------
        dataset_name : str
            Name of dataset ('nsl_kdd' or 'cicids2017')
        **kwargs : dict
            Additional arguments for specific loaders

        Returns:
        --------
        data : DataFrame or tuple of DataFrames
            Loaded dataset(s)
        """
        dataset_name = dataset_name.lower().replace('-', '_')

        if dataset_name == 'nsl_kdd':
            return self.load_nsl_kdd(**kwargs)
        elif dataset_name == 'cicids2017':
            return self.load_cicids2017(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_dataset_info(self, dataset_name: str) -> dict:
        """Get information about a dataset"""
        dataset_name = dataset_name.lower().replace('-', '_')
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.dataset_info[dataset_name]

    def list_available_files(self) -> list:
        """List all files in data directory"""
        return [f.name for f in self.data_dir.iterdir() if f.is_file()]

# Convenience functions
def load_nsl_kdd(data_dir: str = "data/raw", download: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to load NSL-KDD"""
    loader = DatasetLoader(data_dir)
    return loader.load_nsl_kdd(download=download)

def load_cicids2017(data_dir: str = "data/raw") -> pd.DataFrame:
    """Convenience function to load CICIDS2017"""
    loader = DatasetLoader(data_dir)
    return loader.load_cicids2017()

def download_nsl_kdd(data_dir: str = "data/raw") -> bool:
    """Download NSL-KDD dataset"""
    loader = DatasetLoader(data_dir)
    info = loader.dataset_info['nsl_kdd']

    success_train = loader.download_file(info['train_url'], info['train_file'])
    success_test = loader.download_file(info['test_url'], info['test_file'])

    return success_train and success_test
