"""
Data preprocessing for intrusion detection datasets
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class IDSPreprocessor:
    """Preprocessor for intrusion detection datasets"""

    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False

    def preprocess_nsl_kdd(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess NSL-KDD dataset

        Parameters:
        -----------
        train_df, test_df : DataFrames
            Raw NSL-KDD data

        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Preprocessed features and labels
        """
        logger.info("Preprocessing NSL-KDD dataset...")

        # Separate features and labels
        X_train = train_df.drop('class', axis=1).copy()
        y_train = train_df['class'].copy()
        X_test = test_df.drop(['class', 'difficulty'], axis=1).copy()
        y_test = test_df['class'].copy()

        # Encode categorical features
        categorical_features = ['protocol_type', 'service', 'flag']

        for col in categorical_features:
            if col not in X_train.columns:
                continue

            le = LabelEncoder()
            # Fit on combined data to handle unseen categories
            combined_values = pd.concat([X_train[col], X_test[col]])
            le.fit(combined_values.astype(str))

            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            self.label_encoders[col] = le

        # Convert labels to binary (normal vs attack)
        y_train_binary = (y_train != 'normal').astype(int)
        y_test_binary = (y_test != 'normal').astype(int)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['features'] = scaler
        self.feature_names = X_train.columns.tolist()
        self.is_fitted = True

        logger.info(f"NSL-KDD preprocessing completed:")
        logger.info(f"  Training samples: {len(X_train_scaled)} ({y_train_binary.sum()} attacks)")
        logger.info(f"  Testing samples: {len(X_test_scaled)} ({y_test_binary.sum()} attacks)")
        logger.info(f"  Features: {X_train_scaled.shape[1]}")

        return X_train_scaled, X_test_scaled, y_train_binary, y_test_binary

    def preprocess_cicids2017(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        temporal_split: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess CICIDS2017 dataset

        Parameters:
        -----------
        df : DataFrame
            Raw CICIDS2017 data
        test_size : float
            Fraction for test set
        temporal_split : bool
            Whether to use temporal split (last days for test)
        random_state : int
            Random seed

        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Preprocessed features and labels
        """
        logger.info("Preprocessing CICIDS2017 dataset...")

        # Clean data
        df = self._clean_cicids2017(df)

        # Separate features and labels
        label_col = 'Label' if 'Label' in df.columns else ' Label'
        X = df.drop(label_col, axis=1)
        y = df[label_col]

        # Convert labels to binary
        y_binary = (y.str.upper() != 'BENIGN').astype(int)

        # Remove non-numeric columns and handle infinite values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Split data
        if temporal_split:
            # Use timestamp if available, otherwise use row order as proxy
            split_point = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y_binary.iloc[:split_point], y_binary.iloc[split_point:]
            logger.info("Using temporal split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
            )
            logger.info("Using random split")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['features'] = scaler
        self.feature_names = X_train.columns.tolist()
        self.is_fitted = True

        logger.info(f"CICIDS2017 preprocessing completed:")
        logger.info(f"  Training samples: {len(X_train_scaled)} ({y_train.sum()} attacks)")
        logger.info(f"  Testing samples: {len(X_test_scaled)} ({y_test.sum()} attacks)")
        logger.info(f"  Features: {X_train_scaled.shape[1]}")

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    def _clean_cicids2017(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CICIDS2017 dataset"""
        logger.info(f"Cleaning CICIDS2017: {len(df)} samples")

        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_size - len(df)} duplicates")

        # Fix label column name
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if label_cols:
            df = df.rename(columns={label_cols[0]: 'Label'})

        # Remove rows with missing labels
        if 'Label' in df.columns:
            df = df.dropna(subset=['Label'])

        # Clean whitespace in labels
        if 'Label' in df.columns and df['Label'].dtype == 'object':
            df['Label'] = df['Label'].str.strip()

        logger.info(f"After cleaning: {len(df)} samples")
        return df

    def create_splits(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Create train/val/test splits

        Parameters:
        -----------
        X, y : arrays
            Features and labels
        test_size : float
            Test set fraction
        val_size : float  
            Validation set fraction (from remaining data)
        random_state : int
            Random seed

        Returns:
        --------
        splits : dict
            Dictionary with train/val/test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )

        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

        logger.info(f"Created splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return splits

    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet")
        return self.feature_names.copy()

    def save_scalers(self, filepath: str) -> None:
        """Save fitted scalers"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }, f)
        logger.info(f"Scalers saved to {filepath}")

    def load_scalers(self, filepath: str) -> None:
        """Load fitted scalers"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.label_encoders = data['label_encoders']
            self.feature_names = data['feature_names']
            self.is_fitted = True
        logger.info(f"Scalers loaded from {filepath}")

def preprocess_dataset(
    dataset_name: str,
    data_dir: str = "data/raw",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to preprocess datasets

    Parameters:
    -----------
    dataset_name : str
        'nsl_kdd' or 'cicids2017'
    data_dir : str
        Data directory path
    **kwargs : dict
        Additional preprocessing arguments

    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Preprocessed data splits
    """
    from .dataset_loader import DatasetLoader

    loader = DatasetLoader(data_dir)
    preprocessor = IDSPreprocessor()

    if dataset_name.lower() == 'nsl_kdd':
        train_df, test_df = loader.load_nsl_kdd()
        return preprocessor.preprocess_nsl_kdd(train_df, test_df)

    elif dataset_name.lower() == 'cicids2017':
        df = loader.load_cicids2017()
        return preprocessor.preprocess_cicids2017(df, **kwargs)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
