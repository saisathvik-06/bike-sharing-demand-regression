import pandas as pd
import numpy as np

class BikeSharingDataset:
    """
    Handles data loading, feature engineering, and splitting for Bike Sharing Demand.
    """
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = None

    def load_and_preprocess(self):
        """
        Loads data, extracts features, drops forbidden columns, and splits.
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # 1. Feature Engineering: Datetime
        # Convert datetime to datetime object
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract components
        # Note: We treat these as categorical mostly, or ordinal. 
        # Ordinal means we can keep as integers & Categorial means one-hot encode.
        # One hot encoding means creating binary columns for each category.
        # For linear regression, one-hot encoding categorical variables like season/weather is often better.
        # But 'hour', 'year', 'month' contain trend info, i.e. they have an inherent order and cyclical nature.
        
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['weekday'] = df['datetime'].dt.weekday  # 0=Monday, ...
        
        # 2. Drop leakage columns and original datetime
        # 'casual' and 'registered' sum up to 'count', so they are target leakage.
        columns_to_drop = ['datetime', 'casual', 'registered']
        df_clean = df.drop(columns=columns_to_drop, axis=1)
        
        # 3. Handle Categorical Variables via One-Hot Encoding
        # Nominal: season, weather, holiday, workingday (already binary)
        # We should likely One-Hot Encode 'season' (1-4) and 'weather' (1-4).
        # We can also cycle-encode hour/month, but for simplicity, we keep them as is.
        
        categorical_cols = ['season', 'weather']
        df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        # Update: Ensure boolean columns are integers (0/1) for matrix math
        df_clean = df_clean.astype(float)
        
        # 4. Separate Target
        y = df_clean['count'].values
        X = df_clean.drop('count', axis=1).values
        self.feature_names = df_clean.drop('count', axis=1).columns.tolist()
        
        # 5. Split Data (Custom Implementation)
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)
        
        return X_train, X_test, y_train, y_test

    def _train_test_split(self, X, y):
        """
        Custom random split implementation.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        test_samples = int(n_samples * self.test_size)
        
        test_indices = indices[:test_samples]
        train_indices = indices[test_samples:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test

    def get_feature_names(self):
        return self.feature_names
