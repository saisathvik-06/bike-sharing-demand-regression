import numpy as np

class ManualLinearRegression:
    """
    Linear Regression using the Normal Equation: theta = (X^T X)^-1 X^T y
    Standardizes features internally to improve numerical stability.
    """
    def __init__(self):
        self.theta = None
        self.mean = None
        self.std = None
        
    def fit(self, X, y):
        # 1. Standardize (Z-score normalization) to avoid numerical instability
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        
        # Avoid division by zero for constant features
        self.std[self.std == 0] = 1.0
        
        X_scaled = (X - self.mean) / self.std
        
        # 2. Add intercept term (column of ones)
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        
        # 3. Calculate Normal Equation
        # (X^T X) theta = X^T y
        try:
            A = X_b.T.dot(X_b)
            B = X_b.T.dot(y)
            self.theta = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
             print("Warning: Matrix is singular. Using Pseudo-Inverse.")
             self.theta = np.linalg.pinv(X_b).dot(y)
            
    def predict(self, X):
        if self.theta is None or self.mean is None:
            raise ValueError("Model not fitted yet.")
        
        # Standardize using training statistics
        X_scaled = (X - self.mean) / self.std
        
        # Add intercept
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return X_b.dot(self.theta)
    
    # Score and MSE (full form: Mean Squared Error) methods
    def score(self, X, y):
        """Returns R2 score"""
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def mse(self, X, y):
        """Returns Mean Squared Error"""
        y_pred = self.predict(X)
        return ((y - y_pred) ** 2).mean()

class PolynomialRegressor(ManualLinearRegression):
    """
    Extends Linear Regression with Polynomial Features (no interactions).
    Expands X to [x, x^2, ..., x^d].
    """
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        
    def _transform(self, X):
        X_poly = []
        
        # Process each feature column and generate powers
        for d in range(1, self.degree + 1):
             X_poly.append(X**d)
             
        # Concatenate horizontally
        return np.hstack(X_poly)

    def fit(self, X, y):
        X_transformed = self._transform(X)
        super().fit(X_transformed, y)
        
    def predict(self, X):
        X_transformed = self._transform(X)
        return super().predict(X_transformed)

class QuadraticInteractionRegressor(ManualLinearRegression):
    """
    Quadratic Polynomial Model where interaction terms (xi * xj) are permitted only at degree 2.
    """
    def __init__(self):
        super().__init__()
        
    def _transform(self, X):
        n_features = X.shape[1]
        # Base features (degree 1)
        features = [X]
        
        # Quadratic terms (x_i^2)
        features.append(X**2)
        
        # Interaction terms (x_i * x_j for i < j)
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                term = (X[:, i] * X[:, j]).reshape(-1, 1)
                interactions.append(term)
                
        if interactions:
            features.append(np.hstack(interactions))
            
        return np.hstack(features)

    def fit(self, X, y):
        X_transformed = self._transform(X)
        super().fit(X_transformed, y)
        
    def predict(self, X):
        X_transformed = self._transform(X)
        return super().predict(X_transformed)
