import numpy as np

class SVD:
    def __init__(self, n_components=None, random_state=None):
        """
        Singular Value Decomposition (SVD) implementation.

        Parameters:
        -----------
        n_components : int, optional (default=None)
            Number of singular values and vectors to compute.
            If None, compute all singular values and vectors.
        random_state : int, RandomState instance or None, optional (default=None)
            Used for random initialization.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.components_ = None
        self.U = None
        pass

    def fit(self, X):
        """
        Fit the SVD model to the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        X = self.__to_numpy_array(X)

        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
        if self.n_components is None:
            self.n_components = max_components
        elif self.n_components > max_components:
            raise ValueError(f"n_components must be <= min(n_samples, n_features)")

        try:
            covariance = np.dot(X.T, X)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)

            sorted_idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]

            self.singular_values_ = np.sqrt(eigenvalues[:self.n_components])
            self.components_ = eigenvectors[:, :self.n_components].T
            self.U = np.dot(X, self.components_.T) / self.singular_values_
            total_variance = np.sum(eigenvalues)
            self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        except np.linalg.LinAlgError:
            raise ValueError("SVD did not converge.")

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns:
        --------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X.
        """
        X = self.__to_numpy_array(X)
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(f"The number of features in X ({X.shape[1]}) does not match the number of features the model was trained on ({self.components_.shape[1]}).")
        
        # Project the data onto the right singular vectors
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed

    def inverse_transform(self, X):
        """
        Transform data back to its original space.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_components)
            New data.

        Returns:
        --------
        X_original : ndarray of shape (n_samples, n_features)
            X in original space.
        """
        X = self.__to_numpy_array(X)
        if X.shape[1] != self.components_.shape[0]:
            raise ValueError(f"The number of features in X ({X.shape[1]}) does not match the number of features the model was trained on ({self.components_.shape[0]}).")
        
        # Project the data onto the original space
        X_transformed = np.dot(X, self.components_)
        return X_transformed

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns:
        --------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X.
        """
        return self.fit(X).transform(X)

    @staticmethod
    def __to_numpy_array(data):
        """
        Convert input data to a numpy array.
        
        Parameters:
        data: Input data (list, numpy array, pandas DataFrame/Series)
        
        Returns:
        numpy.ndarray: Converted numpy array
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        elif 'pandas' in str(type(data)):
            return data.to_numpy()
        else:
            raise ValueError("Input type not recognized. Please provide a list, numpy array, or pandas DataFrame/Series.")

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(10, 5)
    
    svd = SVD(n_components=3)
    X_transformed = svd.fit_transform(X)
    
    print("Original data shape:", X.shape)
    print("Transformed data shape:", X_transformed.shape)
    print("Singular values:", svd.singular_values_)