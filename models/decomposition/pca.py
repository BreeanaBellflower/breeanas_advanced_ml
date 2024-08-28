import numpy as np
from models.decomposition.svd import SVD

class PCA:
    def __init__(self, n_components=None, random_state=None):
        """
        Principal Component Analysis (PCA)

        PCA is a dimensionality reduction method that finds the vectors
        that maximize the variance of the data.

        Parameters:
        -----------
        n_components : int, float or None, default=None
            Number of components to keep. If n_components is not set,
            all components are kept.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the SVD solver. Pass an int
            for reproducible results across multiple function calls.

        Attributes:
        -----------
        n_components_ : int
            The actual number of components used in the PCA.

        components_ : array, shape (n_components, n_features)
            Principal axes in feature space, representing the directions
            of maximum variance in the data.

        explained_variance_ : array, shape (n_components,)
            The amount of variance explained by each of the selected components.

        explained_variance_ratio_ : array, shape (n_components,)
            Percentage of variance explained by each of the selected components.

        singular_values_ : array, shape (n_components,)
            The singular values corresponding to each of the selected components.

        mean_ : array, shape (n_features,)
            Per-feature empirical mean, estimated from the training set.

        n_features_ : int
            Number of features in the training data.

        n_samples_ : int
            Number of samples in the training data.

        svd : SVD
            The underlying SVD object used for decomposition.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None
        self.svd = None

    def fit(self, X, y=None):
        """
        Fit the model with X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        X = self.__validate_data(X)
        self.n_samples_, self.n_features_ = X.shape

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        total_var = np.var(X, axis=0).sum()

        self.svd = SVD(n_components=self.n_components, random_state=self.random_state)
        self.svd.fit(X_centered)

        self.components_ = self.svd.components_
        self.n_components_ = self.components_.shape[0]
        self.singular_values_ = self.svd.singular_values_
        self.explained_variance_ = (self.singular_values_ ** 2) / (self.n_samples_ - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns:
        --------
        X_new : array-like, shape (n_samples, n_components)
            Projection of X in the first principal components.
        """
        X = self.__validate_data(X, reset=False)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        --------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Transform data back to its original space.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Original data reconstructed from the reduced data.
        """
        X = self.__validate_data(X, reset=False, check_n_features=False)
        return np.dot(X, self.components_) + self.mean_

    def __validate_data(self, X, reset=True, check_n_features=True):
        """
        Validate input data.

        Parameters:
        -----------
        X : array-like
            Input data to validate.
        reset : bool, default=True
            Whether to reset the n_features_ attribute.
        check_n_features : bool, default=True
            Whether to check if n_features matches the training data.

        Returns:
        --------
        X_validated : ndarray
            The validated input.

        Raises:
        -------
        ValueError
            If the input is not 2D, if the number of features doesn't match
            the training data when check_n_features is True, or if the input
            data type is not recognized.
        """
        X = SVD._SVD__to_numpy_array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if reset:
            self.n_features_ = X.shape[1]
        elif check_n_features and X.shape[1] != self.n_features_:
            raise ValueError(f"The number of features in X ({X.shape[1]}) does not match the number of features the model was trained on ({self.n_features_}).")

        return X

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    X = np.random.rand(100, 5)

    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)

    print("Original data shape:", X.shape)
    print("Transformed data shape:", X_transformed.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Test inverse_transform
    X_reconstructed = pca.inverse_transform(X_transformed)
    print("Reconstructed data shape:", X_reconstructed.shape)