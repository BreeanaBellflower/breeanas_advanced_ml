import numpy as np

class LogisticRegression:
    def __init__(self, penalty='l2', C=1.0, learning_rate=0.01, max_iter=100):
        """
        A logistic regression classifier with an interface similar to sklearn.

        This implementation uses the sigmoid function to model the probability
        of the positive class and optimizes the log loss (cross-entropy)
        using batch gradient descent with optional regularization.

        Parameters:
        -----------
        penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
            The penalty (regularization term) to be used.
        C : float, default=1.0
            Inverse of regularization strength; must be a positive float.
            Smaller values specify stronger regularization.
        learning_rate : float, default=0.01
            The step size for gradient descent.
        max_iter : int, default=100
            Maximum number of iterations for gradient descent.
        
        Attributes:
        -----------
        coef_ : ndarray of shape (n_features,)
            Coefficient of the features in the decision function.
        intercept_ : float
            Intercept (bias) added to the decision function.
        n_iter_ : int
            Actual number of iterations used in the optimization.
        """
        self.penalty = penalty
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def __sigmoid(self, z):
        """
        Compute the sigmoid function.

        The sigmoid function is defined as:
        σ(z) = 1 / (1 + exp(-z))

        This function maps any real number to the range (0, 1),
        which we interpret as a probability.

        Parameters:
        -----------
        z : ndarray
            The input to the sigmoid function.

        Returns:
        --------
        ndarray
            The sigmoid of the input, element-wise.
        """
        return 1 / (1 + np.exp(-z))
    
    def __forward(self, X):
        """
        Compute the predicted probabilities for input samples.

        Applies sigmoid(X · coefs + bias) to get probabilities.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns:
        --------
        ndarray of shape (n_samples,)
            Predicted probabilities for the positive class.
        """
        linear_model = np.dot(X, self.coef_) + self.intercept_
        return self.__sigmoid(linear_model)

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

    def __compute_gradient(self, X, y, y_pred):
        """
        Compute the gradient of the loss function with respect to weights and bias.

        This method calculates the gradient of the log loss (cross-entropy loss)
        function, including the regularization term if specified. The gradient
        is used to update the model parameters (weights and bias) during the
        optimization process.

        The gradient computation includes:
        1. The gradient of the log loss with respect to weights and bias.
        2. The gradient of the regularization term (if applicable).

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        y : ndarray of shape (n_samples,)
            The true labels (target values).
        y_pred : ndarray of shape (n_samples,)
            The predicted probabilities for the positive class.

        Returns:
        --------
        dw : ndarray of shape (n_features,)
            The gradient of the loss function with respect to the weights.
        db : float
            The gradient of the loss function with respect to the bias.

        Notes:
        ------
        The gradient computation varies based on the regularization type:
        - For 'l2' regularization: dw += (1 / C) * weights
        - For 'l1' regularization: dw += (1 / C) * sign(weights)
        - For 'elasticnet' regularization: dw += (1 / (2C)) * (weights + sign(weights))

        Where C is the inverse of regularization strength.

        The bias term is not regularized.
        """
        n_samples = X.shape[0]
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        if self.penalty == 'l2':
            dw += (1 / self.C) * self.coef_
        elif self.penalty == 'l1':
            dw += (1 / self.C) * np.sign(self.coef_)
        elif self.penalty == 'elasticnet':
            dw += (1 / (2 * self.C)) * (self.coef_ + np.sign(self.coef_))
        
        return dw, db

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        The update rules for the coefs and bias are:
        coefs := coefs - learning_rate * ∂J/∂coefs
        bias := bias - learning_rate * ∂J/∂bias

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in {0, 1}).

        Returns:
        --------
        self : LogisticRegression
            Returns an instance of self.
        """
        X = self.__to_numpy_array(X)
        y = self.__to_numpy_array(y)

        # Input validation
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2:
            y = y.ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary values (0 and 1).")

        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Gradient descent
        for self.n_iter_ in range(self.max_iter):
            y_pred = self.__forward(X)
            
            # Compute gradients
            dw, db = self.__compute_gradient(X, y, y_pred)
            
            # Update parameters
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

        self.n_iter_ += 1
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        ndarray of shape (n_samples, 2)
            Probability of the sample for each class in the model.
            The columns correspond to the classes in sorted order.
        """
        X = self.__to_numpy_array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, but the model was trained with {self.coef_.shape[0]} features.")

        y_proba = self.__forward(X)
        return np.column_stack((1 - y_proba, y_proba))

    def predict(self, X):
        """
        Predict class labels for samples in X.

        The predicted class label is 1 if the predicted probability
        is greater than or equal to 0.5, and 0 otherwise.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        y_proba = self.predict_proba(X)
        return (y_proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X = self.__to_numpy_array(X)
        y = self.__to_numpy_array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2:
            y = y.ravel()

        return np.mean(self.predict(X) == y)

# Example usage:
if __name__ == "__main__":
    from IPython.display import display, HTML

    # Generate some random data
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create and train the model
    model = LogisticRegression(penalty='l2', C=1.0, learning_rate=0.1, max_iter=1000)
    model.fit(X, y)

    # Display model information
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Number of iterations: {model.n_iter_}")
    print(f"Training Score: {100*model.score(X, y)}%")

    # Make predictions
    X_new = np.array([[1, 2], [-1, -2]])
    print("Probabilities:", model.predict_proba(X_new))
    print("Predictions:", model.predict(X_new))