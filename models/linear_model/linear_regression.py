import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000, batch_size=32, alpha=0.01):
        """
        LinearRegression

        A class implementing linear regression using mini-batch gradient descent with L2 regularization.

        This class performs linear regression on a given dataset, optimizing the parameters
        using mini-batch gradient descent. It includes L2 regularization to prevent overfitting.

        Attributes:
            learning_rate (float, optional): The step size for each iteration of gradient descent. Defaults to 0.001.
            n_iterations (int, optional): The number of iterations to run the gradient descent algorithm. Defaults to 1000.
            batch_size (int, optional): The number of samples to use in each mini-batch. Defaults to 32.
            alpha (float, optional): The regularization strength (L2 regularization parameter). Defaults to 0.01.
            theta (numpy.ndarray): The learned model parameters. Initializes with np.rand

        Methods:
            fit(X, y): Fit the linear regression model to the training data.
            predict(X): Make predictions using the trained model.
            get_cost(X, y): Calculate the cost (mean squared error with regularization) of the model.

        Example:
            >>> model = LinearRegression(learning_rate=0.01, n_iterations=1000, batch_size=32, alpha=0.01)
            >>> model.fit(X_train, y_train)
            >>> predictions = model.predict(X_test)
            >>> cost = model.get_cost(X_test, y_test)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.alpha = alpha
        self.theta = None

    def _initialize_weights(self, n_features):
        # Xavier initialization
        limit = np.sqrt(6 / (n_features + 1))
        self.theta = np.random.uniform(-limit, limit, (n_features + 1,))
        print(f"Initialized theta shape: {self.theta.shape}")  # Debug print

    def _learning_rate_schedule(self, t):
        return self.learning_rate / (1 + 0.01 * t)
    
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

    @staticmethod
    def __sum_of_squared_residuals(y_true, y_pred):
        """
        Calculate the Sum of Squared Residuals.
        
        Parameters:
        y_true (np.array): True target values, shape (n_samples,)
        y_pred (np.array): Predicted values, shape (n_samples,)
        
        Returns:
        float: Sum of Squared Residuals
        """
        residuals = y_true - y_pred
        return np.sum(residuals**2)

    def get_cost(self, X, y):
        """
        Calculate the cost function (Mean Squared Error).
        
        Parameters:
        X (np.array): Input features, shape (n_samples, n_features)
        y (np.array): True target values, shape (n_samples,)
        
        Returns:
        float: Cost (MSE)
        """
        X = self.__to_numpy_array(X)
        y = self.__to_numpy_array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2:
            y = y.ravel()
        n_samples = len(y)
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        regularization = (self.alpha / (2 * n_samples)) * np.sum(self.theta[1:] ** 2)
        return mse + regularization

    def predict(self, X):
        """
        Make predictions using the current model parameters.
        
        Parameters:
        X (np.array): Input features, shape (n_samples, n_features)
        
        Returns:
        np.array: Predictions, shape (n_samples,)
        """
        X = self.__to_numpy_array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == self.theta.shape[0] - 1:
            # If X doesn't include the intercept term, add it
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        else:
            # If X already includes the intercept term, use it as is
            X_with_intercept = X
        return np.dot(X_with_intercept, self.theta)

    def fit(self, X, y):
        """
        Fit the linear regression model using gradient descent.
        
        Parameters:
        X (np.array): Input features, shape (n_samples, n_features)
        y (np.array): True target values, shape (n_samples,)
        
        Returns:
        self: Returns an instance of self.
        """
        X = self.__to_numpy_array(X)
        y = self.__to_numpy_array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y.ndim == 2:
            y = y.ravel()
        n_samples, n_features = X.shape
        print(f"Number of features: {n_features}")  # Debug print
        self._initialize_weights(n_features)
        print(f"Theta shape: {self.theta.shape}")  # Debug print
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        print(f"X_with_intercept shape: {X_with_intercept.shape}")  # Debug print
        for epoch in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                y_pred = np.dot(X_batch, self.theta)
                error = y_pred - y_batch
                gradient = np.dot(X_batch.T, error) / self.batch_size + self.alpha * self.theta
                
                lr = self._learning_rate_schedule(epoch)
                self.theta -= lr * gradient
            
            if epoch % 100 == 0:
                ssr = self.__sum_of_squared_residuals(y, self.predict(X))
                cost = self.get_cost(X_with_intercept, y)
                print(f"Epoch {epoch}, SSR: {ssr}, Cost: {cost}")
        
        return self
