import numpy as np
import math

class KMeans:
    def __init__(
        self,
        n_clusters=8,
        init='random',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        algorithm='auto',
        distance_calculation='Euclidean',
        n_threads=1
    ):
        """
        Initialize the KMeans object.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - init (str, array-like, or callable): Method for initialization. Default "random". More implemented later.
        - n_init (int): Number of times the k-means algorithm will be run with different centroid seeds.
        - max_iter (int): Maximum number of iterations for a single run.
        - tol (float): Relative tolerance with regards to inertia to declare convergence.
        - algorithm (str): K-means algorithm to use. Options are "auto", "full" or "elkan".
        - distance_calculation (str): When calculating distances, which method should KMeans use? Options are "Euclidean" and "Manhattan".
        - n_threads (int): The number of threads to perform calculations on.
        """
        if n_clusters < 2 or not isinstance(n_clusters, (int, float)):
            raise ValueError('n_clusers must be a valid number >= 2')
        if init not in ['random']:
            raise ValueError(f'init type {init} is undefined')
        if n_init < 1 or not isinstance(n_init, (int, float)):
            raise ValueError('n_init must be a valid number >= 1')
        if max_iter < 1 or not isinstance(max_iter, (int, float)):
            raise ValueError('max_iter must be a valid number >= 1')
        if tol < 0 or not isinstance(tol, (int, float)):
            raise ValueError('tol must be a valid number >= 0')
        if algorithm not in  ['auto', 'full', 'elkan']:
            raise ValueError(f'algorithm type {algorithm} is undefined')
        if distance_calculation not in  ['Euclidean', 'Manhattan']:
            raise ValueError(f'distance_calculation type {distance_calculation} is undefined')
        if n_threads < 1 or not isinstance(n_threads, (int, float)):
            raise ValueError('n_threads must be a valid number >= 1')
        self.n_clusters = math.floor(n_clusters)
        self.init = init
        self.n_init = math.floor(n_init)
        self.max_iter = math.floor(max_iter)
        self.tol = tol
        self.algorithm = algorithm
        self.distance_calculation = distance_calculation
        self.n_threads = n_threads
        self.rng = np.random.default_rng()
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

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

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters:
        - X (array-like or sparse matrix): Training instances to cluster.

        Returns:
        - self: The fitted estimator.
        """
        X = self.__check_data(X)
        best_inertia = np.inf
        for iteration in range(self.n_init):
            self.__init_centroids(X)
            prev_inertia = np.inf
            for _ in range(self.max_iter):
                labels = self.predict(X)
                self.__update_centroids(X, labels)
                current_inertia = self.__compute_inertia(X)
                if self.__check_convergence(prev_inertia, current_inertia):
                    break
                prev_inertia = current_inertia
            
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_labels = labels
                best_centroids = self.cluster_centers_.copy()
                best_n_iter = iteration + 1

        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def __compute_inertia(self, X):
        """
        Compute the sum of squared distances of samples to their closest centroid.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - inertia (float): The sum of squared distances.
        """
        distances = np.min(np.sum((X[:, np.newaxis, :] - self.cluster_centers_) ** 2, axis=2), axis=1)
        return np.sum(distances)

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        - X (array-like or sparse matrix): New data to predict, shape (n_samples, n_features).

        Returns:
        - labels (array): Index of the cluster each sample belongs to.
        """
        X = self.__check_data(X)
        
        distances = self.__calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def __calculate_distances(self, X):
        """
        Calculate distances from each point in X to each cluster center.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - distances (array): Distances from each point to each cluster center.
        """
        n_samples = X.shape[0]
        n_clusters = self.cluster_centers_.shape[0]
        distances = np.zeros((n_samples, n_clusters))

        for i in range(n_clusters):
            if self.distance_calculation == 'Euclidean':
                distances[:, i] = np.sqrt(np.sum((X - self.cluster_centers_[i])**2, axis=1))
            elif self.distance_calculation == 'Manhattan':
                distances[:, i] = np.sum(np.abs(X - self.cluster_centers_[i]), axis=1)
            else:
                raise ValueError('Distance Calculation method unknown')
        
        return distances

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters:
        - X (array-like or sparse matrix): New data to predict.

        Returns:
        - labels (array): Index of the cluster each sample belongs to.
        """
        return self.fit(X).predict(X)

    def __init_centroids(self, X):
        """
        Initialize the centroids for the K-means algorithm.

        Parameters:
        - X (array-like): The input data.

        Returns:
        - centroids (array): The initial centroids.
        """
        if self.init not in ['random']:
            raise ValueError('Initialization method unknown')
        if self.init == 'random':
            indices = self.rng.choice(X.shape[0], self.n_clusters, replace=False)
            self.cluster_centers_ = X[indices]

    def __update_centroids(self, X, labels):
        """
        Recalculate centroids based on current cluster assignments.

        Parameters:
        - X (array-like): The input data.
        - labels (array): The current cluster assignments.

        Returns:
        - None: Updates self.cluster_centers_ in-place.
        """
        for k in range(self.n_clusters):
            cluster_datapoints = X[labels == k]
            
            if len(cluster_datapoints) > 0:
                new_centroid = np.mean(cluster_datapoints, axis=0)
            else:
                new_centroid = self.cluster_centers_[k]
            
            self.cluster_centers_[k] = new_centroid

    def __check_convergence(self, prev_inertia, current_inertia):
        """
        Check if the algorithm has converged based on the change in inertia.

        Parameters:
        - prev_inertia (float): The inertia from the previous iteration.
        - current_inertia (float): The inertia from the current iteration.

        Returns:
        - converged (bool): True if converged, False otherwise.
        """
        inertia_change = abs(prev_inertia - current_inertia)
        return inertia_change < self.tol

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        Parameters:
        - X (array-like or sparse matrix): New data to transform.

        Returns:
        - X_new (array): X transformed in the new space.
        """
        X = self.__check_data(X)
        return self.__calculate_distances(X)

    def __check_data(self, X):
        """
        Check if the input data is valid for fitting or predicting.

        Parameters:
        - X (array-like or sparse matrix): Input data.

        Raises:
        - ValueError: If the input data is not valid.
        """
        X = self.__to_numpy_array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
        
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}")
        
        if n_features == 0:
            raise ValueError("n_features must be > 0")
        
        if not np.all(np.isfinite(X)):
            raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64').")
        
        if self.cluster_centers_ is not None and n_features != self.cluster_centers_.shape[1]:
            raise ValueError(f"Incorrect number of features. Got {n_features} features, "
                             f"expected {self.cluster_centers_.shape[1]}")

        return X

if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (20, 2)),
        np.random.normal(5, 1, (20, 2)),
        np.random.normal(-5, 1, (20, 2))
    ])

    # Example 1: Basic usage
    print("Example 1: Basic Usage")
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    print("Cluster centers:", kmeans.cluster_centers_)
    print("Labels1:", labels[0:19], "...")
    print("Labels2:", labels[20:39], "...")
    print("Labels3:", labels[40:], "...")
    print("Inertia:", kmeans.inertia_)
    print()

    # Example 2: Using different distance calculation
    print("Example 2: Manhattan Distance")
    kmeans_manhattan = KMeans(n_clusters=3, distance_calculation='Manhattan')
    kmeans_manhattan.fit(X)
    labels_manhattan = kmeans_manhattan.predict(X)
    print("Cluster centers (Manhattan):", kmeans_manhattan.cluster_centers_)
    print("Labels1:", labels[0:19], "...")
    print("Labels2:", labels[20:39], "...")
    print("Labels3:", labels[40:], "...")
    print("Inertia (Manhattan):", kmeans_manhattan.inertia_)
    print()

    # Example 3: Transform method
    print("Example 3: Transform Method")
    X_transformed = kmeans.transform(X[:5])
    print("Transformed data (first 5 points):")
    print(X_transformed)
    print()

    # Example 4: Trying different numbers of clusters
    print("Example 4: Different Numbers of Clusters")
    for n_clusters in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        print(f"Inertia with {n_clusters} clusters:", kmeans.inertia_)
    print()

    # Example 5: Handling errors
    print("Example 5: Error Handling")
    try:
        kmeans_error = KMeans(n_clusters=10)
        kmeans_error.fit(X[:5])  # Try to fit with fewer samples than clusters
    except ValueError as e:
        print("Caught expected error:", str(e))
    
    try:
        kmeans.predict(X[:, :1])  # Try to predict with wrong number of features
    except ValueError as e:
        print("Caught expected error:", str(e))
    print()

    # Example 6: Using with pandas DataFrame
    print("Example 6: Using with pandas DataFrame")
    try:
        import pandas as pd
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        kmeans_pd = KMeans(n_clusters=3)
        kmeans_pd.fit(df)
        labels_pd = kmeans_pd.predict(df)
        print("Labels1:", labels[0:19], "...")
        print("Labels2:", labels[20:39], "...")
        print("Labels3:", labels[40:], "...")
    except ImportError:
        print("pandas not installed, skipping this example")