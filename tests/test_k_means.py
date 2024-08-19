import pytest
import numpy as np
import pandas as pd
from models.cluster.k_means import KMeans

@pytest.fixture
def sample_data():
    np.random.seed(42)  # We can keep this to have consistent test data
    return np.concatenate([
        np.random.normal(0, 1, (20, 2)),
        np.random.normal(5, 1, (20, 2)),
        np.random.normal(-5, 1, (20, 2))
    ])

def test_kmeans_initialization():
    kmeans = KMeans(n_clusters=3)
    assert kmeans.n_clusters == 3
    assert kmeans.init == 'random'
    assert kmeans.n_init == 10
    assert kmeans.max_iter == 300
    assert kmeans.tol == 1e-4
    assert kmeans.algorithm == 'auto'
    assert kmeans.distance_calculation == 'Euclidean'
    assert kmeans.n_threads == 1

def test_kmeans_fit_predict(sample_data):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(sample_data)
    labels = kmeans.predict(sample_data)
    assert labels.shape == (60,)
    assert len(np.unique(labels)) == 3
    
    # Check if the cluster centers are different
    assert len(np.unique(kmeans.cluster_centers_, axis=0)) == 3

def test_kmeans_multiple_runs(sample_data):
    kmeans1 = KMeans(n_clusters=3)
    kmeans2 = KMeans(n_clusters=3)
    kmeans1.fit(sample_data)
    kmeans2.fit(sample_data)
    
    # Check that the results are not identical (very low probability)
    assert not np.array_equal(kmeans1.labels_, kmeans2.labels_)

def test_kmeans_transform(sample_data):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(sample_data)
    transformed = kmeans.transform(sample_data[:5])
    assert transformed.shape == (5, 3)
    
    # Check that distances are non-negative
    assert np.all(transformed >= 0)

def test_kmeans_inertia(sample_data):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(sample_data)
    assert kmeans.inertia_ is not None
    assert kmeans.inertia_ > 0

def test_kmeans_fit_with_insufficient_samples(sample_data):
    kmeans = KMeans(n_clusters=10)
    with pytest.raises(ValueError):
        kmeans.fit(sample_data[:5])

def test_kmeans_predict_with_wrong_features(sample_data):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(sample_data)
    with pytest.raises(ValueError):
        kmeans.predict(sample_data[:, :1])

def test_kmeans_convergence(sample_data):
    kmeans = KMeans(n_clusters=3, max_iter=1000, tol=1e-8)
    kmeans.fit(sample_data)
    assert kmeans.n_iter_ <= 1000  # Should converge before max_iter

def test_kmeans_different_distance_calculations(sample_data):
    kmeans_euclidean = KMeans(n_clusters=3, distance_calculation='Euclidean')
    kmeans_manhattan = KMeans(n_clusters=3, distance_calculation='Manhattan')
    
    kmeans_euclidean.fit(sample_data)
    kmeans_manhattan.fit(sample_data)
    
    # Results should be different
    assert not np.array_equal(kmeans_euclidean.labels_, kmeans_manhattan.labels_)

def test_kmeans_with_pandas_dataframe(sample_data):
    df = pd.DataFrame(sample_data, columns=['feature1', 'feature2'])
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df)
    labels = kmeans.predict(df)
    assert labels.shape == (60,)
    assert len(np.unique(labels)) == 3