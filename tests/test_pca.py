import pytest
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from models.decomposition.pca import PCA

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.rand(100, 5)

def test_pca_initialization():
    pca = PCA(n_components=3)
    assert pca.n_components == 3
    assert pca.random_state is None
    assert pca.components_ is None
    assert pca.explained_variance_ is None
    assert pca.explained_variance_ratio_ is None
    assert pca.singular_values_ is None
    assert pca.mean_ is None

def test_pca_fit(sample_data):
    pca = PCA(n_components=3)
    sklearn_pca = SklearnPCA(n_components=3)
    
    pca.fit(sample_data)
    sklearn_pca.fit(sample_data)
    
    print("\n--- test_pca_fit ---")
    print(f"PCA n_components_: {pca.n_components_}")
    print(f"sklearn PCA n_components_: {sklearn_pca.n_components_}")
    assert pca.n_components_ == sklearn_pca.n_components_
    
    print(f"PCA n_features_: {pca.n_features_}")
    print(f"sklearn PCA n_features_: {sklearn_pca.n_features_}")
    assert pca.n_features_ == sklearn_pca.n_features_
    
    print(f"PCA n_samples_: {pca.n_samples_}")
    print(f"sklearn PCA n_samples_: {sklearn_pca.n_samples_}")
    assert pca.n_samples_ == sklearn_pca.n_samples_
    
    print("PCA components_:\n", pca.components_)
    print("sklearn PCA components_:\n", sklearn_pca.components_)
    np.testing.assert_allclose(np.abs(pca.components_), np.abs(sklearn_pca.components_), rtol=1e-5, atol=1e-8)
    
    print("PCA explained_variance_:", pca.explained_variance_)
    print("sklearn PCA explained_variance_:", sklearn_pca.explained_variance_)
    np.testing.assert_allclose(pca.explained_variance_, sklearn_pca.explained_variance_, rtol=1e-5, atol=1e-8)
    
    print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_)
    print("sklearn PCA explained_variance_ratio_:", sklearn_pca.explained_variance_ratio_)
    np.testing.assert_allclose(pca.explained_variance_ratio_, sklearn_pca.explained_variance_ratio_, rtol=1e-2, atol=1e-2)
    
    print("PCA singular_values_:", pca.singular_values_)
    print("sklearn PCA singular_values_:", sklearn_pca.singular_values_)
    np.testing.assert_allclose(pca.singular_values_, sklearn_pca.singular_values_, rtol=1e-5, atol=1e-8)
    
    print("PCA mean_:", pca.mean_)
    print("sklearn PCA mean_:", sklearn_pca.mean_)
    np.testing.assert_allclose(pca.mean_, sklearn_pca.mean_, rtol=1e-5, atol=1e-8)

def test_pca_transform(sample_data):
    pca = PCA(n_components=3)
    sklearn_pca = SklearnPCA(n_components=3)
    
    transformed = pca.fit_transform(sample_data)
    sklearn_transformed = sklearn_pca.fit_transform(sample_data)
    
    assert transformed.shape == sklearn_transformed.shape
    np.testing.assert_allclose(np.abs(transformed), np.abs(sklearn_transformed), rtol=1e-5, atol=1e-8)

def test_pca_inverse_transform(sample_data):
    pca = PCA(n_components=3)
    sklearn_pca = SklearnPCA(n_components=3)
    
    print("\n--- test_pca_inverse_transform ---")
    print("Original sample_data shape:", sample_data.shape)
    
    transformed = pca.fit_transform(sample_data)
    sklearn_transformed = sklearn_pca.fit_transform(sample_data)
    
    print("PCA transformed shape:", transformed.shape)
    print("sklearn PCA transformed shape:", sklearn_transformed.shape)
    
    print("PCA n_features_:", pca.n_features_)
    print("sklearn PCA n_features_:", sklearn_pca.n_features_)
    
    try:
        reconstructed = pca.inverse_transform(transformed)
        print("PCA reconstructed shape:", reconstructed.shape)
    except ValueError as e:
        print("PCA inverse_transform error:", str(e))
    
    sklearn_reconstructed = sklearn_pca.inverse_transform(sklearn_transformed)
    print("sklearn PCA reconstructed shape:", sklearn_reconstructed.shape)
    
    np.testing.assert_allclose(reconstructed, sklearn_reconstructed, rtol=1e-5, atol=1e-8)

def test_pca_explained_variance_ratio(sample_data):
    pca = PCA(n_components=5)
    sklearn_pca = SklearnPCA(n_components=5)
    
    pca.fit(sample_data)
    sklearn_pca.fit(sample_data)
    
    print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_)
    print("sklearn PCA explained_variance_ratio_:", sklearn_pca.explained_variance_ratio_)

    np.testing.assert_allclose(pca.explained_variance_ratio_, sklearn_pca.explained_variance_ratio_, rtol=1e-2, atol=1e-2)
    assert np.isclose(np.sum(pca.explained_variance_ratio_), 1.0, rtol=1e-2, atol=1e-2)
    assert np.all(pca.explained_variance_ratio_[:-1] >= pca.explained_variance_ratio_[1:])

def test_pca_with_different_n_components(sample_data):
    for n_components in [1, 3, 5]:
        pca = PCA(n_components=n_components)
        sklearn_pca = SklearnPCA(n_components=n_components)
        
        transformed = pca.fit_transform(sample_data)
        sklearn_transformed = sklearn_pca.fit_transform(sample_data)
        
        assert transformed.shape == sklearn_transformed.shape
        np.testing.assert_allclose(np.abs(transformed), np.abs(sklearn_transformed), rtol=1e-5, atol=1e-8)

def test_pca_with_n_components_none(sample_data):
    pca = PCA(n_components=None)
    sklearn_pca = SklearnPCA(n_components=None)
    
    transformed = pca.fit_transform(sample_data)
    sklearn_transformed = sklearn_pca.fit_transform(sample_data)
    
    assert transformed.shape == sklearn_transformed.shape
    np.testing.assert_allclose(np.abs(transformed), np.abs(sklearn_transformed), rtol=1e-5, atol=1e-8)

def test_pca_with_invalid_n_components(sample_data):
    pca = PCA(n_components=6)
    with pytest.raises(ValueError):
        pca.fit(sample_data)

def test_pca_with_pandas_dataframe(sample_data):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(sample_data)
    
    pca = PCA(n_components=3)
    sklearn_pca = SklearnPCA(n_components=3)
    
    transformed = pca.fit_transform(df)
    sklearn_transformed = sklearn_pca.fit_transform(df)
    
    assert transformed.shape == sklearn_transformed.shape
    np.testing.assert_allclose(np.abs(transformed), np.abs(sklearn_transformed), rtol=1e-5, atol=1e-8)

def test_pca_components_orthogonality(sample_data):
    pca = PCA(n_components=5)
    sklearn_pca = SklearnPCA(n_components=5)
    
    pca.fit(sample_data)
    sklearn_pca.fit(sample_data)
    
    np.testing.assert_allclose(np.dot(pca.components_, pca.components_.T), np.eye(5), atol=1e-7)
    np.testing.assert_allclose(np.dot(sklearn_pca.components_, sklearn_pca.components_.T), np.eye(5), atol=1e-7)

def test_pca_randomized_results(sample_data):
    random_state = 42
    pca = PCA(n_components=3, random_state=random_state)
    sklearn_pca = SklearnPCA(n_components=3, random_state=random_state)
    
    transformed = pca.fit_transform(sample_data)
    sklearn_transformed = sklearn_pca.fit_transform(sample_data)
    
    np.testing.assert_allclose(np.abs(transformed), np.abs(sklearn_transformed), rtol=1e-5, atol=1e-8)

def test_pca_fit_transform_equivalence(sample_data):
    pca = PCA(n_components=3)
    sklearn_pca = SklearnPCA(n_components=3)
    
    transformed1 = pca.fit(sample_data).transform(sample_data)
    sklearn_transformed1 = sklearn_pca.fit(sample_data).transform(sample_data)
    
    transformed2 = pca.fit_transform(sample_data)
    sklearn_transformed2 = sklearn_pca.fit_transform(sample_data)
    
    np.testing.assert_allclose(transformed1, transformed2, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(sklearn_transformed1, sklearn_transformed2, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.abs(transformed1), np.abs(sklearn_transformed1), rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    pytest.main()