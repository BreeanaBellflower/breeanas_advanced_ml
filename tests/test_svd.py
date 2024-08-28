import pytest
import numpy as np
from sklearn.decomposition import TruncatedSVD
from models.decomposition.svd import SVD

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.rand(10, 5)

def test_svd_initialization():
    svd = SVD(n_components=3)
    assert svd.n_components == 3
    assert svd.random_state is None
    assert svd.singular_values_ is None
    assert svd.components_ is None
    assert svd.U is None

def test_svd_fit(sample_data):
    svd = SVD(n_components=3)
    svd.fit(sample_data)
    assert svd.singular_values_ is not None
    assert svd.components_ is not None
    assert svd.U is not None
    assert len(svd.singular_values_) == 3
    assert svd.components_.shape == (3, 5)
    assert svd.U.shape == (10, 3)

def test_svd_transform(sample_data):
    svd = SVD(n_components=3)
    skm = TruncatedSVD(n_components=3)
    svd.fit(sample_data)
    transformed = svd.transform(sample_data)
    compared = skm.fit_transform(sample_data)
    assert transformed.shape == (10, 3)
    np.testing.assert_allclose(np.abs(transformed), np.abs(compared), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.abs(svd.components_), np.abs(skm.components_), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.abs(svd.explained_variance_ratio_), np.abs(skm.explained_variance_ratio_), rtol=1e-5, atol=1e-8)

def test_svd_inverse_transform(sample_data):
    svd = SVD()
    svd.fit(sample_data)
    transformed = svd.transform(sample_data)
    reconstructed = svd.inverse_transform(transformed)
    assert reconstructed.shape == sample_data.shape
    np.testing.assert_allclose(sample_data, reconstructed, rtol=1e-5, atol=1e-8)

def test_svd_fit_transform(sample_data):
    svd = SVD(n_components=3)
    transformed = svd.fit_transform(sample_data)
    assert transformed.shape == (10, 3)

def test_svd_with_different_n_components(sample_data):
    for n_components in [1, 3, 5]:
        svd = SVD(n_components=n_components)
        transformed = svd.fit_transform(sample_data)
        assert transformed.shape == (10, n_components)

def test_svd_with_n_components_none(sample_data):
    svd = SVD(n_components=None)
    transformed = svd.fit_transform(sample_data)
    assert transformed.shape == sample_data.shape

def test_svd_with_invalid_n_components(sample_data):
    svd = SVD(n_components=6)
    with pytest.raises(ValueError):
        svd.fit(sample_data)

def test_svd_with_pandas_dataframe(sample_data):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(sample_data)
    svd = SVD(n_components=3)
    transformed = svd.fit_transform(df)
    assert transformed.shape == (10, 3)

def test_svd_with_list_input(sample_data):
    data_list = sample_data.tolist()
    svd = SVD(n_components=3)
    transformed = svd.fit_transform(data_list)
    assert transformed.shape == (10, 3)

def test_svd_with_invalid_input():
    svd = SVD(n_components=3)
    with pytest.raises(ValueError):
        svd.fit("invalid input")

def test_svd_transform_with_mismatched_features(sample_data):
    svd = SVD(n_components=3)
    svd.fit(sample_data)
    with pytest.raises(ValueError):
        svd.transform(np.random.rand(10, 6))

def test_svd_inverse_transform_with_mismatched_features(sample_data):
    svd = SVD(n_components=3)
    svd.fit(sample_data)
    with pytest.raises(ValueError):
        svd.inverse_transform(np.random.rand(10, 4))

def test_svd_singular_values_order(sample_data):
    svd = SVD(n_components=5)
    svd.fit(sample_data)
    assert np.all(svd.singular_values_[:-1] >= svd.singular_values_[1:])

def test_svd_orthogonality(sample_data):
    svd = SVD(n_components=5)
    svd.fit(sample_data)
    np.testing.assert_allclose(np.dot(svd.components_, svd.components_.T), np.eye(5), atol=1e-7)
    np.testing.assert_allclose(np.dot(svd.U.T, svd.U), np.eye(5), atol=1e-7)