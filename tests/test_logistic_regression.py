import pytest
import numpy as np
from models.linear_model.logistic_regression import LogisticRegression

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def test_initialization():
    model = LogisticRegression()
    assert model.penalty == 'l2'
    assert model.C == 1.0
    assert model.learning_rate == 0.01
    assert model.max_iter == 100

def test_fit(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)
    assert model.n_iter_ > 0

def test_predict_proba(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    probas = model.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.allclose(np.sum(probas, axis=1), 1)

def test_predict(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (100,)
    assert set(np.unique(predictions)) <= {0, 1}

def test_score(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    score = model.score(X, y)
    assert 0 <= score <= 1

def test_input_validation():
    model = LogisticRegression()
    
    # Test with valid input
    X_valid = np.array([[1, 2], [3, 4]])
    y_valid = np.array([0, 1])
    model.fit(X_valid, y_valid)  # This should not raise an error
    
    # Test with mismatched shapes
    X_mismatched = np.array([[1, 2], [3, 4], [5, 6]])
    y_mismatched = np.array([0, 1])
    with pytest.raises(ValueError):
        model.fit(X_mismatched, y_mismatched)
    
    # Test with non-binary y
    X_valid = np.array([[1, 2], [3, 4], [5, 6]])
    y_non_binary = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        model.fit(X_valid, y_non_binary)
    
    # Test with string input
    with pytest.raises(ValueError):
        model.fit(['a', 'b', 'c'], [0, 1, 0])

def test_predict_input_validation():
    model = LogisticRegression()
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model.fit(X_train, y_train)
    
    # Test with valid input
    X_valid = np.array([[5, 6]])
    model.predict(X_valid)  # This should not raise an error
    
    # Test with mismatched feature count
    X_invalid = np.array([[5, 6, 7]])
    with pytest.raises(ValueError):
        model.predict(X_invalid)
    
    # Test with string input
    with pytest.raises(ValueError):
        model.predict(['a', 'b'])

def test_penalty_options():
    penalties = ['l1', 'l2', 'elasticnet', None]
    for penalty in penalties:
        model = LogisticRegression(penalty=penalty)
        assert model.penalty == penalty

def test_convergence():
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = (X.sum(axis=1) > 0).astype(int)
    model = LogisticRegression(max_iter=1000, learning_rate=0.1)
    model.fit(X, y)
    assert model.score(X, y) > 0.8  # Assuming the model converges well

def test_to_numpy_array():
    model = LogisticRegression()
    
    # Test with list
    assert np.array_equal(model._LogisticRegression__to_numpy_array([1, 2, 3]), np.array([1, 2, 3]))
    
    # Test with numpy array
    arr = np.array([1, 2, 3])
    assert np.array_equal(model._LogisticRegression__to_numpy_array(arr), arr)
    
    # Test with pandas Series (if pandas is available)
    try:
        import pandas as pd
        series = pd.Series([1, 2, 3])
        assert np.array_equal(model._LogisticRegression__to_numpy_array(series), np.array([1, 2, 3]))
    except ImportError:
        pass  # Skip this test if pandas is not available

def test_1d_input():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 0, 1, 1, 1])
    model = LogisticRegression()
    model.fit(X, y)
    assert model.coef_.shape == (1,)
    assert model.predict(np.array([2, 4])).shape == (2,)