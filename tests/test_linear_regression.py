import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from models.linear_model.linear_regression import LinearRegression

@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_initialization():
    model = LinearRegression()
    assert model.learning_rate == 0.001
    assert model.n_iterations == 1000
    assert model.batch_size == 32
    assert model.alpha == 0.01
    assert model.theta is None

def test_fit_predict(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    np.random.seed(42)
    model = LinearRegression(n_iterations=500)
    model.fit(X_train, y_train)
    
    assert model.theta is not None
    assert model.theta.shape == (X_train.shape[1] + 1,)
    
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape
    
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.5

def test_get_cost(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    cost = model.get_cost(X_test, y_test)
    assert isinstance(cost, float)
    assert cost > 0

def test_learning_rate_schedule():
    model = LinearRegression(learning_rate=0.1)
    assert model._learning_rate_schedule(0) == 0.1
    assert model._learning_rate_schedule(100) < 0.1

def test_input_validation():
    model = LinearRegression()
    
    with pytest.raises(ValueError):
        model.fit("not an array", [1, 2, 3])
    
    with pytest.raises(ValueError):
        model.predict("not an array")

def test_single_feature():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 5, 4, 5])
    
    model = LinearRegression(n_iterations=1000)
    model.fit(X, y)
    
    y_pred = model.predict(np.array([6]).reshape(-1, 1))
    assert y_pred.shape == (1,)

def test_regularization():
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_with_reg = LinearRegression(alpha=0.1, n_iterations=1000)
    model_without_reg = LinearRegression(alpha=0, n_iterations=1000)
    
    model_with_reg.fit(X_train, y_train)
    model_without_reg.fit(X_train, y_train)
    
    assert np.sum(np.abs(model_with_reg.theta)) < np.sum(np.abs(model_without_reg.theta))

def test_batch_size():
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    model_small_batch = LinearRegression(batch_size=10, n_iterations=100)
    model_large_batch = LinearRegression(batch_size=100, n_iterations=100)
    
    model_small_batch.fit(X, y)
    model_large_batch.fit(X, y)
    
    assert not np.allclose(model_small_batch.theta, model_large_batch.theta)

if __name__ == "__main__":
    pytest.main()