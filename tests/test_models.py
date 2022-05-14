from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import numpy as np
from src import models

def test_train_model() :
    model = LogisticRegression()
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(2, size=20)
    model = models.train_model(model, X_train, y_train)
    assert check_is_fitted(model, "coef_") is None
    
    
def test_compute_model_metrics():
    model = LogisticRegression()
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(2, size=20)
    model = models.train_model(model, X_train, y_train)
    preds = model.predict(X_train)
    p, r, f = models.compute_model_metrics(y_train, preds)
    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= f <= 1
    
    
def test_inference():
    model = LogisticRegression()
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(2, size=20)
    model = models.train_model(model, X_train, y_train)
    preds = models.inference(model, X_train)
    assert np.all([x in [0, 1] for x in preds])
    