import sys
import os
import pandas as pd

# needed to get the class imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knn.knn_classifier import KNNClassifier

def test_fit():
    knn = KNNClassifier()
    data = pd.DataFrame([[1, 1], [2, 2]])
    labels = pd.Series([0, 1])
    knn.fit(data, labels)
    assert knn.training_data_points.tolist() == data.values.tolist()
    assert knn.training_labels.tolist() == labels.values.tolist()

def test_predict_with_exact_neighbors():
    knn = KNNClassifier(n_neighbors=3)
    train_data = pd.DataFrame([[1, 1], [1, 1], [1, 1], [5, 5], [5, 5], [5, 5]])
    train_labels = pd.Series([0, 0, 0, 1, 1, 1])
    knn.fit(train_data, train_labels)
    
    test_data = pd.DataFrame([[1, 1], [5, 5]])
    predictions = knn.predict(test_data)
    
    assert predictions[0] == 0
    assert predictions[1] == 1

def test_predict_with_non_exact_neighbors():
    knn = KNNClassifier(n_neighbors=3)
    train_data = pd.DataFrame([[1, 1], [1, 2], [1, 3], [5, 5], [5, 6], [5, 7]])
    train_labels = pd.Series([0, 0, 0, 1, 1, 1])
    knn.fit(train_data, train_labels)
    
    test_data = pd.DataFrame([[1, 1], [5, 5]])
    predictions = knn.predict(test_data)
    
    assert predictions[0] == 0
    assert predictions[1] == 1
