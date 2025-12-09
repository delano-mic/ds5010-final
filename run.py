import pandas as pd
import os
import argparse
from knn.scaler.min_max_normalization_scaler import MinMaxNormalizationScaler
from knn.knn_classifier import KNNClassifier, EUCLIDEAN_DISTANCE_METRIC, MANHATTAN_DISTANCE_METRIC

"""
This file tests the data points in /data/test against the full data set in /data/train. 
It's goal is to predict is a test point is classified as diabetes, 0/1.

This file is the program entrypoint. 
"""

DATA_DIR = 'data'
DATA_FILE_NAME = "diabetes_binary_health_indicators_BRFSS2015.csv"

def get_data_frame_from_file(data_file_type: str):
    """
    Initializes a data frame from the test data file
    """
    file_path = os.path.join(DATA_DIR, data_file_type, DATA_FILE_NAME)
    return pd.read_csv(file_path)

def prepare_data(df: pd.DataFrame):
    """
    Splits the data frame into features and labels, and normalizes the features
    """
    features = df.drop('Diabetes_binary', axis=1) 
    labels = df['Diabetes_binary']
    scaler = MinMaxNormalizationScaler()
    # How do I run a function on each column of a data frame?
    # https://www.google.com/search?q=how+do+I+runa.+function+on+each+column+of+a+python+data+frame%3F&oq=how+do+I+runa.+function+on+each+column+of+a+python+data+frame
    features = features.apply(scaler.scale)
    return features, labels

def trainKnnClassifier(n_neighbors, distance_metric):
    """
    Trains a KNN classifier on the training data which, beyond scaling does nothing but save the data to memory
    """
    df_train = get_data_frame_from_file("train")
    features, labels = prepare_data(df_train)
    knn = KNNClassifier(n_neighbors=n_neighbors, distance_metric=distance_metric)
    knn.fit(features, labels)

    return knn

def testKnnClassifier(n_neighbors, distance_metric):
    """
    Tests the KNN classifier on the test data
    """
    print(f"Running KNN with {n_neighbors} neighbors and {distance_metric} distance function")
    
    df_test = get_data_frame_from_file("test")
    features, labels = prepare_data(df_test)
    
    knnClassifier = trainKnnClassifier(n_neighbors=n_neighbors, distance_metric=distance_metric)
    predictions = knnClassifier.predict(features)
    
    # Calculate statistics
    correct_predictions = (predictions == labels).sum()
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    
    print(f"Correct: {correct_predictions}")
    print(f"Total: {total_predictions}")
    print(f"Accuracy: {accuracy:.2%}") # 2 decimal place precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--distance_metric', type=str, default="euclidean")
    
    args = parser.parse_args()
    
    testKnnClassifier(n_neighbors=args.n_neighbors, distance_metric=args.distance_metric)



