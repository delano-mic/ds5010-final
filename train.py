import pandas as pd
import os
from knn.scaler.min_max_normalization_scaler import MinMaxNormalizationScaler
from knn.knn_classifier import KNNClassifier, EUCLIDEAN_DISTANCE_METRIC, MANHATTAN_DISTANCE_METRIC

DATA_DIR = 'data'
DATA_FILE_NAME = "diabetes_binary_health_indicators_BRFSS2015.csv"

def trainKnnClassifier():
    file_path = os.path.join(DATA_DIR, "train", DATA_FILE_NAME)
    df_train = pd.read_csv(file_path)

    Y = df_train['Diabetes_binary']
    x = df_train.drop('Diabetes_binary', axis=1) # axis is required to make sure the column is dropped rather than looking for a row with a label of 'Diabetes_binary'

    scaler = MinMaxNormalizationScaler()
    # How do I run a function on each column of a data frame?
    # https://www.google.com/search?q=how+do+I+runa.+function+on+each+column+of+a+python+data+frame%3F&oq=how+do+I+runa.+function+on+each+column+of+a+python+data+frame
    x = x.apply(scaler.scale)

    # print(x.head())

    knn = KNNClassifier(n_neighbors=10, distance_metric=EUCLIDEAN_DISTANCE_METRIC)
    knn.fit(x, Y)

    return knn

def testKnnClassifier():
    file_path = os.path.join(DATA_DIR, "test", DATA_FILE_NAME)
    df_test = pd.read_csv(file_path)
    
    # Drop the label column to match training features
    test_points = df_test.drop('Diabetes_binary', axis=1)
    true_labels = df_test['Diabetes_binary']
    
    # Normalize test data (Crucial: Model expects normalized features)
    scaler = MinMaxNormalizationScaler()
    test_points = test_points.apply(scaler.scale)
    
    knnClassifier = trainKnnClassifier()
    predictions = knnClassifier.predict(test_points)
    
    # Calculate statistics
    correct_predictions = (predictions == true_labels).sum()
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    
    print(f"Correct: {correct_predictions}")
    print(f"Total: {total_predictions}")
    print(f"Accuracy: {accuracy:.2%}")

testKnnClassifier()



