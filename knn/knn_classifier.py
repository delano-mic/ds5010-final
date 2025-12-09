import pandas as pd
import numpy as np
from knn.distance.euclidean_distance import EuclideanDistance
from knn.distance.manhattan_distance import ManhattanDistance

EUCLIDEAN_DISTANCE_METRIC = "euclidean"
MANHATTAN_DISTANCE_METRIC = "manhattan"

class KNNClassifier():
    """
    The KNN classifer is able to classify objects by deriving classification from K neighbors. 
    The classification of an unknown data point is derived from the majority classification of it's neighbors. 
    Neighbors are determined by the proximity between the unknown data point and the training data points. 
    The proximity is determined by distance functions like Euclidean and Manhattan.
    """

    def __init__(self, n_neighbors: int = 3, distance_metric: str = EUCLIDEAN_DISTANCE_METRIC):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

        if self.distance_metric == EUCLIDEAN_DISTANCE_METRIC:
            self.distance_metric_calculator = EuclideanDistance()
        elif self.distance_metric == MANHATTAN_DISTANCE_METRIC:
            self.distance_metric_calculator = ManhattanDistance()
        else:
            raise ValueError(f"disntance_metric must be either {EUCLIDEAN_DISTANCE_METRIC} or {MANHATTAN_DISTANCE_METRIC}")

        # Do nothing now. We'll set these later when we "fit" the training data
        self.training_data_points = None
        self.training_labels = None

    def fit(self, training_data_points: pd.DataFrame, training_labels: pd.Series):
        """
        Fit does nothing but store the data and labels
        https://towardsdatascience.com/k-nearest-neighbor-classifier-explained-a-visual-guide-with-code-examples-for-beginners-a3d85cad00e1/
        """
        # Convert to numpy arrays for vectorization
        self.training_data_points = training_data_points.values
        self.training_labels = training_labels.values

    def predict(self, test_data_points: pd.DataFrame) -> pd.Series:
        predictions = []
        
        # Make a numpy array
        test_data = test_data_points.values

        for row_index, test_point in enumerate(test_data):
            if row_index % 100 == 0:
                print(f"Processing row_index {row_index}")
            
            # In a dissapointing turn of events I had to convert to using numpy for distance 
            # calculations because my original implementation using a loop and disntance functions 
            # in /knn/distance were incredibly slow
            distances = self.distance_metric_calculator.calculate_distances(self.training_data_points, test_point)
            
            # https://www.google.com/search?q=how+to+get+the+top+10+items+from+a+numpy+list
            nearest_neighbors = np.argpartition(distances, self.n_neighbors)[0:self.n_neighbors]

            # Get the labels for the k nearest neighbors
            nearest_neighbor_labels = self.training_labels[nearest_neighbors]
            
            # Determine which label has the majority in the nearesrt neighbors
            # https://www.google.com/search?q=get+the+most+frquent+item+in+a+numpy+array&oq=get+the+most+frquent+item+in+a+numpy+array
            unique_values, counts = np.unique(nearest_neighbor_labels, return_counts=True)
            max_count_index = np.argmax(counts)
            most_frequent_item = unique_values[max_count_index]
            predictions.append(most_frequent_item)
            
        return pd.Series(predictions)