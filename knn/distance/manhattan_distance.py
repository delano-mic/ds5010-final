from knn.distance.abstract_distance import AbstractDistance
import numpy as np

class ManhattanDistance(AbstractDistance):
    """
    This is a distance strategy used in the KNN Classifer

    https://www.geeksforgeeks.org/data-science/manhattan-distance/
    """

    def calculate_distances(self, training_data: np.ndarray, test_point: np.ndarray) -> np.ndarray:
        # https://www.google.com/search?q=how+to+apply+the+manhattan+distance+function+to+a+numpy+array&sca_esv=eac84dcf468067b9&sxsrf=AE3TifMU0wHJRi14Z9zRPnmU3_ZqHHgV1g%3A1765058688672&ei=gKg0abHVKNehiLMP66LTsA4&ved=2ahUKEwixlZWB_KmRAxXXEGIAHWvRFOYQ4dUDegQIBRAN&uact=5&oq=how+to+apply+the+manhattan+distance+function+to+a+numpy+array
        difference = training_data - test_point
        absolute_difference = np.abs(difference)
        sum_absolute_difference = np.sum(absolute_difference, axis=1)
        return sum_absolute_difference
    

    
    
    
