from knn.distance.abstract_distance import AbstractDistance
import numpy as np

class EuclideanDistance(AbstractDistance):
    """
    This is a distance strategy used in the KNN Classifer

    Euclidean distance is the sum of the squared difference between each corresponding features between the two instances.
    """

    def calculate_distances(self, training_data: np.ndarray, test_point: np.ndarray) -> np.ndarray:

        # Euclidena Distance: https://www.google.com/search?q=how+to+apply+the+euclidean+distance+function+to+a+numpy+array&oq=how+to+apply+the+euclidean+distance+function+to+a+numpy+array
        # Square root in python: https://www.google.com/search?q=how+to+get+the+square+root+in+python&oq=how+to+get+the+square+root+in+pyth&gs_lcrp=EgZjaHJvbWUqBwgAEAAYgAQyBwgAEAAYgAQyBggBEEUYOTIICAIQABgWGB4yCAgDEAAYFhgeMggIBBAAGBYYHjIICAUQABgWGB4yCAgGEAAYFhgeMggIBxAAGBYYHjIICAgQABgWGB4yCAgJEAAYFhge0gEIODkwOGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8
        difference = training_data - test_point
        squared_difference = difference**2
        sum_squared_difference = np.sum(squared_difference, axis=1) # axis=1 ensures that we're operating on the rows within the arrays
        distance = np.sqrt(sum_squared_difference)
        return distance
    

    
    
    
