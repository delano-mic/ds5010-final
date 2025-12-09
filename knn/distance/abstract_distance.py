from abc import ABC, abstractmethod
import numpy as np

class AbstractDistance(ABC):
    """
    There are many distinace functions. 
    This approach ensures that we can augment the distaince functions in our toolbox.
    """
    
    @abstractmethod
    def calculate_distances(self, training_data: np.ndarray, test_point: np.ndarray) -> np.ndarray:
        """
        Calculate the distances between a test point and all training data points.
        """
        pass
    

    
    
    
