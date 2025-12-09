import sys
import os
import numpy as np

# needded to get the class imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knn.distance.manhattan_distance import ManhattanDistance

def test_same_points():
    """
    https://www.wolframalpha.com/input?i=ManhattanDistance%5B%7B1%2C+2%2C+3%7D%2C+%7B1%2C+2%2C+3%7D%5D
    """
    point1 = np.array([[1, 2, 3]])
    point2 = np.array([1, 2, 3])
    dist = ManhattanDistance()
    assert dist.calculate_distances(point1, point2)[0] == 0.0

def test_different_points():
    """
    https://www.wolframalpha.com/input?i=ManhattanDistance%5B%7B1%2C+0%2C+3%7D%2C+%7B1%2C+2%2C+3%7D%5D
    """
    point1 = np.array([[1, 0, 3]])
    point2 = np.array([1, 2, 3])
    dist = ManhattanDistance()
    assert dist.calculate_distances(point1, point2)[0] == 2.0

def test_negative_points():
    """
    https://www.wolframalpha.com/input?i=ManhattanDistance%5B%7B-1%2C+0%2C+3%7D%2C+%7B1%2C+2%2C+3%7D%5D
    """
    point1 = np.array([[-1, 0, 3]])
    point2 = np.array([1, 2, 3])
    dist = ManhattanDistance()
    assert dist.calculate_distances(point1, point2)[0] == 4

def test_decimals_points():
    point1 = np.array([[0.1, 2, 3]])
    point2 = np.array([0.1, 2, 3])
    dist = ManhattanDistance()
    assert dist.calculate_distances(point1, point2)[0] == 0

def test_many_many_points():
    p1_list = [1, 2, 3, 4, 5, 6, 7, 8, 9] * 10000
    point1 = np.array([p1_list])
    point2 = np.array(p1_list)
    dist = ManhattanDistance()
    assert dist.calculate_distances(point1, point2)[0] == 0

if __name__ == '__main__':
    test_same_points()
    test_different_points()
    test_negative_points()
    test_decimals_points()
    test_many_many_points()
