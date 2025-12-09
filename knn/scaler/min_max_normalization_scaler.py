import pandas as pd

# https://www.geeksforgeeks.org/data-analysis/normalization-and-scaling/
class MinMaxNormalizationScaler():
    """
    This is a scaler strategy used in the KNN Classifer. 
    It's primary goal is to ensure the equal weighting of data which is important with KNN
    since it calculates distances beteween data points which, if unscaled, gives larger 
    magnitude data points a weight advantage in the distance calculation.
    """
    
    def scale(self, series: pd.Series):
        #https://pandas.pydata.org/docs/reference/api/pandas.Series.max.html
        max = series.max()
        min = series.min()
        
        return (series - min) / (max - min) # Vectorize the calculation
        