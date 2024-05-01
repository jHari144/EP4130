import numpy as np

def remove_outliers_zscore(data, threshold=3):
    """
    Remove outliers from the dataset using Z-score method.

    Parameters:
    - data: numpy array or pandas DataFrame
        The dataset from which outliers will be removed.
    - threshold: float, optional (default=3)
        The threshold value for Z-score. Data points with a Z-score greater than
        this threshold will be considered outliers.

    Returns:
    - cleaned_data: numpy array or pandas DataFrame
        The dataset with outliers removed.
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    cleaned_data = data[(z_scores < threshold).all(axis=1)]
    return cleaned_data

# Example usage:
# Assuming 'data' is your dataset (numpy array or pandas DataFrame)
# cleaned_data = remove_outliers_zscore(data)

import pandas as pd

if __name__ == "__main__":
    # data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    data = pd.read_csv("generated.csv").to_numpy()
    cleaned_data = remove_outliers_zscore(data)
    
    # save the cleaned data to a new csv file
    pd.DataFrame(cleaned_data).to_csv("cleaned_data.csv", header=False, index=False)