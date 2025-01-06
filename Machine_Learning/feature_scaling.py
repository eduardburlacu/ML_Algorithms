"""
Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization.
The function should take a 2D NumPy array as input,
    where each row represents a data sample and each column represents a feature.
It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization.
Make sure all results are rounded to the nearest 4th decimal.
"""
import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    standardized_data = (data - np.mean(data,axis=0,keepdims=True)) / np.std(data,axis=0,keepdims=True)
    print(np.mean(data,axis=0,keepdims=True))
    print(np.std(data, axis=0,keepdims=True))
    min_data = np.min(data, axis=0,keepdims=True)
    max_data = np.max(data, axis=0,keepdims=True)
    normalized_data= (data-min_data)/(max_data-min_data)

    return standardized_data, normalized_data

if __name__ == '__main__':
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(*feature_scaling(data), sep="\n")

    # (array([[-0.81649658, -0.81649658, -0.81649658],
    #        [ 0.        ,  0.        ,  0.        ],
    #        [ 0.81649658,  0.81649658,  0.81649658]]),
    #  array([[0. , 0. , 0. ],
    #         [0.5, 0.5, 0.5],
    #         [1. , 1. , 1. ]]))