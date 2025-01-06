"""
Your task is to implement a function that calculates the Gini Impurity for a set of classes.
Gini impurity is commonly used in decision tree algorithms to measure the impurity or disorder within a node.

Write a function gini_impurity(y) that takes in a list of class labels
"""
import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    probs = np.unique(y, return_counts =True)[1] / np.size(y)
    return 1. - np.sum(probs**2)

if __name__ == '__main__':
    y = [1, 2, 0, 3, 2, 3, 3, 3, 0, 1, 0]
    print(round(gini_impurity(y), 3)) #0.727
