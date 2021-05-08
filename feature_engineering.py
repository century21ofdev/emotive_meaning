import numpy as np


def bow(df):
    """bag og words"""
    bag = np.zeros(len(df['Review']), dtype=np.float64)
    for i in df['Review']:
        for k, j in enumerate(i):
            if j in i:
                bag[k] = 1
    return bag
