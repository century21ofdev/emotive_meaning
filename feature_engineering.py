import numpy as np
from nltk.probability import FreqDist
from collections import namedtuple


def bow(df):
    """bag of words"""
    bag = np.zeros(len(df['Review']), dtype=np.float64)
    for i in df['Review']:
        for k, j in enumerate(i):
            if j in i:
                bag[k] = 1
    return bag


def frequency_distribution(df, len_fd: int = 10):
    """frequency distribution"""
    fd = namedtuple("fd", ["most_common", "chart", "all"])
    words = list()

    for i in df["Review"]:
        for j in i:
            words.append(j)

    f_dist = FreqDist(words)
    fd.all = f_dist
    fd.most_common = f_dist.most_common(len_fd)
    fd.chart = f_dist.plot(10)
    return fd
