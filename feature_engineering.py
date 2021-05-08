import numpy as np
from nltk.probability import FreqDist
from collections import namedtuple, Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def _reviews(df):
    words = list()

    for i in df["Review"]:
        for j in i:
            words.append(j)
    return words


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
    words = _reviews(df)

    f_dist = FreqDist(words)
    fd.all = f_dist
    fd.most_common = f_dist.most_common(len_fd)
    fd.chart = f_dist.plot(10)
    return fd


def word_cloud(df):
    """words & tags cloud from most common frequencies"""

    words = _reviews(df)
    word_could_dict = Counter(words)

    wc = WordCloud().generate_from_frequencies(word_could_dict)

    plt.figure(figsize=(12, 12))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
