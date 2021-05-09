import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from preprocessing import (pipeline_1, pipeline_2, pipeline_3)
from feature_engineering import (bow, frequency_distribution, word_cloud, pos_tagging, ner, count_vectorizer,
                                 terms_frequency)
from sklearn.model_selection import train_test_split
from model import build_model
import matplotlib.pyplot as plt


# @TODO Automize workaround pipeline

def _data_definition():
    df = pipeline_1()

    ner_, tree = ner(df)
    print("Named entity recognition : ", ner_)
    print(tree)

    bow_ = bow(df)
    print("Bag of words : ", bow_)

    pos_tagging_ = pos_tagging(df)
    print("Part of speech tagging", pos_tagging_)

    word_cloud(df)  # draws word cloud

    fd_ = frequency_distribution(df)
    print("10 Most common frequency distribution as default", fd_.most_common)
    print("frequency distribution", fd_.all)
    print(fd_.chart)  # draws 10 most common distribution as default

    df_2 = pipeline_2()
    cv = count_vectorizer(df_2['Review'])

    print("Count Vectorizer BOW", cv.bow)
    print("Count Vectorizer's feature names", cv.feature_names)
    print("Count Vectorizer's vocabularies", cv.vocab)

    tf_idf = terms_frequency(df_2['Review'])

    print("TF-IDF output", tf_idf.output)
    print("TF-IDF feature names", tf_idf.feature_names)
    print("TF-IDF vocabularies", tf_idf.vocab)


def _building():
    df = pipeline_1()
    reviews = df['Review'].array
    name = df['Department Name'].array

    X_train, X_test, y_train, y_test = train_test_split(reviews, name,
                                                        test_size=0.20,
                                                        random_state=0)
    fd = frequency_distribution(df)

    tokenizer = Tokenizer(num_words=1500000)
    tokenizer.fit_on_texts(reviews)

    assert len(fd.all) == len(tokenizer.word_index)

    x_train_tokens = tokenizer.texts_to_sequences(X_train)
    x_test_tokens = tokenizer.texts_to_sequences(X_test)

    num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
    num_tokens = np.array(num_tokens)
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    np.sum(num_tokens < max_tokens) / len(num_tokens)
    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
    x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

    model, history = build_model(x_train_pad, y_train, max_tokens)
    result = model.evaluate(x_test_pad, y_test)

    assert isinstance(result, list)

    plt.figure(figsize=(15, 10))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    # @TODO Implement cosine similarity anf F1
