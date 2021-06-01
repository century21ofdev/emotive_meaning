import logging
from preprocessing import df
from gensim.utils import simple_preprocess
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def genex_reviews():
    for i in df['Review'].values:
        yield simple_preprocess(i)


reviews = df['Review'].values
# reviews = genex_reviews()

# default window size is 5 (two words before and two words after the input word, in addition to the input word itself)
# training the cbow (Continuous Bag of Words) model
model_cbow = Word2Vec(reviews, window=10, min_count=2, workers=10)
model_cbow.train(reviews, total_examples=len(reviews), epochs=50)

# training the char n-gram model (subword information) with fastText
model_subword = FastText(reviews, window=10, min_count=2, workers=10, min_n=3, max_n=6)
model_subword.train(reviews, total_examples=len(reviews), epochs=50)

# training the SkipGram model
model_skipgram = Word2Vec(reviews, window=10, min_count=2, workers=10, sg=1)
model_skipgram.train(reviews, total_examples=len(reviews), epochs=50)

# saving the models
model_cbow.save("cbow.model")
model_subword.save("fasttext.model")
model_skipgram.save("skipgram.model")

# saving the word vectors
model_cbow.wv.save("cbow_vector.bin")
model_subword.wv.save("subword_vector.bin")
model_skipgram.wv.save("skipgram_vector.bin")

print("cbow mean", model_cbow.cbow_mean)
print("cbow window", model_cbow.window)
print("cbow sample", model_cbow.sample)
print("cbow batch words", model_cbow.batch_words)

print("subword vector size", model_subword.vector_size)
print("subword cbow mean", model_subword.cbow_mean)
print("subword window", model_subword.window)
print("subword sample", model_subword.sample)
print("subword batch words", model_subword.batch_words)

print("skipgram vector size", model_skipgram.vector_size)
print("skipgram cbow mean", model_skipgram.cbow_mean)
print("skipgram sample", model_skipgram.sample)
print("skipgram batch words", model_skipgram.batch_words)
