from gensim.models import KeyedVectors, Word2Vec
from utils import html_to_str, render
from pandas import DataFrame
from sklearn.metrics import precision_recall_fscore_support

# load vectors
cbow_vectors = KeyedVectors.load("cbow_vector.bin")
subword_vectors = KeyedVectors.load("subword_vector.bin")
skipgram_vectors = KeyedVectors.load("skipgram_vector.bin")

# load models
cbow_model = Word2Vec.load("cbow.model")
fasttext_model = Word2Vec.load("fasttext.model")
skipgram_model = Word2Vec.load("skipgram.model")


def get_topn_similarities(positive: list, topn=10):
    topn_cbow = cbow_model.wv.most_similar(positive=positive, topn=topn)
    topn_subword = fasttext_model.wv.most_similar(positive=positive, topn=topn)
    topn_skipgram = skipgram_model.wv.most_similar(positive=positive, topn=topn)

    return html_to_str(
        DataFrame(topn_cbow, columns=['cbow similarity', 'cosine_sim']),
        DataFrame(topn_skipgram, columns=['skipgram similarity', 'cosine_sim']),
        DataFrame(topn_subword, columns=['fasttext similarity', 'cosine_sim']))


def get_word_sim(w1, w2):
    sim_cbow = cbow_model.wv.similarity(w1=w1, w2=w2)
    sim_skipgram = skipgram_model.wv.similarity(w1=w1, w2=w2)
    sim_subword = fasttext_model.wv.similarity(w1=w1, w2=w2)

    return {"a_word": w1, "b_word": w2, "score_cbow": sim_cbow, "score_skipgram": sim_skipgram,
            "score_fasttext": sim_subword}


def get_prf(x_true, x_pred):
    per_class_prf = precision_recall_fscore_support(x_true, x_pred, average='binary')
    return {"precision": per_class_prf[0], "recall": per_class_prf[1], "fscore": per_class_prf[2]}


# topn similarities
w1 = ['fit', 'love']
data = get_topn_similarities(w1, topn=10)

# word pairs & computing the similarities
word_pairs = [['summer', 'beautiful'], ['dress', 'good'], ['wear', 'well'], ["fit", "size"],
              ["perfect", "jean"], ["like", "dress"], ["really", "comfortable"], ["perfect", "colour"],
              ["perfect", "color"], ["flattering", "loved"]]
results = [get_word_sim(i[0], i[1]) for i in word_pairs]  # get similarity
sim_df = DataFrame(results)
similarity = html_to_str(sim_df)

# @TODO refactoring for the below lines
score_cbow_skipgram = []
for i, j in zip(sim_df['score_cbow'], sim_df['score_skipgram']):
    if i and j > 0.6:
        score_cbow_skipgram.append([1, 1])
    elif i > 0.6 > j:
        score_cbow_skipgram.append([1, 0])
    elif i < 0.6 < j:
        score_cbow_skipgram.append([0, 1])
    elif i < 0.6 and j < 0.6:
        score_cbow_skipgram.append([0, 0])

cos_sim = [[0, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 1]]

cbow_skip_sim = DataFrame(score_cbow_skipgram, columns=["cbow", "skipgram"])
cos_sim_sim = DataFrame(cos_sim, columns=["cbow", "skipgram"])

prf = list()
prf.append(get_prf(cbow_skip_sim["a"].values, cbow_skip_sim["b"].values))
prf.append(get_prf(cos_sim_sim["a"].values, cos_sim_sim["b"].values))

prf = html_to_str(DataFrame(prf))

# demonstration || easy to integrate to web service or share with users via GUI
render(html_content=data, file_name="data")
render(html_content=similarity, file_name="sim")
render(html_content=prf, file_name="prf")
