import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from utils import remove_punctuation, raw_data_empty_info, contractions, emoticons

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

df = pd.read_csv('E-Commerce_Reviews.csv')
raw_data_analysis = raw_data_empty_info(df)

# load emoticons and  contractions
emoticons = emoticons()
contractions = contractions()

if raw_data_analysis.review:
    # Removing empty reviews row if there are
    df.dropna(subset=['Review'], inplace=True)
    df['Title'].fillna(value="No Title", inplace=True)

df['Review'] = df['Review'].apply(lambda x: " ".join([emoticons[i] if i in emoticons else i for i in x.split()]))
df['Review'] = df['Review'].apply(lambda x: " ".join([contractions[i] if i in contractions else i for i in x.split()]))
df['Review'] = df['Review'].apply(lambda x: remove_punctuation(x))
# df['Review'] = df['Review'].apply(lambda x: tokenizer.tokenize(x.lower()))
df['Review'] = df['Review'].apply(lambda x: [w for w in x if not w in set(stopwords.words('english'))])
# df['Review'] = df['Review'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
# df['Review'] = df['Review'].apply(lambda x: " ".join([stemmer.stem(i) for i in x]))


def pipeline_1():
    """Tokenized & Lemmatized"""
    df['Review'] = df['Review'].apply(lambda x: tokenizer.tokenize(x.lower()))
    df['Review'] = df['Review'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    return df


def pipeline_2():
    """Lemmatized"""
    df['Review'] = df['Review'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    return df


def pipeline_3():
    """Tokenized & Stemmed"""
    df['Review'] = df['Review'].apply(lambda x: tokenizer.tokenize(x.lower()))
    df['Review'] = df['Review'].apply(lambda x: " ".join([stemmer.stem(i) for i in x]))
    return df
