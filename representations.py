import nltk
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def bag_of_words(tweets):
    """
    Computes the bag-of-word vectors for the tweets.
    
    Args: 
    tweets (pandas Series)
    
    Returns:
    features (array): term document matrix; array of shape (n_samples, vocab_size)
    
    """
    vec = CountVectorizer(analyzer = 'word', max_features = 10000)
    features = vec.fit_transform(tweets)
    return features


def tf_idf(tweets):
    """
    Computes word embeddings by applying the tf-idf weighing scheme.
    
    Args: 
    tweets (pandas Series)
    
    Returns:
    features (array): ft-idf matrix; array of shape (n_samples, vocab_size)
    
    """
    tf = TfidfVectorizer(analyzer = 'word', ngram_range=(1,1), min_df = 1, max_features = 10000)
    features = tf.fit_transform(tweets)
    return features

def glove(filename):
    """
    Performs training of 200D GloVe embeddings by calling the appropriate helper functions.

    Args:
    filename (string): path to a file with (pre-processed) tweets

    Creates:
    embeddings.txt: each line contains a word that occurs at least 5
                times in the data and its 200-dim GloVe embedding
    """

    os.system(BUILD_VOCAB + ' ' + filename)
    os.system(CUT_VOCAB)
    os.system('python3 ' + PICKLE_VOCAB)
    os.system('python3 ' + COOC + ' ' + filename)
    os.system('python3 ' + GLOVE_SOL)
    return

def glove_pretrained(path):
    """
    Loads pre-trained GloVe embeddings from a file.

    Args:
    path (string): path to the file with GloVe embeddings; each line consists
                of a token followed by the embedding vector components, separated
                by a whitespace

    Returns:
    embeddings (dictionary): dictionary where the keys are the words and
                the vector embeddings are the values
    """

    embeddings = {}

    with open(path, 'r') as file:
        for line in file:
            vals = line.rstrip().split(' ')
            word = vals[0]
            vector = np.asarray(vals[1:], "float32")
            embeddings[word] = vector

    embeddings['<unk>'] = np.mean(np.array(list(embeddings.values())), axis=0)

    return embeddings

def map_glove(tweets, embeddings):
    """
    Maps tweet tokens to embeddings and takes their average.

    Args:
    tweets (pandas series): tweet bodies as strings
    embeddings (dictionary): the keys are tokens and values are the vector embeddings

    Returns:
    features (numpy array): resulting features obtained by averaging the individual embeddings
            of each token in a tweet
    """

    n = len(list(embeddings.values())[0])
    empty = np.zeros(n)
    notfound = embeddings['<unk>']

    features = []

    for tweet in tweets:
        vecs = [embeddings.get(word, notfound) for word in tweet.split()]
        features.append(np.mean(vecs, axis = 0) if len(vecs) > 0 else empty)

    return np.array(features)
