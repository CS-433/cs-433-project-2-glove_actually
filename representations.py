import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_vocabulary(tweets, thres = 5):
    """
    Builds a vocabulary of words that occur at least 5 times.
    
    Args:
    tweets (pandas series): strings of pre-processed tweets 
    thres (int): required minimum number of occurences to be included in the vocabulary
    
    Returns:
    vocab ():
    """
    # imlementation using CountVectorizer is much faster
    vec = CountVectorizer()  
    words = vec.fit_transform(tweets)
    vocab = pd.DataFrame(words.sum(axis=0), columns=vec.get_feature_names()).transpose()
    vocab.columns = ['freq']
    vocab_cut = vocab[vocab.freq >= thres]
    
    return vocab_cut.index.tolist()

def bag_of_words(tweets):
    vec = CountVectorizer(analyzer = 'word')  
    features = vec.fit_transform(tweets)
    return features
    

def tf_idf(tweets):
    tf = TfidfVectorizer(analyzer = 'word', ngram_range=(1,1), min_df = 1)
    features = tf.fit_transform(tweets)
    return features
    
def glove():
    raise NotImplementedError
    
def glove_pretrained(path):
    """
    Loads pre-trained GloVe embeddings from a file.
    
    Args: 
    path (string): filename
    
    Returns: 
    embeddings (dictionary): dictionary where the keys are the words and
                the vector embeddings are the values
    """

    embeddings = {} 
    
    with open(path) as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
            
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
    notfound = np.zeros(n)
    
    features = []
    
    for tweet in tweets:
        vecs = [embeddings.get(word, notfound) for word in tweet.split()]
        features.append(np.mean(vecs, axis = 0) if len(vecs) > 0 else notfound)
        
    return np.array(features)
                       
    
    
    