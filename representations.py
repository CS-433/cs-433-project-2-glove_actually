import nltk
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

def bag_of_words():
    raise NotImplementedError
    

def tf_idf():
    raise NotImplementedError
    

def glove():
    raise NotImplementedError
    
def glove_pretrained():
    raise NotImplementedError