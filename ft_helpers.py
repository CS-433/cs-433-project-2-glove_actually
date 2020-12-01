import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = './data/'

def load_data(full = True):
    """
    Loads the Twitter data in the fasttext format
    
    Args:
    full (bool): if False, loads only a part of the data
    
    Returns:
    tweets (pandas dataframe): positive and negative tweets with labels
    test: unlabelled data for testing
    """
    
    FULL = ''  
    if full:
        FULL = '_full'
        
    POS_TWEETS = DATA_PATH + 'train_pos' + FULL + '.txt'
    NEG_TWEETS = DATA_PATH + 'train_neg' + FULL + '.txt'
    TEST_DATA = DATA_PATH + 'test_data.txt'
    
    with open(POS_TWEETS) as file:
        pos_tweets_data = [line.rstrip() for line in file]
    pos_tweets = pd.DataFrame(pos_tweets_data, columns=['body'])
    pos_tweets['label'] = "__label__happyface"
    
    with open(NEG_TWEETS) as file:
        neg_tweets_data = [line.rstrip() for line in file]
    neg_tweets = pd.DataFrame(neg_tweets_data, columns=['body'])
    neg_tweets['label'] = "__label__sadface"

    with open(TEST_DATA) as file:
        # removes id at the same time
        test_data = [line.rstrip().split(',', 1)[1] for line in file]

    test = pd.DataFrame(test_data, columns=['body'])

    # merge positive and negative datasets
    tweets = pd.concat([pos_tweets, neg_tweets], axis = 0)
    
    return tweets, test


def reindex_df(df):
    """
    Reindexes a given dataframe for the FastText format (i.e. label first, body second)
    
    Args:
    df (pandas dataframe): tweets with columns indexed as ['body', 'label']
    
    Returns:
    df_reindexed (pandas dataframe): tweets with columns indexed as ['body', 'label']
    """
    
    columnsTitles = ['label', 'body'] 
    df_reindexed = df.reindex(columns=columnsTitles)
    
    return df_reindexed


def train_val_split(attr,label,val_size,random_state):
    """
    Splits the train data into train and validation sets
    
    Args:
    attr (pandas series): tweet body
    label (pandas series): tweet label
    val_size (float): ratio of validation set
    random_state (int): controls the shuffling applied to the data before applying the split
    
    Returns:
    train (pandas dataframe): train set containing tweet label and body
    val (pandas dataframe): validation set containing tweet label and body
    """
    
    X_train, X_val, y_train, y_val = train_test_split(attr, label, test_size=val_size, random_state=random_state)
    
    # merge tweet body and label
    train = pd.concat([X_train, y_train], axis = 1)
    val = pd.concat([X_val, y_val], axis = 1)
    
    return train, val


