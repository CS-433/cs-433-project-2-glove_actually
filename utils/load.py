import pandas as pd

DATA_PATH = './data/'

def load_data(full = True):
    """
    Loads the Twitter data.
    
    Args:
    full (bool): if False, loads only a part of the data
    
    Returns:
    tweets (pandas dataframe): positive and negative tweets with labels
    test_data: unlabelled data for testing
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
    pos_tweets['label'] = 1

    with open(NEG_TWEETS) as file:
        neg_tweets_data = [line.rstrip() for line in file]
    neg_tweets = pd.DataFrame(neg_tweets_data, columns=['body'])
    neg_tweets['label'] = -1

    with open(TEST_DATA) as file:
        # removes id at the same time
        test_data = [line.rstrip().split(',', 1)[1] for line in file]

    test_data = pd.DataFrame(test_data, columns=['body'])

    # merge positive and negative datasets
    tweets = pd.concat([pos_tweets, neg_tweets], axis = 0)
    
    return tweets, test_data
