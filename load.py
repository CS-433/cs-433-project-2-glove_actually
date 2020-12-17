import pandas as pd
import numpy as np
import csv

DATA_PATH = '../data/'
POS_TWEETS = DATA_PATH + 'train_pos.txt'
NEG_TWEETS = DATA_PATH + 'train_neg.txt'
POS_TWEETS_FULL = DATA_PATH + 'train_pos_full.txt'
NEG_TWEETS_FULL = DATA_PATH + 'train_neg_full.txt'
TEST_DATA = DATA_PATH + 'test_data.txt'
PREPROCESSED_DATA = DATA_PATH + 'preprocessed_tweets.txt'
PREPROCESSED_DATA_FULL = DATA_PATH + 'preprocessed_tweets_full.txt'
PREPROCESSED_DATA_TEST = DATA_PATH + 'preprocessed_tweets_test.txt'

def load_data(full = True, preprocessed = False):
    """
    Loads the Twitter data.

    Args:
    full (bool): if False, load only a part of the data
    preprocessed (bool): if True, load preprocessed tweets

    Returns:
    tweets (pandas dataframe): positive and negative tweets with labels
                            (columns: 'body', 'label')
    test_data: unlabelled data for testing
                            (columns: 'id', 'body')
    """
    if full:
        pos = POS_TWEETS_FULL
        neg = NEG_TWEETS_FULL
        prep = PREPROCESSED_DATA_FULL
    else:
        pos = POS_TWEETS
        neg = NEG_TWEETS
        prep = PREPROCESSED_DATA

    with open(pos) as file:
        pos_tweets_data = [line.rstrip() for line in file]
    pos_tweets = pd.DataFrame(pos_tweets_data, columns=['body'])
    pos_tweets['label'] = 1

    with open(neg) as file:
        neg_tweets_data = [line.rstrip() for line in file]
    neg_tweets = pd.DataFrame(neg_tweets_data, columns=['body'])
    neg_tweets['label'] = 0

    with open(TEST_DATA) as file:
        test_bodies = []
        test_id = []
        for line in file:
            split = line.rstrip().split(',', 1)
            test_bodies.append(split[1])
            test_id.append(split[0])

        features = {'id': np.array(test_id).astype(int), 'body': test_bodies}

    test_data = pd.DataFrame(features)

    # merge positive and negative datasets
    tweets = pd.concat([pos_tweets, neg_tweets], axis = 0)

    if preprocessed:
        tweets['body'] = load_from_txt(prep)
        test_data['body'] = load_from_txt(PREPROCESSED_DATA_TEST)

    return tweets, test_data

def load_from_txt(path):
    """
    Reads in data line by line.
    """
    with open(path) as file:
        data = [line.rstrip() for line in file]
    return data

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    # negative class has to be labelled -1 on AIcrowd
    y_pred[y_pred == 0] = -1

    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def create_csv_submission_prob(ids, y_pred, y_prob, name):
    """
    Creates an output file in .csv format with probabilities
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               y_prob (probability of class 1)
               name (string name of .csv output file to be created)
    """
    # negative class has to be labelled -1 on AIcrowd
    y_pred[y_pred == 0] = -1

    df = pd.DataFrame({'id': ids, 'label': y_pred, 'prob': y_prob})
    df.to_csv(name, sep=",", index=False)
