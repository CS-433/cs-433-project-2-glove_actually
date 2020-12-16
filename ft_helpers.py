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


def reindex_dfs(CREATE_SUBMISSION, train, val, test):
    """
    Reindexes given dataframes for the FastText format (i.e. label first, body second)
    
    Args:
    CREATE_SUBMISSION (boolean): If False, does not reindex the test set
    train (pandas dataframe): tweets with columns indexed as ['body', 'label']
    val (pandas dataframe): tweets with columns indexed as ['body', 'label']
    test (pandas dataframe): tweets with columns indexed as ['body', 'label']
    
    Returns:
    train (pandas dataframe): tweets with columns indexed as ['label', 'body']
    val (pandas dataframe): tweets with columns indexed as ['label', 'body']
    test (pandas dataframe): tweets with columns indexed as ['label', 'body']
    """
    
    columnsTitles = ['label', 'body'] 
    
    train = train.reindex(columns=columnsTitles)
    val = val.reindex(columns=columnsTitles)
    
    if CREATE_SUBMISSION == True:
        test = test.reindex(columns=columnsTitles)
        
    return train, val, test


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


def create_csv_submission(model, test, filename):
    """
    Creates the .csv submission file for AICrowd, where the first column is the Id and the second column is the Prediction [-1,1]
    
    Args:
    model (fasttext model): Fasttext training model
    test (pandas dataframe): test set
    filename (string): file output name
    """
    
    # Make the predictions line by line and store them in the list of predictions
    predictions=[]
    
    for line in test['body']:
        pred_label=model.predict(line, k=-1, threshold=0.5)[0][0]
        predictions.append(pred_label)

    # Add the list to the dataframe
    test['Id'] = test.index + 1
    test['Prediction'] = predictions

    # Convert labels back to -1's and 1's
    test['Prediction'] = test['Prediction'].str.replace('__label__sadface','-1')
    test['Prediction'] = test['Prediction'].str.replace('__label__happyface','1')
    test = test[['Id', 'Prediction']]
     
    # Save dataframe into csv    
    test.to_csv(filename, sep=",", index=False)
    

def create_probabilities_csv(model, test, filename):
    """
    Creates a .csv file with the predictions and their probabilties
    
    Args:
    model (fasttext model): fasttext training model
    test (pandas dataframe): test set
    filename (string): file output name
    """
    
    # Make the predictions line by line and store them in the list of predictions and probabilities
    predictions=[]
    probabilities=[]
    
    for line in test['body']:
        pred_label=model.predict(line, k=3, threshold=0.5)[0][0]
        results = model.predict(line, k=3, threshold=0.5)

        predictions.append(pred_label)
        probabilities.append(results[1][0])
        
    # Add the list to the dataframe
    test['Id'] = test.index + 1
    test['Prediction'] = predictions
    test['Probability'] = probabilities

    # Convert labels back to -1's and 1's
    test['Prediction'] = test['Prediction'].str.replace('__label__sadface','-1')
    test['Prediction'] = test['Prediction'].str.replace('__label__happyface','1')
    test = test[['Id', 'Prediction', 'Probability']]    
    
    # Save dataframe into csv    
    test.to_csv(filename, sep=",", index=False)


def save_txt(train, val, test, SUBMISSION_POSTFIX, CREATE_SUBMISSION):
    """
    Saves the train, val, and test sets into .txt files (required for fasttext)
    
    Args:
    train (pandas dataframe): train set
    val (pandas dataframe): validation set
    test (pandas dataframe): test set
    SUBMISSION_POSTFIX (string): postfix of the .csv 
    CREATE_SUBMISSION (boolean): If False, does not create the .txt file for the test set
    
    Returns:
    predictions (int): binary predictions for each entry [-1,1]
    probabilities (float): prediction probabilities for each entry 
    """
    
    TRAIN_TXT = r'train' + SUBMISSION_POSTFIX + '.txt'
    VAL_TXT = r'val' + SUBMISSION_POSTFIX + '.txt'
    TEST_TXT = r'test' + SUBMISSION_POSTFIX + '.txt'

    np.savetxt(TRAIN_TXT, train.values, fmt='%s')
    np.savetxt(VAL_TXT, val.values, fmt='%s')

    if CREATE_SUBMISSION == True:
        np.savetxt(TEST_TXT, test.values, fmt='%s')
        
    return TRAIN_TXT, VAL_TXT, TEST_TXT


def sign(vote_sum):
    """
    Returns the majority vote of ensembles. If the sum of the predictions is negative, returns -1. Otherwise, returns 1.
    
    Args:
    vote_sum (float): sum of the votes of the ensemble models
    
    Returns:
    vote_sum (int): sum of the votes of the ensemble models as predictions
    """
        
    if(vote_sum < 0): 
        vote_sum = -1
    else:
        vote_sum = 1
        
    return vote_sum