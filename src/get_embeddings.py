from load import *
from preprocessing import *
from representations import *
from consts import *

def generate_preprocessed():
    """
    Running this will generate files with preprocessed training and test data.
    This can save time as they take long to process.
    """
    tweets_raw, test_data_raw = load_data(full = True, preprocessed=False)
    tweets = tweets_raw.copy()

    print('Processing training data...')
    tweets['body'] = preprocess_data(tweets['body'])
    tweets['body'].to_csv(PREPROCESSED_DATA_FULL, header=None, index=None, sep='\n')

    print('Processing test data...')
    test = test_data_raw.copy()
    test['body'] = preprocess_data(test['body'])
    test['body'].to_csv(PREPROCESSED_DATA_TEST, header=None, index=None, sep='\n')


if __name__ == "__main__":
    generate_preprocessed()

    # train embeddings based on our corpus of tweets
    print('Training GloVe embeddings...')
    glove(PREPROCESSED_DATA_FULL)
