from rnn import *
import ft_helpers as ft
from load import *
import fasttext
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_pretrained_nn(model, weights, test, TEXT, OUT_PATH, prob=False):
    """
    Uses the pre-trained weights to generate predictions.

    Args:
    model: a RNNClassifier object
    weights (path): a .pt file containing the saved model weights
    test (pandas dataframe): test data
    TEXT (field): contents of the tweets
    OUT_PATH (path): directory to which the output files will be written
    prob (bool): if true, the output includes probabilities of a tweet being classified as 1 (positive)

    """
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    pred = []
    prob = []

    for tweet in test.body:
        l, p = predict(model, tweet, TEXT)
        pred.append(l)
        prob.append(p)

    if prob:
        create_csv_submission_prob(test.id, np.array(pred), np.array(prob), OUT_PATH)
    else:
        create_csv_submission(test.id, np.array(pred), OUT_PATH)



def train_predict_ft(ngrams, lr, epoch, prep, OUT_PATH, prob=False):
    """
    Trains fastText with automatic hyperparameter optimization and makes predictions.

    Args:
    ngrams (int): specifies n for n-grams
    lr (float): learning rate
    epoch (int)
    prep (bool): whether to use pre-processed data or not
    OUT_PATH (path): directory to which the output files will be written
    prob (bool): if true, the output includes probabilities of a tweet being classified as 1 (positive)

    """

    print('We initialized the hyperparameters to save you some time.')
    print('You will get a warning about this as the function is primarily meant for auto-tuning.')

    tweets, test = ft.load_data(full=True, preprocessed=prep)
    train, val = ft.train_val_split(tweets['body'], tweets['label'], 0.2, 42)

    # Reindex the dataframes according to fasttext's format
    CREATE_SUBMISSION = True
    train, val, test = ft.reindex_dfs(CREATE_SUBMISSION, train, val, test)

    # Create data .txt files to be used in fasttext model
    train_txt, val_txt, test_txt = ft.save_txt(train, val, test, 'ft', CREATE_SUBMISSION)

    model_auto = fasttext.train_supervised(input = train_txt, lr=lr, epoch=epoch, wordNgrams=ngrams, autotuneValidationFile = val_txt)
    if prob:
        ft.create_probabilities_csv(model_auto, test, OUT_PATH)
    else:
        ft.create_csv_submission(model_auto, test, OUT_PATH)


def ensemble(lstm_probs_csv, gru_probs_csv, ft_probs_csv, OUT_FILE):
    """
    Combines predictions from 3 models by averaging the prediction probabilities.

    Args:
    lstm_probs_csv (csv file): predictions from LSTM in the from [id, label, prob]
    gru_probs_csv (csv file): predictions from GRU in the from [id, label, prob]
    ft_probs_csv (csv): predictions from fastText in the from [id, label, prob]

    """

    ft = pd.read_csv(ft_probs_csv, names=['id','label','prob'], header=0)
    neg_probs = ft[ft.label == -1]['prob'].values
    ft[ft.label == -1] = ft[ft.label == -1].assign(prob = 1-neg_probs)

    lstm = pd.read_csv(lstm_probs_csv, names=['id','label','prob'], header=0)
    gru = pd.read_csv(gru_probs_csv, names=['id','label','prob'], header=0)

    combined = (1/3)*ft.prob + (1/3)*lstm.prob + (1/3)*gru.prob
    combined_labels = [-1 if x < 0.5 else 1 for x in combined]
    result = pd.DataFrame({'prob': combined, 'label': combined_labels})

    create_csv_submission(result.index.values+1, result.label.values, OUT_FILE)
