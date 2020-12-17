from rnn import *
from predict_helpers import *
from load import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
EMBEDDING_DIM = 200
prep = False # no preprocessing

LSTM_WEIGHTS = '../data/weights_lstm.pt'
GRU_WEIGHTS = '../data/weights_gru.pt'

OUT_PATH_LSTM = '../data/lstm_predictions.csv'
OUT_PATH_GRU = '../data/gru_predictions.csv'
OUT_PATH_FT = '../data/ft_predictions.csv'
OUT_PATH = 'submission.csv'

def generate_best_models():
    TEXT, train_iter, val_iter, test = load_dataset(BATCH_SIZE, EMBEDDING_DIM, prep=prep)

    vocab_size = len(TEXT.vocab)

    print('Re-training fastText and making predictions...')

    train_predict_ft(ngrams=3, lr=0.1, epoch=2, prep = prep, OUT_PATH = OUT_PATH_FT, prob=True)


    model_lstm = RNNClassifier(rnn_type = 'lstm',
                  vocab_size = vocab_size,
                  embed_dim = EMBEDDING_DIM,
                  hidden_dim = 250,
                  output_dim = 1,
                  n_layers = 2,
                  dropout = 0.01,
                  bi = True).to(device)

    print(model_lstm)

    predict_pretrained_nn(model_lstm, LSTM_WEIGHTS, test, TEXT, OUT_PATH_LSTM, prob=True)

    model_gru = RNNClassifier(rnn_type = 'gru',
              vocab_size = vocab_size,
              embed_dim = EMBEDDING_DIM,
              hidden_dim = 250,
              output_dim = 1,
              n_layers = 2,
              dropout = 0.1,
              bi = True).to(device)

    print(model_gru)

    predict_pretrained_nn(model_gru, GRU_WEIGHTS, test, TEXT, OUT_PATH_GRU, prob=True)


if __name__ == "__main__":

    print('1: produce best submission')
    print('2: train best LSTM')
    print('3: train best GRU')
    val = input("Choose option: ")
    print(val)

    if int(val) == 1:
        print('Producing last submission...')
        generate_best_models()
        print('Combining predictions from 3 models...')
        ensemble(OUT_PATH_LSTM, OUT_PATH_GRU, OUT_PATH_FT, OUT_PATH)
    elif int(val) == 2:
        e = input("Epochs: ")
        print('Training best LSTM...')
        train_rnn(rnn_type='lstm', n_layers=2, hidden_dim=250, dropout=0.01, prep=False, num_epochs = int(e))

    elif int(val) == 3:
        e = input("Epochs: ")
        if e > 0:
            print('Training best GRU...')
            train_rnn(rnn_type='gru', n_layers=2, hidden_dim=250, dropout=0.1, prep=False, num_epochs = int(e))
        else:
            print('non-negative integers only')
    else:
        print('Invalid input')
