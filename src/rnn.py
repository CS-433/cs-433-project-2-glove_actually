from load import *
from rnn_classifier import RNNClassifier
import random
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.vocab import Vectors, GloVe

warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
EMBEDDING_DIM = 200

def accuracy(y_pred, y_test):
    """
    Calculates binary classification accuracy.

    Args:
    y_pred (1D tensor): predicted classes
    y_test (1D tensor): ground truth

    Returns:
    acc (float): binary accuracy
    """
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    true = (y_pred_tag == y_test).sum().float()
    acc = true/y_test.shape[0]
    return acc

def load_dataset(batch_size, embedding_dim, prep=False):
    """
    Loads the data and defines iterators for training.

    Args:
    batch_size (int)
    embedding_dim (int)
    prep (bool): whether to use preprocessed data or not
    """

    print("Loading data...")

    tweets, test = load_data(full = True, preprocessed = prep)
    tweets.to_csv('rnn_data.csv', index = False, header = True)
    # it is easier to import data using TabularDataset from .csv files than from pandas

    # initialize fields for tweet text and labels
    TEXT = data.Field(tokenize=lambda x: x.split(), batch_first = True, include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float, batch_first = True)

    fields = [('text', TEXT),('label', LABEL)]
    training_data = data.TabularDataset(path = 'rnn_data.csv', format = 'csv', fields = fields, skip_header = True)

    # split data for training
    train_data, val_data = training_data.split(split_ratio=0.8, random_state = random.seed(0))

    # map words to pre-trained embeddings
    print('Preparing embeddings...')
    TEXT.build_vocab(train_data, min_freq=5, vectors = GloVe(name='twitter.27B', dim=embedding_dim))
    LABEL.build_vocab(train_data)

    # forms batches so that tweets with similar length are in the same batch
    train_iter, val_iter = data.BucketIterator.splits((train_data, val_data),
                                                      batch_size = batch_size,
                                                      sort_key = lambda x: len(x.text),
                                                      sort_within_batch = True,
                                                      device = device,
                                                      repeat=False,
                                                      shuffle=True)

    print ("Size of vocabulary: " + str(len(TEXT.vocab)))

    return TEXT, train_iter, val_iter, test

def train(model, iterator, optimizer, criterion):
    # reset every epoch
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_len = batch.text
        text = text.to(device)
        pred = model(text, text_len).squeeze()  # 1D tensor

        label = batch.label.to(device)
        loss = criterion(pred, label)
        acc = accuracy(pred, label)

        # backprop step + gradient computation
        loss.backward()

        # update weights
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()


    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_len = batch.text
            text = text.to(device)
            pred = model(text, text_len).squeeze()

            label = batch.label.to(device)
            loss = criterion(pred, label)
            acc = accuracy(pred, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def predict(model, tweet, TEXT):
    """
    Make predictions for test tweets.
    """

    tokens = [t for t in tweet.split()]
    idx = [TEXT.vocab.stoi[t] for t in tokens]

    tensor = torch.LongTensor(idx).to(device)
    tensor = tensor.unsqueeze(1).T

    len_tensor = torch.LongTensor([len(idx)])
    pred = model(tensor, len_tensor)
    prob = torch.sigmoid(-pred)

    return torch.round(prob).item(), prob.item()


def train_rnn(rnn_type, n_layers, hidden_dim, dropout, prep=False, num_epochs = 10):
    TEXT, train_iter, val_iter, test = load_dataset(BATCH_SIZE, EMBEDDING_DIM, prep=prep)

    vocab_size = len(TEXT.vocab)
    embed_dim = EMBEDDING_DIM
    output_dim = 1

    model = RNNClassifier(rnn_type = rnn_type,
                  vocab_size = vocab_size,
                  embed_dim = embed_dim,
                  hidden_dim = hidden_dim,
                  output_dim = output_dim,
                  n_layers = n_layers,
                  dropout = dropout,
                  bi = True).to(device)

    print(model)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data = pretrained_embeddings.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_loss = float('inf')

    for epoch in range(num_epochs):

        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_iter, criterion)

        # save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '..data/weights.pt')

        print('\tEpoch {}: Train Loss: {:.3f} | Val. Loss: {:.3f}  | Train Acc: {:.2f}% |  Val. Acc: {:.2f}%'.format(epoch,
                                                                                                                    train_loss,
                                                                                                                    val_loss,
                                                                                                                    train_acc*100,
                                                                                                                    val_acc*100))

    # load saved weights and predict test data
    model.load_state_dict(torch.load('../data/weights.pt', map_location=device))
    model.eval()
    pred = []

    for tweet in test.body:
        l, p = predict(model, tweet, TEXT)
        pred.append(l)
    create_csv_submission(test.id, np.array(pred), '../data/submission.csv')
