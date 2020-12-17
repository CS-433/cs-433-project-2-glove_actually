import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    """
     Defines a class for a PyTorch recurrent neural network.
     
     Args:
     rnn_type (string): defines gated architecture – 'lstm' or 'gru'
     vocab_size (int): size of the vocabulary of training data
     embed_dim (int): dimensionality of embeddings
     hidden_dim (int): number of units in the feed-forward part of network
     output_dim (int): number of output units
     dropout (float): proportion of nodes to drop
     n_layers (int): number of LSTM or GRU layers
     bi (bool): bidirectional RNN if True
     
    """

    def __init__(self, rnn_type, vocab_size, embed_dim, hidden_dim, output_dim, dropout, n_layers, bi):

        super().__init__()

        self.bi = bi
        self.type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_directions = 2 if bi else 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # initialize the look-up table
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if rnn_type == 'lstm':
            self.mod = nn.LSTM
        if rnn_type == 'gru':
            self.mod = nn.GRU

        self.rnn = self.mod(embed_dim,
                           hidden_dim,
                           num_layers = n_layers,
                           bidirectional = bi,
                           dropout = dropout,
                           batch_first = True)

        self.dropout = nn.Dropout(p=dropout)

        # dense layer:
        n_units = 2*hidden_dim if bi else hidden_dim
        self.fc = nn.Linear(n_units, output_dim)

    def forward(self, text, text_len):

        embedded = self.embedding(text) # (batch_size x text_len x embed_dim) tensor

        # packed sequence
        packed = nn.utils.rnn.pack_padded_sequence(embedded, text_len.cpu(), batch_first = True, enforce_sorted=False).to(self.device)

        if self.type == 'lstm':
            packed_output, (h, c) = self.rnn(packed)

        if self.type == 'gru':
            packed_output, h = self.rnn(packed)

        # h : n_lstm_layers*n_directions x batch_size x hidden_dim
        h = h.to(self.device)

        batch_size = len(text)
        h = h.view(self.n_layers, self.n_directions, batch_size, self.hidden_dim)
        h_out = h[-1]
        # if bidirectional concatenate final forward + backward hidden state
        h_out = torch.cat((h_out[-2,:,:], h_out[-1,:,:]), dim = 1) if self.bi else h_out[-1,:,:]
        # h_out: batch_size x hidden_dim*2

        dense = self.fc(h_out) # dense: batch_size x output_dim
        dense = self.dropout(dense)

        return dense
