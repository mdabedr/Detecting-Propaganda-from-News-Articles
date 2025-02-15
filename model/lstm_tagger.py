"""
Created by Chenyang Huang, 2018
Modified from https://github.com/jiangqy/LSTM-Classification-Pytorch/blob/master/utils/LSTMClassifier.py
Original author: Qinyuan Jiang, 2017
"""

import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math
import os
from torchcrf import CRF

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 label_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bidirectional = True
        self.num_layers = 2
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional, dropout=0.25)
        # self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)
        if self.bidirectional:
            self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        else:
            self.hidden2label = nn.Linear(hidden_dim, label_size)

        # self.last_layer = nn.Linear(hidden_dim, label_size * 100)
        # loss
        #weight_mask = torch.ones(vocab_size).cuda()
        #weight_mask[word2id['<pad>']] = 0
        # self.loss_criterion = nn.BCELoss()

    @staticmethod
    def sort_batch(batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        rever_sort = np.zeros(len(seq_lengths))
        for i, l in enumerate(perm_idx):
            rever_sort[l] = i
        return seq_tensor, seq_lengths, rever_sort.astype(int)

    def init_hidden(self, x):
        batch_size = x.size(0)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        else:
            h0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.hidden_dim), requires_grad=False).cuda()
        return (h0, c0)

    def forward(self, x, seq_len):

        x, seq_len, reverse_idx = self.sort_batch(x, seq_len.view(-1))

        embedded = self.embeddings(x)
        embedded = self.dropout(embedded)

        # embedded = torch.nn.BatchNorm1d(embedded)  # batch normalization
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, seq_len.cpu().numpy(), batch_first=True)
        hidden = self.init_hidden(x)
        packed_output, hidden = self.lstm(packed_input, hidden)
        lstm_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # global attention
            # output = lstm_out
            # seq_len = tor

        bs, pad_len, lstm_dim = lstm_out.size()
        lstm_out = lstm_out.reshape(-1, lstm_dim)
        y_pred = self.hidden2label(lstm_out)
        y_pred = y_pred.reshape(bs, pad_len, -1)

        y_pred = y_pred[reverse_idx]
        unpacked_len = unpacked_len[reverse_idx]

        y_pred = [y[:l] for y, l in zip(y_pred, unpacked_len)]

        return torch.cat(y_pred, dim=0)

    def load_embedding(self, emb):
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
