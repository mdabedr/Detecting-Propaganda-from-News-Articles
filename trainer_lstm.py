from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
import pickle
from model import lstm_tagger
from utils.tokenizer import GloveTokenizer
from copy import deepcopy
from utils.early_stopping import EarlyStopping
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# hyper-parameters
MAX_EPOCH = 300
NUM_FOLDS = 5
EMBEDDING_DIM = 300
PAD_LEN = 50
MIN_LEN_DATA = 3
BATCH_SIZE = 16
CLIPS = 0.888  # ref. I Ching, 750 BC
HIDDEN_DIM = 800
VOCAB_SIZE = 20000
LEARNING_RATE = 5e-4
PATIENCE = 3
USE_ATT = False
BIDIRECTIONAL = True
GLOVE_PATH = '/remote/eureka1/chuang8/glove.840B.300d.txt'
tmp = 0
tokenizer = GloveTokenizer()


class TestDataLoader(Dataset):
    def __init__(self, X, y, pad_len, max_size=None):
        self.source = []
        self.source_len = []
        self.target = y
        self.pad_len = pad_len
        self.read_data(X, y)

    def read_data(self, X, y):
        for src in X:
            src = tokenizer.encode_ids(src)
            if len(src) < self.pad_len:
                src_len = len(src)
                src = src + [0] * (self.pad_len - len(src))
            else:
                src = src[:self.pad_len]
                src_len = self.pad_len

            self.source_len.append(src_len)
            self.source.append(src)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.LongTensor(self.source[idx]), \
               torch.LongTensor([self.source_len[idx]]), \
               torch.LongTensor([self.target[idx]])


def main():
    training_data, A, X_test, Y_test, Atest = pickle.load(open("FromSahir'sCode.pk", "rb"))
    X_train = [x[0] for x in training_data]
    y_train = [x[1] for x in training_data]
    tokenizer.build_tokenizer(X_train + X_test, need_split=False)
    tokenizer.build_embedding(GLOVE_PATH, save_pkl=True, dataset_name='tmpmt')
    tmp = 0


if __name__ == '__main__':
    main()