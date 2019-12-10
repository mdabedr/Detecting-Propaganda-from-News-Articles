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
from model.lstm_tagger import LSTMTagger
from tqdm import tqdm


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
tag_to_ix = {"O": 0, "I": 1}


class TrainDataLoader(Dataset):
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


class TestDataLoader(Dataset):
    def __init__(self, X, pad_len, max_size=None):
        self.source = []
        self.source_len = []
        self.pad_len = pad_len
        self.read_data(X, y)

    def read_data(self, X):
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
        return len(self.source)

    def __getitem__(self, idx):
        return torch.LongTensor(self.source[idx]), \
               torch.LongTensor([self.source_len[idx]])


def train(X_train, y_train, X_dev, y_dev):

    vocab_size = VOCAB_SIZE

    print('NUM of VOCAB' + str(vocab_size))
    train_data = TrainDataLoader(X_train, y_train, PAD_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    dev_data = TrainDataLoader(X_dev, y_dev, PAD_LEN)
    dev_loader = DataLoader(dev_data, batch_size=int(BATCH_SIZE/3)+2, shuffle=False)


    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, 2, BATCH_SIZE)

    model.load_embedding(tokenizer.get_embeddings())
    # multi-GPU
    # model = nn.DataParallel(model)
    model.cuda()

    loss_criterion = nn.CrossEntropyLoss()  #

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    es = EarlyStopping(patience=PATIENCE)
    old_model = None
    for epoch in range(1, MAX_EPOCH):
        print('Epoch: ' + str(epoch) + '===================================')
        train_loss = 0
        model.train()
        for i, (data, seq_len, label) in tqdm(enumerate(train_loader),
                                              total=len(train_data)/BATCH_SIZE):
            optimizer.zero_grad()
            y_pred = model(data.cuda(), seq_len)
            loss = loss_criterion(y_pred, label.view(-1).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPS)
            optimizer.step()
            train_loss += loss.data.cpu().numpy() * data.shape[0]
            del y_pred, loss

        test_loss = 0
        model.eval()
        y_eval_list = []
        for _, (_data, _seq_len, _label) in enumerate(dev_loader):
            with torch.no_grad():
                y_pred = model(_data.cuda(), _seq_len)
                loss = loss_criterion(y_pred, _label.view(-1).cuda())
                test_loss += loss.data.cpu().numpy() * _data.shape[0]
                y_eval_list.append(y_pred.data.cpu().numpy())
                del y_pred, loss

        y_eval_list = np.argmax(np.concatenate(y_eval_list, axis=0), axis=1)

        print("Train Loss: ", str(train_loss / len(train_data)), " Evaluation: ", str(test_loss / len(dev_data)))


def main():
    training_data, A, X_dev, Y_dev, Atest = pickle.load(open("FromSahir'sCode.pk", "rb"))
    X_train = [x[0] for x in training_data]
    y_train = [x[1] for x in training_data]
    tokenizer.build_tokenizer(X_train + X_dev, need_split=False)
    tokenizer.build_embedding(GLOVE_PATH, save_pkl=True, dataset_name='tmpmt')
    train(X_train, y_train, X_dev, y_dev)


    tmp = 0


if __name__ == '__main__':
    main()