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
from sklearn.metrics import classification_report
from utils.focalloss import FocalLoss

# hyper-parameters
MAX_EPOCH = 300
EMBEDDING_DIM = 300
PAD_LEN = 60
MIN_LEN_DATA = 3
BATCH_SIZE = 8
CLIPS = 0.888  # ref. I Ching, 750 BC
HIDDEN_DIM = 200
VOCAB_SIZE = 60000
LEARNING_RATE = 5e-4
PATIENCE = 3
USE_ATT = False
GLOVE_PATH = '/remote/eureka1/chuang8/glove.840B.300d.txt'
tokenizer = GloveTokenizer()
tag_to_ix = {"O": 0, "I": 1, "PAD": 999}


class TrainDataLoader(Dataset):
    def __init__(self, X, y, pad_len, max_size=None):
        self.source = []
        self.source_len = []
        self.target = []
        self.pad_len = pad_len
        self.read_data(X, y)

    def read_data(self, X, y):
        for src, trg in zip(X, y):
            src = tokenizer.encode_ids(src, split_tokens=False)
            trg = [tag_to_ix[x] for x in trg]
            assert len(src) == len(trg)
            if len(src) < self.pad_len:
                src_len = len(src)
                src = src + [0] * (self.pad_len - src_len)
                trg = trg + [tag_to_ix['PAD']] * (self.pad_len - src_len)
            else:
                src = src[:self.pad_len]
                trg = trg[:self.pad_len]
                src_len = self.pad_len

            self.source_len.append(src_len)
            self.source.append(src)
            self.target.append(trg)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.LongTensor(self.source[idx]), \
               torch.LongTensor([self.source_len[idx]]), \
               torch.LongTensor(self.target[idx])


class TestDataLoader(Dataset):
    def __init__(self, X, pad_len, max_size=None):
        self.source = []
        self.source_len = []
        self.pad_len = pad_len
        self.read_data(X)

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
    dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, 2)

    model.load_embedding(tokenizer.get_embeddings())
    # multi-GPU
    # model = nn.DataParallel(model)
    model.cuda()

    # loss_criterion = nn.CrossEntropyLoss()  #
    loss_criterion = FocalLoss(2)

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
            label = torch.cat([x[x!=tag_to_ix['PAD']] for x in label])
            loss = loss_criterion(y_pred, label.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPS)
            optimizer.step()
            train_loss += loss.data.cpu().numpy() * data.shape[0]
            del y_pred, loss

        test_loss = 0
        model.eval()
        y_eval_list = []
        label_eval_list = []
        for _, (_data, _seq_len, _label) in enumerate(dev_loader):
            with torch.no_grad():
                _y_pred = model(_data.cuda(), _seq_len)
                _label = torch.cat([x[x != tag_to_ix['PAD']] for x in _label])
                loss = loss_criterion(_y_pred, _label.cuda())
                test_loss += loss.data.cpu().numpy() * _data.shape[0]
                y_eval_list.append(_y_pred.data.cpu().numpy())
                label_eval_list.append(_label)
                del _y_pred, loss

        y_eval_list = np.argmax(np.concatenate(y_eval_list, axis=0), axis=1)
        label_eval_list = np.concatenate(label_eval_list, axis=0)
        print(classification_report(label_eval_list, y_eval_list))
        print("Train Loss: ", str(train_loss / len(train_data)), " Evaluation: ", str(test_loss / len(dev_data)))


def main():
    training_data, A, X_dev, y_dev, Atest = pickle.load(open("FromSahir'sCode.pk", "rb"))
    X_train = [x[0] for x in training_data]
    y_train = [x[1] for x in training_data]
    tokenizer.build_tokenizer(X_train + X_dev, need_split=False)
    tokenizer.build_embedding(GLOVE_PATH, save_pkl=True, dataset_name='tmpmt')
    train(X_train, y_train, X_dev, y_dev)

    tmp = 0


if __name__ == '__main__':
    main()