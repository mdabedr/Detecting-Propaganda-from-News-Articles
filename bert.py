import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #handling files and directories
import torch #pytorch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig #bert
from pytorch_pretrained_bert import BertForTokenClassification #bert
import unidecode, unicodedata  #to deal with accented characters
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam #Adam
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler #data loading
from keras.preprocessing.sequence import pad_sequences #padding
from sklearn.model_selection import train_test_split #data splitting
from early_stopping import EarlyStopping
import time
from load_data import load_data

tstart = time.time()
timestr = time.strftime("%Y%m%d-%H%M%S")

X, Y, Xtest, Ytest = load_data()

#Following on the code inspired from: https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

#Printing the GPU model
print(torch.cuda.get_device_name(0))

#Labels to id's
tag2idx = {"0": 0, "1": 1}
print(tag2idx)

#Assign tag values
tags_vals = ['0' , '1']

#Defining MAX_LEN to be 210 as per the paper
MAX_LEN = 210

#Padding inputs
tr_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in X],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


#Padding targets
tr_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in Y],
                     maxlen=MAX_LEN, value=tag2idx['0'], padding="post",
                     dtype="long", truncating="post")

#Setting up attention masks
tr_masks = [[float(i>0) for i in ii] for ii in tr_inputs]


#Padding inputs
val_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in Xtest],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


#Padding targets
val_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in Ytest],
                     maxlen=MAX_LEN, value=tag2idx['0'], padding="post",
                     dtype="long", truncating="post")

#Setting up attention masks
val_masks = [[float(i>0) for i in ii] for ii in val_inputs]

#Converting the dataset to torch tensors
tr_inputs = torch.LongTensor(tr_inputs)
val_inputs = torch.LongTensor(val_inputs)
tr_tags = torch.LongTensor(tr_tags)
val_tags = torch.LongTensor(val_tags)
tr_masks = torch.LongTensor(tr_masks)
val_masks = torch.LongTensor(val_masks)

#Setting the batch size
BATCH_SIZE = 16

#Shuffling the dataset
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data,sampler=valid_sampler, batch_size=BATCH_SIZE)

#Init BERT
model = BertForTokenClassification.from_pretrained("/home/snoorali/projects/def-lilimou/snoorali/uncased_L-12_H-768_A-12/", num_labels=len(tag2idx))
#model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx))

#Transfer the model to cuda
print("Detect ", torch.cuda.device_count(), "GPUs!")
#model = nn.DataParallel(model)
model.to("cuda")

epochs = 4
warmup_proportion = 0.1
num_train_optimization_steps = (len(X) / BATCH_SIZE) * epochs
param_optimizer = list(model.classifier.named_parameters()) 
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
patience = 2
early_stopping = EarlyStopping(patience=patience, verbose=True)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=3e-5,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

class_weights = torch.Tensor([0.109355466, 0.890644534]).cuda()
loss_criterion = nn.CrossEntropyLoss(weight=class_weights)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#Training BERT
max_grad_norm = 1.0
train_loss_list = []
val_loss_list = []
for e in range(epochs):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        # forward pass
        logits = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask)

        logits = logits.argmax(-1)
        
        loss = binary_criterion(logits.float(), b_labels.float())
        loss.requires_grad = True

        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Epoch #",e+1)
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    train_loss_list.append(tr_loss/nb_tr_steps)
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    val_loss_list.append(eval_loss)
    
    pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
    valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
    
    #Store the validation f1
    val_f1 = f1_score(valid_tags, pred_tags)
    #print("F1-Score: {}".format(val_f1))
    print("Validation F1-Score: {}".format(f1_score(valid_tags, pred_tags)))
    print("Validation Precision-Score: {}".format(precision_score(valid_tags, pred_tags)))
    print("Validation Recall-Score: {}".format(recall_score(valid_tags, pred_tags)))
    print(classification_report(valid_tags, pred_tags))

    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
    spath = os.path.join('checkpoints', timestr+".pt")
    #Do early stopping with validation f1-score
    early_stopping(val_f1, model, spath)

    #If patience met, stop training
    if early_stopping.early_stop:
        print("Early stopping")
        break  

#Test BERT
model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l]
print("Loss: {}".format(eval_loss/nb_eval_steps))
print("Accuracy: {}".format(eval_accuracy/nb_eval_steps))

print("F1-Score: {}".format(f1_score(valid_tags, pred_tags)))
print("Precision-Score: {}".format(precision_score(valid_tags, pred_tags)))
print("Recall-Score: {}".format(recall_score(valid_tags, pred_tags)))

print("\n")
print("Classification Report: ")
print(classification_report(valid_tags, pred_tags))
print("Time taken by the script:",time.time()-tstart)