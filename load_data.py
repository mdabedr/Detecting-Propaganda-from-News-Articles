import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #handling files and directories
import torch #pytorch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig #bert
import unidecode, unicodedata  #to deal with accented characters
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler #data loading
from keras.preprocessing.sequence import pad_sequences #padding

def read_train_sent(files, path):
    """ Returns a list of read text sentences from the given list of file names (with respective locations)
    
    Input:
        files: A list of strings, each element having file names with location
    
    Output:
        ret: A list of sentences read from each file
        
    """
    ret = []
    i = 0
    #Read text from all directories in files
    for x in files:
        with open(path+x, encoding='utf8') as f:
            j = 0 #sentence
            for line in f:
                ret.append([int((x.replace("article",'')).replace('.txt','')), j,  line.lower()])
                j+=1
            i+=1  #articles
    return ret    

def read_label_data(files):
    """ Returns a list of read label positions & indices from the given list of file names (with respective locations)
    
    Input:
        files: A list of strings, each element having file names with location
    
    Output:
        ret: A list of label & indices read from each file
        
    """
    ret = []
    i = 0
    #Read text from all directories in files
    for x in files:
        with open(x, encoding='utf8') as f:
            #For each line
            for line in f:
                #Read the ints
                labels = [int(x) for x in line.split()]
                #Disregard the article id
                #labels[0] = i
                #Store the labels
                ret.append(labels)
        i+=1
    return ret

def get_prop_index(y_train, doc_id):
    """ Returns a list of start and end range for a given doc_id
    """
    lst = []
    for l in y_train:
        #if test == True:
        #   print(l)
        if l[0] == doc_id:
            lst.append((l[1], l[2]))
    return lst

def strip_accents(text):
    """Returns a utf-8 string with replaced accented characters
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
            pass

    text = unicodedata.normalize('NFD', text)\
            .encode('ascii', 'ignore')\
            .decode("utf-8")

    return str(text)


def load_data():
    #Set up reading arguments
    path_train = 'train-articles/'
    files = os.listdir(path_train)
    files.sort()
    #train = [path_train+s for s in files]
    X_train_sent = read_train_sent(files, path_train)

    path_labels = 'train-labels-task1-span-identification/'
    files = os.listdir(path_labels)
    files.sort()
    labels = [path_labels+s for s in files]

    #Read sentences per article
    y_train = read_label_data(labels)

    ##########################################################
    #Code for creating features and labels for each sentence:#
    ##########################################################
    #Init Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    #tokenizer = BertTokenizer.from_pretrained("/home/snoorali/projects/def-lilimou/snoorali/uncased_L-12_H-768_A-12/", do_lower_case=True)
    train_list = []
    labels_list = []
    z = -1 #track document id
    k = 0  #track document index
    for elements in X_train_sent:
        if elements[0] > z:
            z+=1 #+1 document id
            k=0  #New index for new doc.
            prop_list = get_prop_index(y_train, elements[0])
        flag = False
        move = False
        i, j = 0, 0
        text = strip_accents(elements[2])
        tokenized_text = tokenizer.tokenize(text)#text.split()##nltk.word_tokenize(text)##.split()
        #Empty/New Line token => Skip
        if len(tokenized_text) == 0:
            k+=1
            continue
        #[article_id, sent_id, tokenized_text]
        train_list.append([elements[0], elements[1], tokenized_text])
        labels_sublist = []
        #For 1 sentence:
        while i < len(text):
            if text[i] == ' ' or text[i] == '\n':
                i+=1
                k+=1
                continue
            token = tokenized_text[j]
            move = False
            for token_char in token:
                if token_char == text[i]:
                    if move == True:
                        i+=1
                        k+=1
                        continue
                    flag = False
                    #check if i is in propaganda index
                    for r1, r2 in prop_list:
                        if k >= r1 and k < r2:
                            #propaganda exists
                            labels_sublist.append('1')
                            flag = True
                            move = True
                            break
                    if flag == False:
                        #propaganda doesn't exist
                        labels_sublist.append('0')
                        move = True
                    i+=1
                    k+=1
                #chars don't match - do nothing
            j+=1
        #[article_id, sent_id, propaganda]
        labels_list.append([elements[0], elements[1], labels_sublist])


    X = []
    length = 0
    for e in train_list:
        #Making our dataset of sentences
        X.append(e[2])
        #Finding the max length of tokenized sequence
        if len(e[2]) > length:
                length = len(e[2])

    Y = []
    for e in labels_list:
        #Making our targets for each sentence (per word)
        Y.append(e[2])

    # print(X[:100])
    # print(Y[:100])

    #Max length of a tokenized sentence
    print(length) #176

    ############################################
    # Same stuff for test data:
    ############################################
    #Set up reading arguments
    path_test = 'dev-articles/'
    files = os.listdir(path_test)
    files.sort()
    #test = [path_test+s for s in files]

    label_test_file = 'dev-task-TC-template.out'

    #Read sentences per article
    X_test_sent = read_train_sent(files, path_test)
    y_test = []

    with open(label_test_file, encoding='utf8') as f:
        #For each line
        for line in f:
            #Read the ints
            ints = line.split()
            #Disregard the article id
            #labels[0] = i
            #Store the labels
            y_test.append([int(ints[0]), int(ints[2]), int(ints[3])])
    #y_train = read_label_data(labels)

    #print(y_test[:2])
    #print()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    #tokenizer = BertTokenizer.from_pretrained("/home/snoorali/projects/def-lilimou/snoorali/uncased_L-12_H-768_A-12/", do_lower_case=True)
    test_list = []
    test_labels_list = []
    z = -1 #track document id
    k = 0  #track document index
    for elements in X_test_sent:
        if elements[0] > z:
            z+=1 #+1 document id
            k=0  #New index for new doc.
            prop_list = get_prop_index(y_test, elements[0])
            #print(prop_list)
        flag = False
        move = False
        i, j = 0, 0
        text = strip_accents(elements[2])
        tokenized_text = tokenizer.tokenize(text)#text.split()#nltk.word_tokenize(text)###split()
        #Empty/New Line token => Skip
        if len(tokenized_text) == 0:
            k+=1
            continue
        #[article_id, sent_id, tokenized_text]
        test_list.append([elements[0], elements[1], tokenized_text])
        labels_sublist = []
        #For 1 sentence:
        while i < len(text):
            if text[i] == ' ' or text[i] == '\n':
                i+=1
                k+=1
                continue
            token = tokenized_text[j]
            move = False
            for token_char in token:
                if token_char == text[i]:
                    if move == True:
                        i+=1
                        k+=1
                        continue
                    flag = False
                    #check if i is in propaganda index
                    for r1, r2 in prop_list:
                        if k >= r1 and k < r2:
                            #propaganda exists
                            labels_sublist.append("1")
                            flag = True
                            move = True
                            break
                    if flag == False:
                        #propaganda doesn't exist
                        labels_sublist.append('0')
                        move = True
                    i+=1
                    k+=1
                #chars don't match - do nothing
            j+=1
        #[article_id, sent_id, propaganda]
        test_labels_list.append([elements[0], elements[1], labels_sublist])


    Xtest = []
    length = 0
    for e in test_list:
        #Making our dataset of sentences
        Xtest.append(e[2])
        #Finding the max length of tokenized sequence
        if len(e[2]) > length:
                length = len(e[2])

    print(test_labels_list[:3])
    Ytest = []
    for e in test_labels_list :
        #Making our targets for each sentence (per word)
        Ytest.append(e[2])

    return X, Y, Xtest, Ytest
