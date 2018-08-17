from _collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import string
import time
import math
import random


class Config():
    vocab_dir = 'data/glove.6B.50d.txt'

    relation_dir = 'data/all_relation.txt'
    train_dir = 'data/all_train.txt'
    sample_dir = 'data/all_sample.txt'
    test_dir = 'data/all_test.txt'

    input_maxlen = 100
    position_size = 200
    position_dim = 5

    word_dim = 50
    input_dim = word_dim + position_dim * 2
    

    batch_size = 1
    hidden_size = 100
    nn_lr = 0.001
    train_number_epochs = 50
    plot_every = 100

    output_size = 10
    loss_margin = 10
    sample_length = 20
    test_K = 30


def readRelation(file_dir=Config.relation_dir):
    relList = []
    with open(file_dir) as f:
        for line in f.readlines():
            relList.append(line.strip())
    return relList


def readTrainFile(file_dir=Config.train_dir):
    id2sid = []
    id2rel = []
    id2sent = []
    id2position = []
    relLine = {}
    
    with open(file_dir) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            sid = line[0]
            rel = line[1]
            sent = line[2]
            position = line[3]

            id2sid.append(sid)
            id2rel.append(rel)
            id2sent.append(sent)
            id2position.append(position)
    
    relList = readRelation()
    
    for rel in relList:
        lines = []
        with open(file_dir) as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                
                if rel == line[1]:
                    lines.append(id2sid.index(line[0]))
        relLine[rel] = lines
          
    return id2rel, id2sent, id2position, relLine


def readFile(file_dir):
    id2rel = []
    id2sent = []
    id2position = []
    
    with open(file_dir) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            rel = line[1]
            sent = line[2]
            position = line[3]

            id2rel.append(rel)
            id2sent.append(sent)
            id2position.append(position)
    return id2rel, id2sent, id2position



class Vocab(object):

    def __init__(self, path_vec=Config.vocab_dir):
        self.path = path_vec
        self.word2id = defaultdict(int)
        self.word_vectors = None
        self.word_size = 0

        self.load_data()

    def load_data(self):
        with open(self.path, 'rb') as f:
            self.word_vectors = []
            idx = 0
            for line in f.readlines():
                line = line.decode().split(' ')
                self.word2id[line[0]] = idx
                vector = np.asarray( line[1:], dtype=np.float32)
                self.word_vectors.append(vector)
                idx += 1

            self.word_size = idx
            # print('Vocab size:', idx)

def positionToTensor(position, lineLen):
    position = position.split(' ')
    e1start = int(position[0])
    e1end = int(position[1])
    e2start = int(position[2])
    e2end = int(position[3])
    
    e1_result = []
    e2_result = []
    for i in range(lineLen):
        e1_flag = 0
        if i >= e1start and i <= e1end:
            e1_flag = 0
        if i < e1start:
            e1_flag = i - e1start
        if i > e1end:
            e1_flag = i - e1end

        e1_result.append(e1_flag)

        e2_flag = 0
        if i >= e2start and i <= e2end:
            e2_flag = 0
        if i < e2start:
            e2_flag = i - e2start
        if i > e2end:
            e2_flag = i - e2end

        e2_result.append(e2_flag)
    
    e1_result = np.reshape(e1_result, (lineLen, 1))
    e2_result = np.reshape(e2_result, (lineLen, 1))
    result = np.hstack((e1_result, e2_result))
    return result


def cleanLines(line):
    identify = str.maketrans('', '')
    delEStr = string.punctuation + string.digits
    cleanLine = line.translate(identify, delEStr)
    return cleanLine


def lineToTensor(sent,position,vocab):
    sent = sent.lower()
    # sent = cleanLines(sent)
    sent = sent.split(' ')

    lineLen = len(sent)

    sent_result = []
    for word in sent:
        wordid = vocab.word2id[word]
        sent_result.append(int(wordid))

    sent_result = np.reshape(sent_result, (lineLen, 1))
    position_result = positionToTensor(position, lineLen)

    result = np.hstack((sent_result, position_result))

    temp = np.zeros((Config.input_maxlen - lineLen, 3), dtype=np.int)
    result = np.vstack((result, temp))
 
    return result, lineLen


def sort_batch(data, length, batch_size):
    data_len = data.size()[0]
    data_id = torch.arange(data_len, dtype=torch.int)

    _, inx =torch.sort(length, descending=True)

    data = data[inx]
    data_id = data_id[inx]
    length = length[inx]

    if data_len < batch_size:
        temp_data = torch.zeros((batch_size - data_len, Config.input_maxlen, 3), dtype=torch.int)
        data = torch.cat((data, temp_data), 0)

        temp_id = torch.arange(data_len, batch_size, dtype=torch.int)
        data_id = torch.cat((data_id, temp_id), 0)

        temp_length = torch.ones((batch_size - data_len), dtype=torch.long)
        length = torch.cat((length, temp_length), 0)


    length = list(length)
    
    return (data, data_id, length, data_len)

def recover_batch(data, data_id, data_len):
    _, inx = torch.sort(data_id)
    data = data[inx]
    data = data[0: data_len]

    return data


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


class TestDataset(Dataset):

    def __init__(self, vocab, path):
        self.test_rel, self.test_line, self.test_position = readFile(path)
        self.relList = readRelation()
        self.class_num = len(self.relList)
        self.test_len = len(self.test_line)
        self.vocab = vocab

    def __getitem__(self, index):
        rel = self.test_rel[index]
        line = self.test_line[index]
        position = self.test_position[index]

        result, lineLen = lineToTensor(line, position, self.vocab)

        return result, self.relList.index(rel), lineLen

    def __len__(self):
        return self.test_len

def get_output(path, vocab, net):
    test_dataset = TestDataset(vocab, path)
    test_dataloader = DataLoader(test_dataset,
                                  shuffle=False,
                                  num_workers=1,
                                  batch_size=Config.batch_size)

    test_out = []
    test_rel = []

    for i, data in enumerate(test_dataloader, 0):
        data, label, length = data
        out, data_id = net.forward_once(data, length)

        _, inx = torch.sort(data_id)
        out = out[inx]
        out = out[0: len(label)]

        test_out.append(out)
        test_rel.append(label)

    test_out = torch.cat(test_out)
    test_rel = torch.cat(test_rel)

    return test_out, test_rel