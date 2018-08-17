import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from data import *

torch.manual_seed(1)

class SiameseNetLSTM(nn.Module):
    def __init__(self, word_size, word_vectors):
        super(SiameseNetLSTM, self).__init__()
        self.hidden_size = Config.hidden_size
        self.batch_size = Config.batch_size
        num_layers = 1
        biFlag = True

        if(biFlag):
            self.bi_num = 2
        else:
            self.bi_num = 1

        self.rnn = nn.LSTM(
            input_size=Config.input_dim,
            hidden_size=Config.hidden_size,
            num_layers=num_layers,
            bidirectional=biFlag,
        )

        self.h0 = Variable(torch.randn(num_layers * self.bi_num, self.batch_size, self.hidden_size))
        self.c0 = Variable(torch.randn(num_layers * self.bi_num, self.batch_size, self.hidden_size))


        self.out = nn.Linear(self.hidden_size * self.bi_num , Config.output_size)

        self.word_embedding = nn.Embedding(word_size, Config.word_dim)
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.array(word_vectors)))
        # self.word_embedding = nn.Embedding.from_pretrained(vocab.word_vectors)
        
        self.position_embedding = nn.Embedding(Config.position_size, Config.position_dim)


    def forward_once(self, data, length):
        data, data_id, length, data_len = sort_batch(data, length, self.batch_size)

        max_length=length[0]
        data = data[:, 0:max_length, :]
        data = Variable(data)

        tensor_word = data[:, :, 0].long()
        data_word = self.word_embedding(tensor_word)

        tensor_position1 = (data[:, :, 1] + (Config.position_size/2)).long()
        data_position1 = self.position_embedding(tensor_position1)

        tensor_position2 = (data[:, :, 2] + (Config.position_size/2)).long()
        data_position2 = self.position_embedding(tensor_position2)

        data = torch.cat((data_word, data_position1, data_position2), 2)

        data = nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)

        out, (h_n, h_c) = self.rnn(data, (self.h0, self.c0))
        out,length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = torch.mean(out,1)

        output = self.out(out)

        # output = recover_batch(output, data_id, data_len)

        return output, data_id

    def forward(self, data1, length1, data2, length2):
        output1, data1_id = self.forward_once(data1, length1)
        output2, data2_id = self.forward_once(data2, length2)

        return output1, data1_id, output2, data2_id

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.margin = Config.loss_margin

    def forward(self, output1, id1, output2, id2, flag):
        _, inx1 = torch.sort(id1)
        out1 = output1[inx1]

        _, inx2 = torch.sort(id2)
        out2 = output2[inx2]

        euclidean_distance = F.pairwise_distance(out1, out2)

        loss_contrastive = torch.mean((1 - flag) * torch.pow(torch.clamp(euclidean_distance - 0, min=0.0), 2) +
                                      (flag) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

       
        return loss_contrastive