from data import *
from model import *

from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn as nn

import numpy as np
import random

class TrainDataset(Dataset):

	def __init__(self,vocab):
		self.id2rel, self.id2sent, self.id2position, self.relLine = readTrainFile()
		self.relList = readRelation()
		self.class_num = len(self.relList)
		self.train_len = len(self.id2rel)
		self.vocab = vocab

	def __getitem__(self, index):
		rel1 = self.id2rel[index]
		line1 = self.id2sent[index]
		position1 = self.id2position[index]

		result1, lineLen1 = lineToTensor(line1, position1, self.vocab)

		same_class = random.randint(0, 1)

		if same_class == 0:
			rel2 = rel1
		else:
			while True:
				rel2 = randomChoice(self.relList)
				if rel2 != rel1:
					break

		line2id = randomChoice(self.relLine[rel2])
		line2 = self.id2sent[line2id]
		position2 = self.id2position[line2id]

		result2, lineLen2 = lineToTensor(line2, position2, self.vocab)

		flag = np.array([int(same_class)], dtype=np.float32)

		return result1, lineLen1, result2, lineLen2, flag

	def __len__(self):
		return self.train_len

def train():
	vocab = Vocab()

	train_dataset = TrainDataset(vocab)
	train_dataloader = DataLoader(train_dataset,
	                              shuffle=True,
	                              num_workers=1,
	                              batch_size=Config.batch_size)

	train_len = train_dataset.train_len

	

	net = SiameseNetLSTM(vocab.word_size, vocab.word_vectors)
	# net = torch.load('out/net-classification.pt')
	loss_func = ContrastiveLoss()
	optimizer = optim.Adam(net.parameters(), lr=Config.nn_lr)

	start = time.time()

	current_loss = 0
	all_losses = []

	for epoch in range(0, Config.train_number_epochs):
		current_loss = 0
		print('epoch:' + str(epoch))

		for i, data in enumerate(train_dataloader, 0):
			data1, length1, data2, length2, flag = data

			out1, id1, out2, id2 = net(data1, length1, data2, length2)
			
			optimizer.zero_grad()
			loss_contrastive = loss_func(out1, id1, out2, id2, flag)
			
			loss_contrastive.backward()
			optimizer.step()
			
			current_loss += loss_contrastive.data.item()
			j = (i + 1) * Config.batch_size
			if j % Config.plot_every == 0:
				print('%d %d %d%% (%s) %.4f' % (epoch, j, j * 100 / train_len, timeSince(start), current_loss))
				all_losses.append(current_loss / Config.plot_every)
				current_loss = 0
	
	np.savetxt("out/all_losses.txt", all_losses)
	torch.save(net, 'out/net-classification.pt')	

	print('train done!')


if __name__ == '__main__':
	train()