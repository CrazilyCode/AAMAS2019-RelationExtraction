from data import *
from model import *

from torch.autograd import Variable
import numpy as np
import subprocess


def test(test_dir):
	net = torch.load('out/net-classification.pt')
	vocab = Vocab()
	print('load vocab done!')

	sample_out, sample_rel = get_output(Config.sample_dir, vocab, net)
	test_out, test_rel = get_output(test_dir, vocab, net)

	relList = readRelation()

	answerlist = []
	guesslist = []

	print('start test...')

	for index in range(len(test_rel)):
		t_o = test_out[index]
		euclidean_distance = F.pairwise_distance(t_o, Variable(sample_out))
		dist = euclidean_distance.data.numpy()

		mink_id = np.argsort(dist)[0 : Config.test_K]

		categorylist = []
		for m_id in mink_id:
			categorylist.append(sample_rel[int(m_id)])

		guess = max(categorylist, key=categorylist.count)

		answerlist.append(str(index) + '\t' + relList[test_rel[index]])
		guesslist.append(str(index) + '\t' + relList[guess])


	answerpath = 'out/test_answer.txt'
	guesspath =  'out/test_guess.txt'
	
	with open(answerpath, 'w') as f:
	    for answer in answerlist:
	        f.write(answer + '\n') 
	with open(guesspath, 'w') as f:
	    for guess in guesslist:
	        f.write(guess + '\n')

	result = subprocess.check_output(["perl", "score.pl", guesspath, answerpath])
	result = result.split()[-1]
	print('F1 score:', result.decode())

	print('test done!')


if __name__ == '__main__':
	test(Config.test_dir)