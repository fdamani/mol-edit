'''
regression main file
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import rnn
import rdkit as rd
import pandas as pd
import IPython
import utils
import numpy as np

from IPython import embed
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# read data
dat = pd.read_csv('/data/potency_dataset_with_props.csv')
# permute rows
dat = dat.sample(frac=1).reset_index(drop=True)

structure = dat['Structure']
pot = torch.tensor(dat['pot_uv'], device=device)



chars = utils.unique_chars(structure)
n_chars = len(chars)
n_samples = structure.shape[0]
chars_to_int = {}
int_to_chars = {}
for i in range(len(chars)):
	chars_to_int[chars[i]] = i
	int_to_chars[i] = chars[i]

tensor = utils.lineToTensor(structure[0], chars_to_int, len(chars))
indices = torch.LongTensor(utils.lineToIndices(structure[0], chars_to_int)).view(-1, 1)
indices = indices.to('cuda')
struct = utils.tensorToLine(tensor, int_to_chars)

input_size = n_chars
hidden_size = 128

# declare RNN object
net = rnn.EncoderRNN(input_size, hidden_size)
# place network on cuda device
net.to(device)
# define loss
criterion = nn.MSELoss()
# define optimizer
lr = 1e-2
optimizer = optim.Adam(net.parameters(), lr=lr)
n_iters = 100000
losses = []
for iter in range(0, n_iters):
	optimizer.zero_grad()
	sample = structure[iter % n_samples]
	target = pot[iter % n_samples]
	indices = torch.LongTensor(utils.lineToIndices(sample, chars_to_int)).view(-1, 1).to('cuda')
	# initialize hidden state
	hidden = net.initHidden()
	for i in range(len(indices)):
		_, hidden = net(indices[i], hidden)
	output = net.score(hidden)
	loss = criterion(output, target.view(1, 1, -1))
	loss.backward()
	optimizer.step()
	losses.append(loss.item())
	with torch.no_grad():
		if iter % 100 == 0:
			print ('iter: ', iter, ' loss: ', np.mean(losses))
			losses = []
