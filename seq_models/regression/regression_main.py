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
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
import selfies
import helpers

from selfies import encoder, decoder
from helpers import *
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed
from torch import optim
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
seed=0
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.


load_pytorch_obj = False
public = True
if load_pytorch_obj:
	f = '/data/pytorch_obj/smiles_data.pth'
	#f = '/data/pytorch_obj/selfies_data.pth'
	X_train, Y_train, X_test, Y_test, n_train_samples, n_test_samples, vocab_size = torch.load(f)
else:
	#f = '/home/fdamani/mol-edit/data/logp06/train_pairs.txt'
	f = '/home/fdamani/mol-edit/data/qed/train_pairs.txt'
	#f = '/data/potency_dataset_with_props.csv'
	if public:
		sx = process_public_data(f)
	X, Y, vocab_size, n_total_samples = process_data(f)
	X_train, Y_train, X_test, Y_test = train_test_split(X, Y, n_total_samples)

	n_train_samples = len(X_train)
	n_test_samples = len(X_test)

	# torch.save((X_train, Y_train, X_test, Y_test, n_train_samples, n_test_samples, vocab_size), save_file)

# declare RNN object
embed_size = 128
hidden_size = 512

net = rnn.RegressionRNN(vocab_size, embed_size, hidden_size)
net.to(device)

# define loss
criterion = nn.MSELoss()

# define optimizer
lr = 1e-5
optimizer = optim.Adam(net.parameters(), lr=lr)

losses = []
avg_loss = []
test_avg_loss, r2_avg = [], []


batch_size = 128
iter = 0
valid_every = 500
max_iter = 50000
n_epochs = int(float(max_iter) / (float(n_train_samples) / batch_size))

for epoch in range(n_epochs):
	permutation = torch.randperm(n_train_samples)

	for i in range(0, n_train_samples, batch_size):
		optimizer.zero_grad()

		# access mini-batch
		indices = permutation[i:i+batch_size]
		batch_x, batch_y = get_mini_batch(X_train, Y_train, indices)
		batch_packed_x, batch_y = get_packed_batch(batch_x, batch_y)

		# forward pass
		output = net(batch_packed_x)
		loss = criterion(output, batch_y.view(1, -1, 1))
		
		# backprop and gradient step
		loss.backward()
		clip_grad_norm_(net.parameters(), 0.5)
		optimizer.step()

		with torch.no_grad():
			losses.append(loss.item())
			if iter % valid_every == 0 and iter > 0:
				avg_loss.append(np.mean(losses))
				print ('iter: ', iter, ' loss: ', np.mean(losses),)
				losses = []
			if iter % valid_every == 0 and iter > 0:
				# print validation accuracy
				test_permutation = torch.randperm(n_test_samples)
				valid_loss = []
				valid_preds = []
				valid_target = []
				for j in range(0, n_test_samples, batch_size):
					test_indices = test_permutation[j:j+batch_size]
					batch_x_test, batch_y_test = get_mini_batch(X_test, Y_test, test_indices)
					# how to measure validation error.
					batch_packed_x_test, batch_packed_y_test = get_packed_batch(batch_x_test, batch_y_test)
					test_output = net(batch_packed_x_test)
					loss = criterion(test_output, batch_packed_y_test.view(1, -1, 1))
					valid_loss.append(loss.item())
					valid_preds.append(test_output.flatten())
					valid_target.append(batch_packed_y_test)
				r2_val = r2_score(torch.stack(valid_target[:-1]).flatten(), 
						torch.stack(valid_preds[:-1]).flatten())
				print ('valid: ', np.mean(valid_loss), 'r2 score: ', r2_val)
				print ('\n')
				test_avg_loss.append(np.mean(valid_loss))
				r2_avg.append(r2_val)
		iter +=1

plt.plot(avg_loss)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.savefig('training_loss.png')
plt.cla()

plt.plot(test_avg_loss)
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.savefig('test_avg_loss.png')
plt.cla()

plt.plot(r2_avg)
plt.xlabel('Iteration')
plt.ylabel('Validation R2 score')
plt.ylim(.5, 1)
plt.savefig('r2_valid.png')

