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

def process_data(file, isSELFIE=False):
	# read data
	dat = pd.read_csv(file)
	# limit to first 10k samples
	# dat = dat.head(1000)
	
	# permute rows
	dat = dat.sample(frac=1).reset_index(drop=True)

	structure = dat['Structure']
	pot = dat['pot_uv']
	if isSELFIE:
		structure, pot, invalid_selfies = utils.encode_to_selfies(structure, pot)

	# take -log10 of potency. we want higher values to mean more potent.
	Y = -torch.log10(torch.tensor(pot, device=device))
	if isSELFIE:
		chars = utils.unique_selfies_chars(structure)
	else:
		chars = utils.unique_chars(structure)
	n_chars = len(chars)
	n_samples = len(structure)
	chars_to_int = {}
	int_to_chars = {}
	for i in range(len(chars)):
		chars_to_int[chars[i]] = i
		int_to_chars[i] = chars[i]

	X = []
	for struct in structure:
		indices = utils.lineToIndices(struct, chars_to_int, isSELFIE)
		X.append(indices)

	return X, Y, n_chars, n_samples

def train_test_split(X, Y, n_samples, train_percent=.7):
	train_ind = int(n_samples * train_percent)
	X_train, X_test = [], []
	
	X_train, Y_train = X[0:train_ind], Y[0:train_ind]
	X_test, Y_test = X[train_ind:], Y[train_ind:]
	return X_train, Y_train, X_test, Y_test


def get_mini_batch(X, Y, indices):
	# get minibatch from rand indices
	batch_x, batch_y = [X[ind] for ind in indices], Y[indices]
	return batch_x, batch_y

def get_packed_batch(batch_x, batch_y):
	# get length of each sample in mb
	batch_lens = [len(sx) for sx in batch_x]
	# arg sort batch lengths in descending order
	sorted_inds = np.argsort(batch_lens)[::-1]
	batch_x = [batch_x[sx] for sx in sorted_inds]
	batch_y = torch.stack([batch_y[sx] for sx in sorted_inds])

	# pack x
	batch_packed_x = nn.utils.rnn.pack_sequence([torch.LongTensor(s) for s in batch_x])
	batch_packed_x = batch_packed_x.to('cuda')
	return batch_packed_x, batch_y

def get_packed_batch_xy(batch_x, batch_y):
	# get length of each sample in mb
	batch_lens = [len(sx) for sx in batch_x]
	# arg sort batch lengths in descending order
	sorted_inds = np.argsort(batch_lens)[::-1]
	batch_x = [batch_x[sx] for sx in sorted_inds]
	batch_y = torch.stack([batch_y[sx] for sx in sorted_inds])

	# pack x
	batch_packed_x = nn.utils.rnn.pack_sequence([torch.LongTensor(s) for s in batch_x])
	batch_packed_x = batch_packed_x.to('cuda')
	return batch_packed_x, batch_y

def process_public_data(file, isSELFIE=False):
	dat = pd.read_csv(file, header=None, delimiter=' ')
	# stack training pairs to single column
	dat = dat.stack().reset_index(drop=True)
	# permute rows
	dat = dat.sample(frac=1).reset_index(drop=True)
	qed = []
	import sys
	sys.path.insert(0, '../data_analysis')
	import properties as prop
	qed = [prop.qed(s) for s in dat]
	ax = pd.concat([dat, pd.Series(qed)],axis=1)
	ax.columns = ['structure', 'qed']
	ax.to_csv('/home/fdamani/mol-edit/data/qed/train_regression.txt', index=None)
	sys.exit(0)

def process_seq_to_seq(file, isSELFIE=False):
	# read data
	dat = pd.read_csv(file, header=None, delimiter=' ')
	# limit to first 10k samples
	# dat = dat.head(1000)
	
	# permute rows
	dat = dat.sample(frac=1).reset_index(drop=True)
	input_seq = dat[0]
	output_seq = dat[1]

	if isSELFIE:
		input_seq, invalid_selfies = utils.encode_to_selfies_nopot(input_seq)
		output_seq, invalid_selfies = utils.encode_to_selfies_nopot(output_seq)

	complete_seq = pd.concat([input_seq, output_seq]).reset_index(drop=True)
	if isSELFIE:
		chars = utils.unique_selfies_chars(complete_seq)
	else:
		chars = utils.unique_chars(complete_seq)
	n_chars = len(chars)
	n_samples = len(input_seq)
	chars_to_int = {'SOS': 0, 'EOS': 1}
	int_to_chars = {0: 'SOS', 1: 'EOS'}
	for i in range(len(chars)):
		chars_to_int[chars[i]] = i+2
		int_to_chars[i+2] = chars[i]

	X = []
	for struct in input_seq:
		indices = utils.lineToIndices(struct, chars_to_int, isSELFIE)
		X.append(indices)
	Y = []
	for struct in output_seq:
		indices = utils.lineToIndices(struct, chars_to_int, isSELFIE)
		Y.append(indices)

	embed()
	return X, Y, n_chars, n_samples