import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit as rd
import pandas as pd
import IPython
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
import re

from selfies import encoder, decoder
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed
from torch import optim
import random
import Lang
from Lang import Lang
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_data(file, selfies=False, reverse=False):
	# read data
	dat = pd.read_csv(file, header=None, delimiter=' ')
	dat = dat.sample(frac=1).reset_index(drop=True)
	# limit to first 10k samples
	if reverse:
		input_seq, output_seq = dat[1], dat[0]
	else:
		input_seq, output_seq = dat[0], dat[1]

	if selfies:
		lx = Lang('selfies')
		input_seq = smiles_to_selfies(input_seq)
		output_seq = smiles_to_selfies(output_seq)
	else:
		lx = Lang('smiles')
	# filter out NaNs
	seqs = pd.DataFrame([input_seq, output_seq]).T
	seqs = seqs.dropna()
	input_seq, output_seq = seqs[0], seqs[1]

	# possible filter out pairs that are too long?

	# index characters
	for comp in input_seq:
		lx.index_chars(comp)
	for comp in output_seq:
		lx.index_chars(comp)

	return input_seq, output_seq, lx


def read_single_data(file, lang, selfies=False):
	dat = pd.read_csv(file, header=None).squeeze()
	if selfies:
		dat = smiles_to_selfies(dat)
	dat = pd.DataFrame(dat).dropna().squeeze()
	seqs = []
	for sx in dat:
		seqs.append(indexes_from_compound(lang, sx))
	lengths = [len(s) for s in seqs]
	seqs = torch.LongTensor(seqs).transpose(0, 1)
	seqs.to(device)
	return seqs, lengths


def filter_low_resource(input_list, output_list, lx, MIN_COUNT=5):
	'''

	filter vocab to chars that repeat at least MIN_COUNT times
	filter pairs
	'''
	lx.trim(MIN_COUNT)
	filtered_input, filtered_output = [], []
	for i in range(len(input_list)):
		input_seq = input_list[i]
		output_seq = output_list[i]
		keep_input = True
		keep_output = True
		sx = re.findall(r"\[[^\]]*\]", input_seq)
		for char in sx:
			if char not in lx.char2index:
				keep_input = False
				break
		sx = re.findall(r"\[[^\]]*\]", output_seq)
		for char in sx:
			if char not in lx.char2index:
				keep_output = False
				break

		if keep_input and keep_output:
			filtered_input.append(input_seq)
			filtered_output.append(output_seq)
	print ('trimmed from %d pairs to %d' % (len(input_list), len(filtered_input)))
	return filtered_input, filtered_output, lx

def smiles_to_selfies(x):
	'''smiles to selfies'''
	output = []
	for i in range(x.shape[0]):
		ax = encoder(x[i])
		if ax != -1:
			output.append(ax)
		else:
			output.append('NaN')
	return output

def selfies_to_smiles(x):
	'''smiles to selfies'''
	valid_x = []
	invalid_x = []
	for i in range(x.shape[0]):
		ax = decoder(x[i])
		if ax != -1:
			valid_x.append(ax)
		else:
			invalid_x.append(x[i])
	return valid_x, invalid_x


# return list of indexes, one for each char in compound, plus EOS
def indexes_from_compound(lang, x):
	sx = re.findall(r"\[[^\]]*\]", x)
	return [lang.char2index[char] for char in sx] + [lang.EOS_token]

def compound_from_indexes(lang, x):
	return [lang.index2char[ind.item()] for ind in x]

def pad_seq(seq, max_length, lx):
    seq += [lx.PAD_token for i in range(max_length - len(seq))]
    return seq

def train_test_split(X, Y, train_percent=.7):
	'''train test split: assume data has already been permuted'''
	n_samples = len(X)
	train_ind = int(n_samples * train_percent)
	X_train, X_test = [], []
	
	X_train, Y_train = X[0:train_ind], Y[0:train_ind]
	X_test, Y_test = X[train_ind:], Y[train_ind:]
	return X_train, Y_train, X_test, Y_test

def random_batch(batch_size, pairs, lx, replace=True):
	'''get a random mini batch of training pairs

	sort by length in descending order
	compute indexes and pad
	'''
	input_seqs, output_seqs = [], []
	num_samples = pairs.shape[0]
	rand_inds = np.random.choice(num_samples, batch_size, replace=replace)
	for ind in rand_inds:
		pair = pairs.iloc[ind]
		input_seqs.append(indexes_from_compound(lx, pair[0]))
		output_seqs.append(indexes_from_compound(lx, pair[1]))
	
	# zip into pairs, sort by length (descending), unzip
	seq_pairs = sorted(zip(input_seqs, output_seqs), key=lambda p: len(p[0]), reverse=True)
	input_seqs, output_seqs = zip(*seq_pairs)

	input_lengths = [len(s) for s in input_seqs]
	input_padded = [pad_seq(s, max(input_lengths), lx) for s in input_seqs]
	output_lengths = [len(s) for s in output_seqs]
	output_padded = [pad_seq(s, max(output_lengths), lx) for s in output_seqs]

	input_var = torch.LongTensor(input_padded).transpose(0, 1)
	output_var = torch.LongTensor(output_padded).transpose(0, 1)

	input_var = input_var.to(device)
	output_var = output_var.to(device)
	return input_var, input_lengths, output_var, output_lengths


	# complete_seq = pd.concat([input_seq, output_seq]).reset_index(drop=True)
	# if isSELFIE:
	# 	chars = utils.unique_selfies_chars(complete_seq)
	# else:
	# 	chars = utils.unique_chars(complete_seq)
	# n_chars = len(chars)
	# n_samples = len(input_seq)
	# chars_to_int = {'SOS': 0, 'EOS': 1}
	# int_to_chars = {0: 'SOS', 1: 'EOS'}
	# for i in range(len(chars)):
	# 	chars_to_int[chars[i]] = i+2
	# 	int_to_chars[i+2] = chars[i]

	# X = []
	# for struct in input_seq:
	# 	indices = utils.lineToIndices(struct, chars_to_int, isSELFIE)
	# 	X.append(indices)
	# Y = []
	# for struct in output_seq:
	# 	indices = utils.lineToIndices(struct, chars_to_int, isSELFIE)
	# 	Y.append(indices)

	# embed()
	# return X, Y, n_chars, n_samples