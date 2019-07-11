'''
Predicts potency conditioned on compound structure using a RNN.
Options for SMILES and SIMILES-based representation.
'''
import torch
import rdkit as rd
import pandas as pd
import numpy as np
import IPython
import re
import selfies
from selfies import encoder, decoder

from IPython import embed
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=0
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.

def train_test_split(x, y, test=.3, imbalance=True):
	'''
		train test split.

		:param x (str) structure
		:param y (float) potency value
		:param test (float) percent test
		:param imbalance (bool) if true, separate train/test conditioned on active/inactive
			active = potency value not equal to 1000.000
			inactive = potency value 1000.000

		return x_train, y_train, x_test, y_test
	'''
	return None

def unique_chars(x):
	'''
		Iterate through all chars in dataset and return unique set of chars
		
		:param x (list of str) compounds
		return y (list of chars) unique characters
	'''
	chars = set([])
	for i in range(x.shape[0]):
		sx = list(x[i])
		for j in range(len(sx)):
			chars.add(sx[j])
	return list(chars)

def encode_to_selfies(struct, pot):
	'''smiles to selfies'''
	selfies_struct = []
	selfies_pot = []
	invalid_selfies = []
	for i in range(struct.shape[0]):
		ax = encoder(struct[i])
		if ax != -1:
			selfies_struct.append(ax)
			selfies_pot.append(pot[i])
		else:
			invalid_selfies.append(struct[i])
	return selfies_struct, selfies_pot, invalid_selfies


def encode_to_selfies_nopot(struct):
	'''smiles to selfies'''
	selfies_struct = []
	invalid_selfies = []
	for i in range(struct.shape[0]):
		ax = encoder(struct[i])
		if ax != -1:
			selfies_struct.append(ax)
		else:
			invalid_selfies.append(struct[i])
	return selfies_struct, invalid_selfies

def unique_selfies_chars(x):
	chars = set([])
	count = 0
	invalid_selfies = []
	for i in range(len(x)):
		sx = re.findall(r"\[[^\]]*\]", x[i])
		for j in range(len(sx)):
			chars.add(sx[j])
	return list(chars)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter, chars_to_int):
	return chars_to_int[letter]

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter, chars_to_int):
	tensor = torch.zeros(1, n_letters, dtype=torch.Long, device=device)
	tensor[0][chars_to_int[letter]] = 1
	return tensor

def onehotToLetter(tensor, int_to_chars):
	return int_to_chars[(tensor.flatten()==1).nonzero().item()]

def tensorToLine(tensor, int_to_chars):
	line = []
	for i in range(tensor.shape[0]):
		line.append(onehotToLetter(tensor[i], int_to_chars))
	return ''.join(line)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line, chars_to_int, n_letters):
	tensor = torch.zeros(len(line), 1, n_letters, device=device)
	for li, letter in enumerate(line):
		tensor[li][0][chars_to_int[letter]] = 1
	return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToIndices(line, chars_to_int, isSELFIE=True):
	indices = []
	if isSELFIE:
		line = re.findall(r"\[[^\]]*\]", line)
	for li, letter in enumerate(line):
		indices.append(chars_to_int[letter])
	return indices

def indicesToLine(indices, int_to_chars):
	chars = []
	for i in range(len(indices)):
		chars.append(int_to_chars[indices[i]])
	return ''.join(chars)



