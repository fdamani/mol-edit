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

from selfies import encoder, decoder
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed
from torch import optim
import random
import re

class Lang:
	def __init__(self, name):
		'''
		name: selfies or smiles
		'''
		self.PAD_token = 0
		self.SOS_token = 1
		self.EOS_token = 2
		self.name = name
		self.trimmed = False
		self.char2index = {}
		self.char2count = {}
		self.index2char = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
		self.n_chars = 3 # Count default tokens

	def index_chars(self, x):
		if self.name == 'selfies':
			sx = re.findall(r"\[[^\]]*\]", x)
		elif self.name == 'smiles':
			sx = x.split('')
		else:
			print 'ERROR: specify proper name.'
			sys.exit(0)

		for char in sx:
			self.index_char(char)

	def index_char(self, char):
		if char not in self.char2index:
			self.char2index[char] = self.n_chars
			self.char2count[char] = 1
			self.index2char[self.n_chars] = char
			self.n_chars += 1
		else:
			self.char2count[char] += 1

	def trim(self, min_count):
		if self.trimmed: return
		self.trimmed = True
		
		keep_chars = []
		
		for k, v in self.char2count.items():
			if v >= min_count:
				keep_chars.append(k)

		print('keep_chars %s / %s = %.4f' % (
			len(keep_chars), len(self.char2index), float(len(keep_chars)) / len(self.char2index)
		))

		# Reinitialize dictionaries
		self.char2index = {}
		self.char2count = {}
		self.index2char = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
		self.n_chars = 3 # Count default tokens

		for char in keep_chars:
			self.index_char(char)