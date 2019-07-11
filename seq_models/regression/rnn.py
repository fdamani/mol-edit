from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=0
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.

class RegressionRNN(nn.Module):
	'''
	Input: sequence of chars
	Output: regression score for last token.
	'''
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, input):
		b_size = input.batch_sizes[0].item()

		embedded = nn.utils.rnn.PackedSequence(self.embedding(input.data), 
			input.batch_sizes)
		hidden = self.initHidden(b_size)
		output, hidden = self.gru(embedded, hidden)
		score = self.score(hidden)
		return score

	def score(self, hidden):
		'''
			fully connected layer outputting a single scalar value
		'''
		return self.fc(hidden)

	def initHidden(self, b_size=1):
		return torch.zeros(1, b_size, self.hidden_size, device=device)

class Seq2Seq(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(Seq2Seq, self).__init__()
		self.hidden_size = hidden_size
		self.emb_enc = nn.Embedding(vocab_size, embed_size)
		self.gru_enc = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
		self.out_enc = nn.Linear(hidden_size, hidden_size)
		

		self.emb_dec = nn.Embedding(vocab_size, embed_size)
		self.gru_dec = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
		self.out_dec = nn.Linear(hidden_size, vocab_size)

	def encoder(self, inp):
		'''
			inp: input
		'''
		b_size = inp.batch_sizes[0].item()
		h = self.initHidden(b_size)
		emb = nn.utils.rnn.PackedSequence(self.embedding(inp.data), 
			inp.batch_sizes)
		_, h = self.gru_enc(emb, h)
		h = self.out_enc(h)
		return h

	def decoder(self, dec_inp, h):
		'''
			dec_inp: decoder input
			h: hidden state
		'''
		emb = nn.utils.rnn.PackedSequence(self.embedding(dec_inp.data), 
			dec_inp.batch_sizes)
		
		emb = self.gru_dec(emb, h)


class EncoderRNN(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, input):
		b_size = input.batch_sizes[0].item()

		embedded = nn.utils.rnn.PackedSequence(self.embedding(input.data), 
			input.batch_sizes)
		hidden = self.initHidden(b_size)
		output, hidden = self.gru(embedded, hidden)
		return hidden

	def score(self, hidden):
		'''
			fully connected layer outputting a single scalar value
		'''
		return self.fc(hidden)

	def initHidden(self, b_size=1):
		return torch.zeros(1, b_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, input, hidden):
		embedded = nn.utils.rnn.PackedSequence(self.embedding(input.data), 
			input.batch_sizes)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)













