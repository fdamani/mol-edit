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

class EncoderRNN(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, n_layers=2, dropout=0.5):
		super(EncoderRNN, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(vocab_size, self.embed_size)
		if self.n_layers > 1:
			self.gru = nn.GRU(self.embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=False)
		else:
			self.gru = nn.GRU(self.embed_size, hidden_size, n_layers)
	
	def forward(self, input_seqs, input_lengths):
		embedded = self.embedding(input_seqs)
		b_size = input_seqs.size(1)
		hidden = self.initHidden(b_size)
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack - back to padded
		return outputs, hidden

	def initHidden(self, b_size=1):
		return torch.zeros(self.n_layers, b_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, n_layers=2, dropout=.5):
		super(DecoderRNN, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

		if self.n_layers > 1:
			self.gru = nn.GRU(self.embed_size, self.hidden_size, self.n_layers, dropout=dropout)
		else:
			self.gru = nn.GRU(self.embed_size, self.hidden_size, self.n_layers)

		self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.vocab_size)

		#self.fc = nn.Linear(hidden_size, 1)

	def forward(self, input_seq, last_hidden, encoder_outputs=None):
		'''run one step at a time'''
		b_size = input_seq.size(0)
		embedded = self.embedding(input_seq)
		embedded = embedded.view(1, b_size, self.embed_size)

		output, hidden = self.gru(embedded, last_hidden)
		output = self.out(output)
		return output, hidden

class Attn(nn.Module):
	def __init__(self, method_str, hidden_size):
		super(Attn, self).__init__()

		self.method_str = method_str
		self.hidden_size = hidden_size

		#if method_str == 'general':
		#self.attn = nn.Linear(self.hidden_size, self.hidden_size)

		# elif method_str == 'concat':
		# 	self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
		# 	self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

	def forward(self, hidden, encoder_outputs):
		max_len = encoder_outputs.size(0)
		this_batch_size = encoder_outputs.size(1)

		attn_energies = torch.zeros(this_batch_size, max_len)

		if torch.cuda.is_available():
			attn_energies = attn_energies.cuda()

		for b in range(this_batch_size):
			attn_energies[b] = self.score(hidden[:,b], encoder_outputs[:, b])
		
		# can use torch.baddbmm(torch.zeros(4,1), x, y).squeeze(2)
		# or torch.bmm


			# for i in range(max_len):
			# 	attn_energies[b,i] = self
			# 	attn_energies[b, i] = self.score(hidden[:, b].squeeze(), encoder_outputs[i, b])
			# embed()
		return F.softmax(attn_energies, dim=-1).unsqueeze(1)

	def score(self, hidden, encoder_output):
		# dot product attention
		return torch.matmul(hidden, torch.t(encoder_output))
		# torch.sum(hidden[:,b]*encoder_outputs[:,b],dim=1)
		
		# energy = hidden.dot(encoder_output)
		# return energy

		# print('score func')
		# embed()
		# if self.method_str == 'dot':
		# 	energy = hidden.dot(encoder_output)
		# 	return energy

		# energy = self.attn(encoder_output)
		# energy = hidden.dot(energy)	
		# return energy

		# elif self.method_str == 'general':
		# 	energy = self.attn(encoder_output)
		# 	energy = hidden.dot(energy)
		# 	embed()
		# 	return energy
		# elif self.method_str == 'concat':
		# 	energy = self.attn(torch.cat((hidden, encoder_output), 1))
		# 	energy = self.v.dot(energy)
		# 	return energy

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, hidden_size, vocab_size, n_layers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.n_layers = n_layers
		self.dropout = dropout
		self.attn_weights = None

		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.embedding_dropout = nn.Dropout(dropout)

		self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=dropout)
		self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)

		self.out = nn.Linear(self.hidden_size, self.vocab_size)

		if self.attn_model != 'none':
			self.attn = Attn(self.attn_model, self.hidden_size)

	def forward(self, input_seq, last_hidden, encoder_outputs):

		batch_size = input_seq.size(0)
		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.hidden_size)

		rnn_output, hidden = self.gru(embedded, last_hidden)

		attn_weights = self.attn(rnn_output, encoder_outputs)

		# linear combination of encoder outputs weighted by attention
		# padded chars = 0, so 0 * attn_weight = 0
		# takes care of variable length sequences
		context = attn_weights.bmm(encoder_outputs.transpose(0,1))

		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))

		output = self.out(concat_output)

		# save attn weights
		self.attn_weights = attn_weights

		return output, hidden
