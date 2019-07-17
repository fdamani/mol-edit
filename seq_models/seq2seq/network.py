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


# class Seq2Seq(nn.Module):
# 	def __init__(self, vocab_size, embed_size, hidden_size):
# 		super(Seq2Seq, self).__init__()
# 		self.hidden_size = hidden_size
# 		self.emb_enc = nn.Embedding(vocab_size, embed_size)
# 		self.gru_enc = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
# 		self.out_enc = nn.Linear(hidden_size, hidden_size)
		

# 		self.emb_dec = nn.Embedding(vocab_size, embed_size)
# 		self.gru_dec = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
# 		self.out_dec = nn.Linear(hidden_size, vocab_size)

# 	def encoder(self, inp):
# 		'''
# 			inp: input
# 		'''
# 		b_size = inp.batch_sizes[0].item()
# 		h = self.initHidden(b_size)
# 		emb = nn.utils.rnn.PackedSequence(self.embedding(inp.data), 
# 			inp.batch_sizes)
# 		_, h = self.gru_enc(emb, h)
# 		h = self.out_enc(h)
# 		return h

# 	def decoder(self, dec_inp, h):
# 		'''
# 			dec_inp: decoder input
# 			h: hidden state
# 		'''
# 		emb = nn.utils.rnn.PackedSequence(self.embedding(dec_inp.data), 
# 			dec_inp.batch_sizes)
		
# 		emb = self.gru_dec(emb, h)






