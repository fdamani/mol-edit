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