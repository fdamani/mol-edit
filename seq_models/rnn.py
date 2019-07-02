from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def forward_full(self, input, hidden):
		hidden = self.initHidden()
		embed()


	def score(self, hidden):
		'''
			fully connected layer outputting a single scalar value
		'''
		return self.fc(hidden)

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)