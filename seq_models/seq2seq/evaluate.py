import torch
import torch.nn as nn
import torch.nn.functional as F
import network as net
import rdkit as rd
import pandas as pd
import IPython
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
import selfies
import process_data
import masked_cross_entropy
import random
import time
import math
import sys
sys.path.insert(0, '../../data_analysis/')
import properties
import mmpa
import os
import rdkit

from process_data import *
from selfies import encoder, decoder
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed, display
from torch import optim
from masked_cross_entropy import *
from rdkit import Chem

def evaluate(input_batches,
						 input_lengths,
						 batch_size,
						 encoder,
						 decoder,
						 search='greedy'):
		'''
						:param search: decoding search strategies
										{"greedy", "beam"}
		'''
		loss = 0
		# Run words through encoder
		encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

		# Prepare input and output variables
		decoder_input = torch.LongTensor([lang.SOS_token] * batch_size)
		# Use last (forward) hidden state from encoder
		decoder_hidden = encoder_hidden[:decoder.n_layers]

		max_target_length = 100
		all_decoder_outputs = torch.zeros(
				max_target_length, batch_size, decoder.vocab_size)

		if torch.cuda.is_available():
			decoder_input = decoder_input.cuda()
			all_decoder_outputs = all_decoder_outputs.cuda()

		decoded_chars = []

		# Run through decoder one time step at a time
		for t in range(max_target_length):
			decoder_output, decoder_hidden = decoder(
					decoder_input, decoder_hidden, encoder_outputs)
			output = functional.log_softmax(decoder_output, dim=-1)
			# max
			if search == 'greedy':
				topv, topi = output.data.topk(1)
				decoder_input = topi.view(1)
			else:
				print("ERROR: please specify greedy or beam.")
				sys.exit(0)

			# # sample
			# else:
			# 	m = torch.distributions.Multinomial(logits=output)
			# 	samp = m.sample()
			# 	decoder_input = samp.squeeze().nonzero().view(1)

			if decoder_input.item() == lang.EOS_token:
				decoded_chars.append('EOS')
				break
			else:
				decoded_chars.append(lang.index2char[decoder_input.item()])

			# if t == (max_target_length-1):
			# 	print('Failed to return EOS token.')

		decoded_str = ''.join(decoded_chars[:-1])
		return decoded_str


def validate(input_batches,
						 input_lengths,
						 target_batches,
						 target_lengths,
						 batch_size,
						 encoder,
						 decoder,
						 teacher_forcing=True):
		loss = 0
		# Run words through encoder
		encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

		# Prepare input and output variables
		decoder_input = torch.LongTensor([lang.SOS_token] * batch_size)
		# Use last (forward) hidden state from encoder
		decoder_hidden = encoder_hidden[:decoder.n_layers]

		max_target_length = max(target_lengths)
		all_decoder_outputs = torch.zeros(
				max_target_length, batch_size, decoder.vocab_size)

		if torch.cuda.is_available():
				decoder_input = decoder_input.cuda()
				all_decoder_outputs = all_decoder_outputs.cuda()

		# Run through decoder one time step at a time
		for t in range(max_target_length):
				decoder_output, decoder_hidden = decoder(
						decoder_input, decoder_hidden, encoder_outputs)

				all_decoder_outputs[t] = decoder_output
				if teacher_forcing:
						# Teacher forcing: next input is current target
						decoder_input = target_batches[t]

		# Loss calculation
		loss = masked_cross_entropy(
				all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
				target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
				target_lengths)

		return loss.item()