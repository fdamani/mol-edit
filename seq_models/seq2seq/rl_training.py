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

def rl_train_outer_loop(input_batches,
					  input_lengths,
					  target_batches,
					  target_lengths,
					  batch_size,
					  encoder,
					  decoder,
					  encoder_optimizer,
					  decoder_optimizer,
					  reward_func,
					  lang,
					  clip):
	''' loop over minibatch. for each sample pass to rl_train_inner_loop to compute
	loss. sum over samples then call backward'''
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	b_size = input_batches.shape[1]
	loss = 0
	count = 0
	for i in range(b_size):
		sample_loss = rl_train_inner_loop(input_batches[:,i][:,None],
										[input_lengths[i]],
										target_batches[:,i][:,None],
										[target_lengths[i]],
										1,
										encoder,
										decoder,
										reward_func,
										lang)
		if sample_loss is not None:
			loss += sample_loss
			count += 1
	loss = loss / float(count)
	print('count: ', count)
	try:
		loss.backward()
	except:
		return 0.0, 0.0, 0.0
	ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
	encoder_optimizer.step()
	decoder_optimizer.step()
	return loss.item(), ec, dc

def rl_train_inner_loop(input_batches,
		  input_lengths,
		  target_batches,
		  target_lengths,
		  batch_size,
		  encoder,
		  decoder,
		  reward_func,
		  lang):
	'''
		train a seq2seq model to minimize negative expected reward
		using policy gradient training.
		note: only works for batch of size 1.
	'''
	input_smiles = []
	for i in range(input_lengths[0]):
		input_smiles.append(lang.index2char[input_batches[i].item()])
	input_smiles = selfies.decoder(''.join(input_smiles[:-1]))
	
	target_smiles = []
	for i in range(target_lengths[0]):
		target_smiles.append(lang.index2char[target_batches[i].item()])	
	target_smiles = selfies.decoder(''.join(target_smiles[:-1]))

	# Run words through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

	max_target_length = max(target_lengths)

	############## greedy decoding ##############
	with torch.no_grad():
		# Prepare input and output variables
		decoder_input = torch.LongTensor([lang.SOS_token] * batch_size)
		# Use last (forward) hidden state from encoder
		decoder_hidden = encoder_hidden[:decoder.n_layers]

		all_decoder_outputs = torch.zeros(
			max_target_length, batch_size, decoder.vocab_size)

		if torch.cuda.is_available():
			decoder_input = decoder_input.cuda()
			all_decoder_outputs = all_decoder_outputs.cuda()

		# no gradient here
		greedy_decoded_chars = []
		for t in range(max_target_length):
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden, encoder_outputs)
			output = functional.log_softmax(decoder_output, dim=-1)
			topv, topi = output.data.topk(1)
			decoder_input = topi.view(1)
			if decoder_input.item() == lang.EOS_token:
				greedy_decoded_chars.append('EOS')
				break
			else:
				greedy_decoded_chars.append(lang.index2char[decoder_input.item()])

			all_decoder_outputs[t] = decoder_output
		greedy_smiles = selfies.decoder(''.join(greedy_decoded_chars[:-1]))
		if greedy_smiles == -1:
			return None
		if Chem.MolFromSmiles(greedy_smiles) is None:
			return None
	############## sample decoding ##############
	# this needs to be fixed
	# we only want to retain the computation graph for the decoded input we use
	# otherwise might max out on memory

	sampled_decoded_chars = []
	sampled_decoded_indices = []
	logits = []
	
	# Prepare input and output variables
	decoder_input = torch.LongTensor([lang.SOS_token] * batch_size)
	# Use last (forward) hidden state from encoder
	decoder_hidden = encoder_hidden[:decoder.n_layers]
	all_decoder_outputs = torch.zeros(
		max_target_length, batch_size, decoder.vocab_size)
	
	if torch.cuda.is_available():
		decoder_input = decoder_input.cuda()
		all_decoder_outputs = all_decoder_outputs.cuda()

	for t in range(max_target_length):
		decoder_output, decoder_hidden = decoder(
			decoder_input, decoder_hidden, encoder_outputs)
		output = functional.log_softmax(decoder_output, dim=-1)
		logits.append(output)
		m = torch.distributions.Multinomial(logits=output)
		samp = m.sample()
		decoder_input = samp.squeeze().nonzero().view(1)
		sampled_decoded_indices.append(decoder_input)
		
		if decoder_input.item() == lang.EOS_token:
			sampled_decoded_chars.append('EOS')
			break
		else:
			sampled_decoded_chars.append(lang.index2char[decoder_input.item()])

	sampled_smiles = selfies.decoder(''.join(sampled_decoded_chars[:-1]))
	if sampled_smiles == -1:
		return None
	if Chem.MolFromSmiles(sampled_smiles) is None:
		return None

	############## reward ##############
	supervised_rw = reward_func(input_smiles, target_smiles)
	sampled_rw = reward_func(input_smiles, sampled_smiles) - supervised_rw
	baseline_rw = reward_func(input_smiles, greedy_smiles) - supervised_rw

	############## compute loss #########
	# compute log p(y^s|x)
	loss = 0
	criterion = nn.CrossEntropyLoss()
	sampled_decoded_indices = torch.stack(sampled_decoded_indices)
	logits = torch.stack(logits)
	for i in range(sampled_decoded_indices.shape[0]):
		loss += criterion(logits[i].squeeze(dim=0), sampled_decoded_indices[i])

	loss = loss * (sampled_rw - baseline_rw)
	return loss

	# # sample a sequence s^ from the decoder.
	# # compute grad log p
	# # evaluate reward_func(s^)



	# # Run through decoder one time step at a time
	# for t in range(max_target_length):
	#   decoder_output, decoder_hidden = decoder(
	#       decoder_input, decoder_hidden, encoder_outputs)

	#   all_decoder_outputs[t] = decoder_output
	#   decoder_input = target_batches[t]  # Next input is current target

	# # Loss calculation and backpropagation
	# loss = masked_cross_entropy(
	#   all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
	#   target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
	#   target_lengths)

	# loss.backward()

	# # Clip gradient norms
	# ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	# dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

	# # Update parameters with optimizers
	# encoder_optimizer.step()
	# decoder_optimizer.step()

	# return loss.item(), ec, dc