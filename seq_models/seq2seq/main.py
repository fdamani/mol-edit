'''
seq2seq model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import network as net
import rdkit as rd
import pandas as pd
import IPython
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
import selfies
import process_data
import Lang
import masked_cross_entropy
import random
import time
import math

from process_data import *
from selfies import encoder, decoder
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed
from torch import optim
from masked_cross_entropy import *

# read and process data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
seed=0
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.

f = '/home/fdamani/mol-edit/data/qed/train_pairs.txt'
input_seq, output_seq, lang = read_data(f, selfies=True)
input_seq, output_seq, lang = filter_low_resource(input_seq, output_seq, lang)
pairs = pd.DataFrame([input_seq, output_seq]).T

# batch_size=2
# input_var, input_lengths, target_var, target_lengths = random_batch(batch_size, pairs, lx)
# # declare RNN object
# hidden_size = 128
# vocab_size = lx.n_chars

# # initialize encoder/decoder
# enc_net = rnn.EncoderRNN(vocab_size, hidden_size)
# dec_net = rnn.DecoderRNN(vocab_size, hidden_size)
# # put on device
# enc_net.to(device)
# dec_net.to(device)

# # encoder forward pass
# enc_outputs, enc_hidden = enc_net(input_var, input_lengths)

# # prepare decoder inputs
# max_target_length = max(target_lengths)
# dec_input = torch.LongTensor([lx.SOS_token] * batch_size)
# dec_hidden = enc_hidden
# all_dec_outputs = torch.zeros(max_target_length, batch_size, 
# 	dec_net.vocab_size)

# dec_input = dec_input.to(device)
# all_dec_outputs = all_dec_outputs.to(device)
# criterion = nn.NLLLoss()
# loss = 0
# for t in range(max_target_length):
# 	dec_output, dec_hidden = dec_net(dec_input, dec_hidden)
# 	all_dec_outputs[t] = dec_output
# 	dec_input = target_var[t]

# loss = masked_cross_entropy(all_dec_outputs.transpose(0,1).contiguous(),
# 	target_var.transpose(0,1).contiguous(), target_lengths)
# embed()


def train(input_batches,
		  input_lengths, 
		  target_batches, 
		  target_lengths, 
		  encoder, 
		  decoder, 
		  encoder_optimizer, 
		  decoder_optimizer):
	
	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0

	# Run words through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)
	
	# Prepare input and output variables
	decoder_input = torch.LongTensor([lang.SOS_token] * batch_size)
	decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

	max_target_length = max(target_lengths)
	all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.vocab_size)

   
	decoder_input = decoder_input.cuda()
	all_decoder_outputs = all_decoder_outputs.cuda()

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		decoder_output, decoder_hidden = decoder(
			decoder_input, decoder_hidden)

		all_decoder_outputs[t] = decoder_output
		decoder_input = target_batches[t] # Next input is current target

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(
		all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_batches.transpose(0, 1).contiguous(), # -> batch x seq
		target_lengths)
	loss.backward()
	
	# Clip gradient norms
	ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

	# Update parameters with optimizers
	encoder_optimizer.step()
	decoder_optimizer.step()
	print(loss.item())
	return loss.item(), ec, dc

def evaluate(input_seq, max_length=100):
	input_lengths = [len(input_seq)]
	input_seqs = [indexes_from_compound(lang, input_seq)]
	input_batches = torch.LongTensor(input_seqs).transpose(0, 1)
	
	input_batches = input_batches.cuda()
		
	# Set to not-training mode to disable dropout
	encoder.train(False)
	decoder.train(False)
	
	# Run through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

	# Create starting vectors for decoder
	decoder_input = torch.LongTensor([lang.SOS_token]) # SOS
	decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
	

	decoder_input = decoder_input.cuda()

	# Store output words and attention states
	decoded_chars = []
	
	# Run through decoder
	for di in range(max_length):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

		# Choose top word from output
		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0]
		if ni == lang.EOS_token:
			decoded_chars.append('<EOS>')
			break
		else:
			decoded_chars.append(output_lang.index2char[ni])
			
		# Next input is chosen word
		decoder_input = torch.LongTensor([ni])
		decoder_input = decoder_input.cuda()

	# Set back to training mode
	encoder.train(True)
	decoder.train(True)
	
	return decoded_chars

def evaluate_randomly():
	num_samples = pairs.shape[0]
	rand_ind = np.random.choice(num_samples)
	rand_pair = pairs.iloc[rand_ind]
	input_seq, target_seq = rand_pair[0], rand_pair[1]
	output_seq = evaluate(input_seq)
	print('>', input_seq)
	print('=', target_seq)
	print('<', output_seq)



def as_minutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def time_since(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

hidden_size = 56
n_layers = 1
dropout = 0.1
batch_size = 10

clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 1e-3
n_epochs = 10000
epoch = 0
plot_every = 1
print_every = 1
evaluate_every = 1

# initialize models
encoder = net.EncoderRNN(vocab_size=lang.n_chars, 
					 hidden_size=hidden_size, 
					 n_layers=n_layers,
					 dropout=dropout)
decoder = net.DecoderRNN(vocab_size=lang.n_chars,
					 hidden_size=hidden_size,
					 n_layers=n_layers,
					 dropout=dropout)

# initialize optimizers and criterion
encoder_opt = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_opt = optim.Adam(decoder.parameters(), lr=learning_rate)

# move to decide
encoder.to(device)
decoder.to(device)

# keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every


ecs, dcs = [], []
eca = 0
dca = 0

while epoch < n_epochs:
	print(epoch)
	epoch += 1

	# get random batch
	input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, pairs, lang)

	# run train func
	loss, ec, dc = train(input_batches,
					     input_lengths,
					     target_batches,
					     target_lengths,
					     encoder,
					     decoder,
					     encoder_opt,
					     decoder_opt)
	print_loss_total += loss
	plot_loss_total += loss
	eca += ec
	dca += dc

	if epoch % print_every == 0:
		with torch.no_grad():
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / float(n_epochs)), 
				epoch, epoch / n_epochs * 100, print_loss_avg)
	
	# if epoch % evaluate_every == 0:
	# 	with torch.no_grad():
	# 		evaluate_randomly()

	# if epoch % plot_every == 0:
	# 	plot_loss_avg = plot_loss_total / plot_every
	# 	plot_losses.append(plot_loss_avg)
	# 	plot_loss_total = 0
	# 	plt.cla()
	# 	plt.plot(plot_losses)
	# 	plt.savefig('training_loss.png')



