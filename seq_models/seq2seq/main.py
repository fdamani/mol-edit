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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
import selfies
import process_data
import Lang
import masked_cross_entropy
import utils
import random
import time
import math
import sys
sys.path.insert(0, '../../data_analysis/')
import properties
import mmpa
import rl_training
import os

from process_data import *
from selfies import encoder, decoder
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed, display
from torch import optim
from masked_cross_entropy import *
from utils import similarity
from rl_training import rl_train

# read and process data
#output_dir_str = '/home/fdamani/mol-edit/output/'
output_dir_str = '/tigress/fdamani/mol-edit-output/'
output_dir = output_dir_str + \
		str(sys.argv[1])+'_'+str(time.time())
os.mkdir(output_dir)
os.mkdir(output_dir+'/figs')
os.mkdir(output_dir+'/saved_model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.

pytorch_load = True
if pytorch_load:
		X_train, Y_train, X_valid, Y_valid, lang = torch.load(
				'/home/fdamani/mol-edit/data/qed/pytorch_data.pth')
else:
		f = '/home/fdamani/mol-edit/data/qed/train_pairs.txt'
		input_seq, output_seq, lang = read_data(f, selfies=True)
		input_seq, output_seq, lang = filter_low_resource(
				input_seq, output_seq, lang)
		X_train, Y_train, X_valid, Y_valid = train_test_split(
				input_seq, output_seq)
		data = [X_train, Y_train, X_valid, Y_valid, lang]
		torch.save(data, '/home/fdamani/mol-edit/data/qed/pytorch_data.pth')

pairs = pd.DataFrame([X_train, Y_train]).T
valid_pairs = pd.DataFrame([X_valid, Y_valid]).T


# sim = []
# for i in range(pairs.shape[0]):
# 	sim.append(mmpa.similarity(decoder(pairs.iloc[i, 0]), decoder(pairs.iloc[i, 1])))
# plt.cla()
# plt.hist(sim)
# plt.ylabel('Count')
# plt.xlabel('Similarity')
# plt.savefig('/home/fdamani/mol-edit/data/qed/training_sim_hist.png')
# embed()
# test data
#f = '/home/fdamani/mol-edit/data/qed/valid.txt'
#test_seqs, lengths = read_single_data(f, lang, selfies=True)


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
					batch_size,
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
				decoder_input = target_batches[t]  # Next input is current target

		# Loss calculation and backpropagation
		loss = masked_cross_entropy(
				all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
				target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
				target_lengths)

		loss.backward()

		# Clip gradient norms
		ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
		dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

		# Update parameters with optimizers
		encoder_optimizer.step()
		decoder_optimizer.step()

		return loss.item(), ec, dc


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

def as_minutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def time_since(since, percent):
	now = time.time()
	s = now - since
	es = s / (float(percent))
	rs = es - s
	return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


embed_size = 128
hidden_size = 256
n_layers = 1
dropout = 0.5
batch_size = 1 #128
#valid_batch_size = len(valid_pairs)
valid_batch_size = 128
evaluate_batch_size = 1
num_evaluate = 1000
similarity_thresh = .4
qed_target = .9

clip = 5.0
teacher_forcing_ratio = 0.5
learning_rate = 1e-3
n_epochs = 500000
epoch = 0
plot_every = 10  # 2000
print_every = 10  # 2000
valid_every = 10  # 2000
evaluate_every = 10  # 2000
save_every = 1000
attn = False
is_rl_train = True
# load saved weights
pytorch_weights_load = True
# initialize models
encoder = net.EncoderRNN(vocab_size=lang.n_chars,
						 embed_size=embed_size,
						 hidden_size=hidden_size,
						 n_layers=n_layers,
						 dropout=dropout)
if attn:
	attention = net.Attn(method_str='general', hidden_size=hidden_size)
	decoder = net.LuongAttnDecoderRNN(attn_model=attention,
									hidden_size=hidden_size,
									vocab_size=lang.n_chars)
else:
	decoder = net.DecoderRNN(vocab_size=lang.n_chars,
							 embed_size=embed_size,
							 hidden_size=hidden_size,
							 n_layers=n_layers,
							 dropout=dropout)

if pytorch_weights_load:
	encoder.load_state_dict(torch.load('/tigress/fdamani/mol-edit-output/xent_train_128_256_1564678283.77/saved_model/encoder_99600.pth'))       
	decoder.load_state_dict(torch.load('/tigress/fdamani/mol-edit-output/xent_train_128_256_1564678283.77/saved_model/decoder_99600.pth'))

# initialize optimizers and criterion
encoder_opt = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_opt = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-5)

# move to decide
encoder.to(device)
decoder.to(device)

# keep track of time elapsed and running averages
start = time.time()
plot_train_losses = []
plot_valid_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every
percent_valid_decoded, percent_similar, percent_in_target, valid_input_qed, valid_decode_qed, percent_success = [], [], [], [], [], []

ecs, dcs = [], []
eca = 0
dca = 0

while epoch < n_epochs:

	epoch += 1
	if epoch % 10 == 0:
		print(epoch)
	# get random batch
	input_batches, input_lengths, target_batches, target_lengths = random_batch(
			batch_size, pairs, lang)
	if is_rl_train:
		loss, ec, dc = rl_train(input_batches,
								input_lengths,
								target_batches,
								target_lengths,
								batch_size,
								encoder,
								decoder,
								encoder_opt,
								decoder_opt,
								utils.similarity,
								lang)
	
	else:
		loss, ec, dc = train(input_batches,
							 input_lengths,
							 target_batches,
							 target_lengths,
							 batch_size,
							 encoder,
							 decoder,
							 encoder_opt,
							 decoder_opt)
	print_loss_total += loss
	plot_loss_total += loss
	eca += ec
	dca += dc

	if epoch % evaluate_every == 0:
			with torch.no_grad():
					delta_qed = []
					input_qed_list = []
					decoded_qed_list = []
					similarity = []
					valid = []
					input_strs = []
					decoded_strs = []
					input_diversity, decoded_diversity = [], []
					# evaluate samples one-by-one
					for i in range(num_evaluate):
							input_batches, input_lengths, target_batches, target_lengths = random_batch(
									evaluate_batch_size, valid_pairs.iloc[i:i+1], lang, replace=False)

							# decoded_str = evaluate_beam(input_batches,
							# 				input_lengths,
							# 				evaluate_batch_size,
							# 				encoder,
							# 				decoder)
							decoded_str = evaluate(input_batches,
												   input_lengths,
												   evaluate_batch_size,
												   encoder,
												   decoder,
												   search='greedy')
							decoded_str = selfies.decoder(decoded_str)
							# invalid string
							if decoded_str == -1:
									continue
							input_str = selfies.decoder(
									''.join(compound_from_indexes(lang, input_batches)[:-1]))
							input_qed = properties.qed(input_str)
							decoded_qed = properties.qed(decoded_str)
							# if invalid (qed value = 0)
							if input_qed == 0.0 or decoded_qed == 0.0:
									valid.append(0)
									continue

							input_strs.append(input_str)
							decoded_strs.append(decoded_str)
							decoded_qed_list.append(decoded_qed)
							input_qed_list.append(input_qed)
							delta_qed.append(decoded_qed - input_qed)
							similarity.append(mmpa.similarity(input_str, decoded_str))
							valid.append(1)
					print('similarity: ', np.mean(np.array(similarity)))
					# print('delta qed mean: ', np.mean(delta_qed), \
					# 	  ' std: ', np.std(delta_qed), \
					# 	  'percent valid decoded: ', np.mean(valid), \
					# 	  'percent similar: ', np.mean(np.array(similarity) > similarity_thresh), \
					# 	  'percent in target range: ', np.mean(np.array(decoded_qed_list) > qed_target))

					percent_valid_decoded.append(np.mean(valid))
					percent_similar.append(
							np.mean(np.array(similarity) > similarity_thresh))
					percent_in_target.append(
							np.mean(np.array(decoded_qed_list) > qed_target))

					# compute 'success rate' if translation satisfies similarity constraint and property score falls in target range
					# note this is divided by number of valid decodings, not total number of evaluations.
					sx = np.array(similarity)[np.array(decoded_qed_list) > qed_target]
					num_success = len(sx[sx > similarity_thresh])
					if len(similarity) == 0:
						success_rate = 0
					else:
						success_rate = float(num_success) / len(similarity)
					percent_success.append(success_rate)
					# input_diversity.append(mmpa.population_diversity(input_strs))
					# decoded_diversity.append(mmpa.population_diversity(decoded_strs))

	if epoch % valid_every == 0:
			with torch.no_grad():
					input_batches, input_lengths, target_batches, target_lengths = random_batch(
							valid_batch_size, valid_pairs, lang, replace=False)
					valid_loss = validate(input_batches,
										input_lengths,
										target_batches,
										target_lengths,
										valid_batch_size,
										encoder, decoder,
										teacher_forcing=True)

	if epoch % print_every == 0:
			with torch.no_grad():
					print_loss_avg = print_loss_total / print_every
					print_loss_total = 0
					print_summary = '%s (%d %d%%) %.4f %.4f' % (time_since(start, epoch / float(n_epochs)),
																											epoch, float(epoch) / n_epochs * 100, print_loss_avg, valid_loss)
					print(print_summary)

	if epoch % plot_every == 0:
			with torch.no_grad():
					plot_train_losses.append(print_loss_avg)
					plot_valid_losses.append(valid_loss)
					plt.cla()
					plt.plot(plot_train_losses, label='train')
					plt.plot(plot_valid_losses, label='valid')
					plt.xlabel('Iter')
					plt.ylabel('NLL')
					plt.legend(loc='lower right')
					plt.savefig(output_dir+'/figs/loss.png')

					plt.cla()
					plt.plot(percent_valid_decoded)
					plt.xlabel('Iter')
					plt.ylabel('Percent Valid Decoded')
					plt.savefig(output_dir+'/figs/percent_valid_decoded.png')

					plt.cla()
					plt.plot(percent_similar)
					plt.xlabel('Iter')
					plt.ylabel('Percent Tanimoto Similar Above ' +
										 str(similarity_thresh))
					plt.savefig(output_dir+'/figs/percent_similar.png')

					plt.cla()
					plt.plot(percent_in_target)
					plt.xlabel('Iter')
					plt.ylabel('Percent Decoded in Target Range > ' + str(qed_target))
					plt.savefig(output_dir+'/figs/percent_in_target.png')

					plt.cla()
					plt.plot(percent_success)
					plt.xlabel('Iter')
					plt.ylabel('Percent Success Similarity and Property Goal')
					plt.savefig(output_dir+'/figs/percent_success.png')

					plt.cla()
					plt.hist(input_qed_list, label='Input')
					plt.hist(decoded_qed_list, label='Output')
					plt.xlabel("QED")
					plt.legend()
					plt.savefig(output_dir+'/figs/hist_qed.png')

					plt.cla()
					plt.hist(similarity)
					plt.xlabel("Tanimoto Similarity")
					plt.savefig(output_dir+'/figs/hist_tanimoto_similarity.png')

					# plt.cla()
					# plt.hist(input_diversity, label='Input')
					# plt.hist(decoded_diversity, label='Output')
					# plt.xlabel("Diversity")
					# plt.legend()
					# plt.savefig(output_dir+'/figs/hist_diversity.png')

	if epoch % save_every == 0:
			with torch.no_grad():
					# save model
					torch.save(encoder.state_dict(), output_dir +
										 '/saved_model/encoder_'+str(epoch)+'.pth')
					torch.save(decoder.state_dict(), output_dir +
										 '/saved_model/decoder_'+str(epoch)+'.pth')
