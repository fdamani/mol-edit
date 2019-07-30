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
import random
import time
import math
import sys
sys.path.insert(0, '../../data_analysis/')
import properties
import mmpa
import os

from process_data import *
from selfies import encoder, decoder
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed, display
from torch import optim
from masked_cross_entropy import *


def rl_train(input_batches,
          input_lengths,
          target_batches,
          target_lengths,
          batch_size,
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          reward_func):
	'''
		train a seq2seq model to minimize negative expected reward
		using policy gradient training.
	'''

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

    # need two separate output sequences
    ############## greedy decoding ##############
    greedy_decoded_chars = []
    for t in range(max_target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        output = functional.log_softmax(decoder_output, dim=-1)
        topv, topi = output.data.topk(1)
        decoder_input = topi.view(1)
        if decoder_input.item() == lang.EOS_token:
        	greedy_decoded_chars.append('EOS')
        else:
        	greedy_decoded_chars.append(lang.index2char[decoder_input.item()])


        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    ############## sample decoding ##############
    sampled_decoded_chars = []
    
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
    	m = torch.distributions.Multinomial(logits=output)
    	samp = m.sample()
    	decoder_input = sampl.squeeze().nonzero().view(1)
        
        if decoder_input.item() == lang.EOS_token:
        	sampled_decoded_chars.append('EOS')
        else:
        	sampled_decoded_chars.append(lang.index2char[decoder_input.item()])


    embed()


    # sample a sequence s^ from the decoder.
    # compute grad log p
    # evaluate reward_func(s^)



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