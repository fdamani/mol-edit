
def evaluateOLD(input_seq, max_length=100):
	input_seqs = [indexes_from_compound(lang, input_seq)]
	input_lengths = len(input_seqs)

	#input_lengths = [len(input_seq)]
	#input_seqs = [indexes_from_compound(lang, input_seq)]
	embed()
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
	embed()
	output_seq = evaluate(input_seq)
	print('>', input_seq)
	print('=', target_seq)
	print('<', output_seq)
