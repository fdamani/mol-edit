
def evaluate_beam(input_batches,
                  input_lengths,
                  batch_size,
                  encoder,
                  decoder,
                  search='beam',
                  num_beams=20):
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
    beam_char_inds = torch.zeros(
        num_beams, max_target_length, dtype=torch.long, device=device)
    beam_log_probs = torch.zeros(num_beams, max_target_length, device=device)
    beam_hiddens = torch.zeros(num_beams, decoder_hidden.shape[0], decoder_hidden.shape[1],
                               decoder_hidden.shape[2], device=device)
    final_beams = []
    beam_probs = []

    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

    for i in range(num_beams):
        beam_hiddens[i] = decoder_hidden

    log_probs = functional.log_softmax(decoder_output, dim=-1)
    topv, topi = log_probs.data.topk(num_beams)
    beam_char_inds[:, 0] = topi.squeeze()
    beam_log_probs[:, 0] = topv.squeeze()

    # Run through decoder one time step at a time
    for t in range(1, max_target_length):
        # if all beams have ended break
        if num_beams == 0:
            break

        beam_cand_log_probs = []
        beam_cand_inds = []

        for i in range(num_beams):
            decoder_input = beam_char_inds[i, t-1].unsqueeze(dim=0)
            # check if decoder input is ever lang.EOS_token. if it is there is a bug!
            decoder_hidden = beam_hiddens[i]
            decoder_output, beam_hiddens[i] = decoder(
                decoder_input, decoder_hidden)
            log_probs = functional.log_softmax(decoder_output, dim=-1)
            log_probs += beam_log_probs[i, t-1]
            topv, topi = log_probs.data.topk(num_beams)
            beam_cand_log_probs.append(topv.squeeze())
            beam_cand_inds.append(topi.squeeze())

        # if greater than one beam -> concatenate
        if num_beams > 1:
            beam_cand_log_probs = torch.cat(beam_cand_log_probs)
            beam_cand_inds = torch.cat(beam_cand_inds)
        else:
            beam_cand_log_probs = torch.stack(beam_cand_log_probs)
            beam_cand_inds = torch.stack(beam_cand_inds)

        # pick topk
        a, b = beam_cand_log_probs.topk(num_beams)

        for i in range(num_beams):
            what_beam = b[i] / num_beams
            char_ind = beam_cand_inds[b[i]]
            log_prob = beam_cand_log_probs[b[i]]

            # copying over particles
            beam_char_inds[i] = beam_char_inds[what_beam]
            beam_char_inds[i, t] = char_ind

            beam_log_probs[i] = beam_log_probs[what_beam]
            beam_log_probs[i, t] = log_prob

            beam_hiddens[i] = beam_hiddens[what_beam]

        # save trajectories that have ended
        ax = beam_char_inds[beam_char_inds[:, t] == lang.EOS_token]
        if ax.shape[0] != 0:
            final_beams.append(ax)

        # keep trajectories that have not ended
        ax = beam_char_inds[beam_char_inds[:, t] != lang.EOS_token]
        if ax.shape[0] != 0:
            inds = beam_char_inds[:, t] != lang.EOS_token
            beam_char_inds = beam_char_inds[inds]
            beam_log_probs = beam_log_probs[inds]
            beam_hiddens = beam_hiddens[inds]
        # no trajectories left
        else:
            num_beams = 0
            break
        num_beams = beam_char_inds.shape[0]
        print num_beams

    embed()

    # NEED TO DEBUG BELOW.
    # append beams that have reached max length
    if num_beams > 0:
        final_beams.append(beam_char_inds)

    embed()

    # decode beams
    decoded_chars = []
    for beam_set in final_beams:
        for beam in beam_set:
            decoded_chars.append(
                ''.join([lang.index2char[beam[i].item()] for i in range(len(beam))][:-1]))

    embed()
    return decoded_chars

    # 	# max
    # 	if search == 'greedy':
    # 		topv, topi = output.data.topk(1)
    # 		decoder_input = topi.view(1)
    # 	else:
    # 		print("ERROR: please specify greedy or beam.")
    # 		sys.exit(0)

    # 	# # sample
    # 	# else:
    # 	# 	m = torch.distributions.Multinomial(logits=output)
    # 	# 	samp = m.sample()
    # 	# 	decoder_input = samp.squeeze().nonzero().view(1)

    # 	if decoder_input.item() == lang.EOS_token:
    # 		decoded_chars.append('EOS')
    # 		break
    # 	else:
    # 		decoded_chars.append(lang.index2char[decoder_input.item()])

    # 	if t == (max_target_length-1):
    # 		print('Failed to return EOS token.')

    # decoded_str = ''.join(decoded_chars[:-1])

    return decoded_str