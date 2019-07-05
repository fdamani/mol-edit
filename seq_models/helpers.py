def process_data(file, isSELFIE=False):
	# read data
	dat = pd.read_csv('/data/potency_dataset_with_props.csv')
	# limit to first 10k samples
	# dat = dat.head(1000)
	
	# permute rows
	dat = dat.sample(frac=1).reset_index(drop=True)

	structure = dat['Structure']
	pot = dat['pot_uv']
	if isSELFIE:
		structure, pot, invalid_selfies = utils.encode_to_selfies(structure, pot)

	# take -log10 of potency. we want higher values to mean more potent.
	Y = -torch.log10(torch.tensor(pot, device=device))
	if isSELFIE:
		chars = utils.unique_selfies_chars(structure)
	else:
		chars = utils.unique_chars(structure)
	n_chars = len(chars)
	n_samples = len(structure)
	chars_to_int = {}
	int_to_chars = {}
	for i in range(len(chars)):
		chars_to_int[chars[i]] = i
		int_to_chars[i] = chars[i]

	X = []
	for struct in structure:
		indices = utils.lineToIndices(struct, chars_to_int, isSELFIE)
		X.append(indices)

	return X, Y, n_chars, n_samples

def train_test_split(X, Y, n_samples, train_percent=.7):
	train_ind = int(n_samples * train_percent)
	X_train, X_test = [], []
	
	X_train, Y_train = X[0:train_ind], Y[0:train_ind]
	X_test, Y_test = X[train_ind:], Y[train_ind:]
	return X_train, Y_train, X_test, Y_test


def get_mini_batch(X, Y, indices):
	# get minibatch from rand indices
	batch_x, batch_y = [X[ind] for ind in indices], Y[indices]
	return batch_x, batch_y

def get_packed_batch(batch_x, batch_y):
	# get length of each sample in mb
	batch_lens = [len(sx) for sx in batch_x]
	# arg sort batch lengths in descending order
	sorted_inds = np.argsort(batch_lens)[::-1]
	batch_x = [batch_x[sx] for sx in sorted_inds]
	batch_y = torch.stack([batch_y[sx] for sx in sorted_inds])

	# pack x
	batch_packed_x = nn.utils.rnn.pack_sequence([torch.LongTensor(s) for s in batch_x])
	batch_packed_x = batch_packed_x.to('cuda')
	return batch_packed_x, batch_y