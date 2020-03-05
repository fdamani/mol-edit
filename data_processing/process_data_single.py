"""Take as input test.txt file.
	
- input is single selfies per line
- convert to selfies
- save src/target separately with space-separated tokens, one compound per line


Example command: python process_data.py ../data/logp04/train_pairs.txt ../data/logp04"""

import torch
import numpy as np
import pandas as pd
import sys
import selfies
import re

from selfies import encoder, decoder
from IPython import embed, display

def read_data(file, selfies=True):
	data = pd.read_csv(file, delimiter=' ', header=None)
	#data = data.head(1000)
	data = data.sample(frac=1).reset_index(drop=True)
	src = pd.DataFrame(smiles_to_selfies(data[0])).dropna()
	return src

def smiles_to_selfies(x, token_sep=True):
	"""smiles to selfies
	if token_sep=True -> return spaces between each token"""
	output = []
	for i in range(x.shape[0]):
		ax = encoder(x[i])
		if ax != -1:
			if token_sep:
				sx = re.findall(r"\[[^\]]*\]", ax)
				ax = ' '.join(sx)
			output.append(ax)
		else:
			output.append('NaN')
	return output
if __name__ == '__main__':
	file = sys.argv[1]
	output_dir = sys.argv[2]
	src = read_data(file)
	fname = file.split('/')[-1]
	# save to dict
	src.to_csv(output_dir+'/'+fname,index=None, header=None)
