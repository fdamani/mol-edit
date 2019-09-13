'''
this ranking strategy is designed to help generate candidates
with lower on average molecular weight. it is based on the observation
that generated compounds tend to have higher mol wt than training data.

input
arg1: output of translate file with 20 cands per seed
arg2: output file to save ranked compounds
arg3: number of candidates per seed

algorithm:
for all candidates that have logp > prev iter seed logp
	save mol wt of cand
return cand with min mol wt
'''

import pandas as pd
import selfies
import numpy as np
from selfies import encoder, decoder
from IPython import embed
import sys
sys.path.insert(0, '../data_analysis')
import mmpa
import properties
import rdkit
from rdkit import Chem

def remove_spaces(x):
	return x.replace(" ", "")

X = pd.read_csv(sys.argv[1], header=None, skip_blank_lines=False)
output_file = sys.argv[2]
n_best = int(sys.argv[3])
X_prev = pd.read_csv(sys.argv[4], header=None, skip_blank_lines=False)
top_logp_X = []
num_evaluate = int(X.shape[0] / n_best)
for i in range(0, n_best*num_evaluate, n_best):
	vals = []
	prev_ind = int(i / n_best)
	sx_prev = decoder(remove_spaces(''.join(X_prev.iloc[prev_ind])))
	for j in range(0, n_best):
		itr = i+j
		try:
			sx = decoder(remove_spaces(''.join(X.iloc[itr])))
			if properties.penalized_logp(sx) > properties.penalized_logp(sx_prev):
				vals.append(rdkit.Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(sx)))
		except:
			# append a large value
			vals.append(100000.)
	try:
		ind = i+np.argmin(vals)
	except:
		ind = i+0
	top_logp_X.append(X.iloc[ind].values[0])
pd.DataFrame(top_logp_X).to_csv(output_file, header=None, index=None)
