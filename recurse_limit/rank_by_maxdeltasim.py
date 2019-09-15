'''
rank compounds by similarity with previous step
this is designed for highly interpretable traces (few steps between steps)
take as input output of translate file with 20 cands per seed

input
arg1: output of translate file with 20 cands per seed
arg2: output file to save ranked compounds
arg3: number of candidates per seed
arg4: previous iter seed compounds

algorithm
for all cands per seed that have higher logp compared to prev iter seed:
	save sim(cand, prev iter seed)
return cand with highest sim with prev iter seed
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


def remove_spaces(x):
	return x.replace(" ", "")

X = pd.read_csv(sys.argv[1], header=None, skip_blank_lines=False)
output_file = sys.argv[2]
n_best = int(sys.argv[3])
X_prev = pd.read_csv(sys.argv[4], header=None, skip_blank_lines=False)
top_logp_X = []
num_evaluate = int(X.shape[0] / n_best)
logpvals = []
for i in range(0, n_best*num_evaluate, n_best):
	vals = []
	prev_ind = int(i / n_best)
	sx_prev = decoder(remove_spaces(''.join(X_prev.iloc[prev_ind])))
	for j in range(0, n_best):
		itr = i+j
		try:
			sx = decoder(remove_spaces(''.join(X.iloc[itr])))
			if properties.penalized_logp(sx) > properties.penalized_logp(sx_prev):
				vals.append(mmpa.similarity(sx_prev, sx))
			else:
				vals.append(-100.)
		except:
			vals.append(-100.)
	ind = i+np.argmax(vals)
	logpvals.append(properties.penalized_logp(decoder(remove_spaces(''.join(X.iloc[ind])))))
	top_logp_X.append(X.iloc[ind].values[0])
pd.DataFrame(top_logp_X).to_csv(output_file, header=None, index=None)