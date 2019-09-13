'''
rank compounds by max logp
this is designed for finding compounds with highest property value
take as input output of translate file with 20 cands per seed

input
arg1: output of translate file with 20 cands per seed
arg2: output file to save ranked compounds
arg3: number of candidates per seed

algorithm
for all cands per seed:
	save cand property value
return cand with max property value
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
top_logp_X = []
num_evaluate = int(X.shape[0] / n_best)
logpvals = []
for i in range(0, n_best*num_evaluate, n_best):
	vals = []
	for j in range(0, n_best):
		itr = i+j
		try:
			sx = decoder(remove_spaces(''.join(X.iloc[itr])))
			vals.append(properties.penalized_logp(sx))
		except:
			vals.append(-100.)
	ind = i+np.argmax(vals)
	logpvals.append(np.max(vals))
	top_logp_X.append(X.iloc[ind].values[0])
pd.DataFrame(top_logp_X).to_csv(output_file, header=None, index=None)
