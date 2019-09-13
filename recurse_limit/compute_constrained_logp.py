'''given csv file in spaced out selfies, compute population level qed

1. unconstrained mean logp (avg logp of max(cands/seed) across all compounds)
	- only condition is value must be greater than seed
2. constrained mean logp (tanimoto similarity > 0.4)
	- value must be greater than seed
	- tanimoto similarity must be greater than 0.4
	- plot number of compounds that meet threshold as a function of time
3. optimal logp (with constraint)
	- best logp compound for each seed compound up to that iteration (e.g. as recursive iterations increase, do compounds improve).
	- do we need tanimoto similarity here?


do above experiments for single output per seed AND k outputs per seed. in the k outputs, we compute k new translations but rank
k best from k*k and then take max for optimal logp. take the max logp compound but always propagate k compounds to the next step.

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

xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/greedy'
logp_means = []
logp_stds = []
prop_valid = []
num_samples = []
start = 0
end = 12
filenums = np.arange(start, end)
seeds = pd.read_csv(xdir+'/'+str(0)+'.csv', header=None, skip_blank_lines=False)
for num in filenums:
	X = pd.read_csv(xdir+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
	smiles = []
	local_logp=[]
	for i in range(X.shape[0]):
		sx = decoder(remove_spaces(''.join(X.iloc[i])))
		x_seed = decoder(remove_spaces(''.join(seeds.iloc[i])))
		val = properties.penalized_logp(sx)
		x_seed_val = properties.penalized_logp(x_seed)
		if num == 0:
			local_logp.append(val)
		else:
			sim = mmpa.similarity(sx, x_seed)
			# if compound has improved property value and tm sim > 0.4
			if val > x_seed_val and sim > 0.4:
				local_logp.append(val)
		smiles.append(sx)
	logp_means.append(np.mean(local_logp))
	logp_stds.append(np.std(local_logp))
	prop_valid.append(len(local_logp)/float(X.shape[0]))

embed()