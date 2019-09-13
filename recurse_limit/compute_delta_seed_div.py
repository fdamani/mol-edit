'''given csv file in spaced out selfies, compute avg diversity of compounds relative to seed

this is a direct measure of "decay".


this file needs to be edited.

'''

import pandas as pd
import selfies
import numpy as np
from selfies import encoder, decoder
from IPython import embed
import sys
sys.path.insert(0, '../data_analysis')
import mmpa

def remove_spaces(x):
	return x.replace(" ", "")
xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop2/'
rank_types=["maxdeltasim", "logp"]
labels = ['Max Iter Sim', "LogP"]

delta_means = []
delta_stds = []
num_samples = []

# read in seed data
seeds = pd.read_csv(xdir+'/'+str(0)+'.csv', header=None, skip_blank_lines=False)
smiles_seeds = []
for i in range(seeds.shape[0]):
	smiles_seeds.append(decoder(remove_spaces(''.join(seeds.iloc[i]))))

start = 1
end = 12
filenums = np.arange(start, end)
for num in filenums:
	X = pd.read_csv(xdir+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
	smiles = []
	local_sim = []
	for i in range(X.shape[0]):
		x = decoder(remove_spaces(''.join(X.iloc[i])))
		seed_x = smiles_seeds[i]
		local_sim.append(mmpa.similarity(x, seed_x))
	avg_local_sim = np.mean(local_sim)
	sd_local_sim = np.std(local_sim)
	delta_means.append(avg_local_sim)
	delta_stds.append(sd_local_sim)

embed()