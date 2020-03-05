'''given csv file in spaced out selfies, compute population level qed

1. unconstrained mean logp (avg logp of max(cands/seed) across all compounds)
	- only condition is value must be greater than seed
2. constrained mean logp (tanimoto similarity > 0.4)
	- plot number of compounds that meet threshold as a function of time
3. optimal logp (with constraint)
	- best logp compound for each seed compound up to that iteration (e.g. as recursive iterations increase, do compounds improve).

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
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import torch

def remove_spaces(x):
	return x.replace(" ", "")

#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/greedy'
#xdir = '/tigress/fdamani/mol-edit-data/results_pt_1/recursive_g2g/src_train_900maxdiv_seeds/logp04/logp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5/logp/cands'
xdir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5/qed'

logp_means = []
logp_stds = []
prop_valid = []
num_samples = []
start = 1 # 0 
end = 25 # 5
selfies=True
filenums = np.arange(start, end)
#seeds = pd.read_csv(xdir+'/seeds_0'+str(0)+'.csv', header=None, skip_blank_lines=False)
# seeds = pd.read_csv(xdirseed+'/'+str(0)+'.csv', header=None, skip_blank_lines=False)

smiles, local_logp = [], []
all_vals = []
all_smiles = []
# for i in range(seeds.shape[0]):
# 	sx = seeds.iloc[i].values[0]
# 	all_smiles.append(sx)
# 	val = properties.penalized_logp(sx)
# 	all_vals.append(val)
# 	local_logp.append(val)
# logp_means.append(np.mean(local_logp))
# logp_stds.append(np.std(local_logp))
invalid_samples = 0
for num in filenums:
	#X = pd.read_csv(xdir+'/best_cmpds_0'+str(num)+'.csv', header=None, skip_blank_lines=False)
	X = pd.read_csv(xdir+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
	smiles = []
	local_logp=[]
	for i in range(X.shape[0]):
		if selfies:
			try:
				sx = decoder(remove_spaces(''.join(X.iloc[i])))
			except:
				invalid_samples+=1
				continue
			# x_seed = decoder(remove_spaces(''.join(seeds.iloc[i])))
		else:
			sx = X.iloc[i].values[0]
			# x_seed = seeds.iloc[i].values[0]
		if sx in all_smiles:
			continue
		try:
			#val = properties.penalized_logp(sx)
			val = properties.qed(sx)
		except:
			invalid_samples+=1
			continue
		all_vals.append(val)
		all_smiles.append(sx)
		# x_seed_val = properties.penalized_logp(x_seed)
		# if val > x_seed_val:
		local_logp.append(val)
		smiles.append(sx)
	logp_means.append(np.mean(local_logp))
	logp_stds.append(np.std(local_logp))
	prop_valid.append(len(local_logp)/float(X.shape[0]))
	print(np.max(all_vals))
all_vals = np.array(all_vals)
all_smiles = np.array(all_smiles)
sorted_inds = np.argsort(all_vals)[::-1]
all_vals = all_vals[sorted_inds]
all_smiles = all_smiles[sorted_inds]
top20_smiles = all_smiles[0:30]
top20_mols = [Chem.MolFromSmiles(sx) for sx in top20_smiles]
top20_vals = all_vals[0:30]
logpvals_strs = []
embed()
top100_smiles = all_smiles[0:100]
torch.save(top100_smiles, "r_seq2seq_top100qed_smiles.pth")
embed()
for i in range(30):
	logpvals_strs.append("QED="+"{:.3f}".format(top20_vals[i]))
img=Draw.MolsToGridImage(top20_mols, molsPerRow=5, subImgSize=(400,400), legends=logpvals_strs)
#path_to_fig = '/tigress/fdamani/mol-edit-data/results_pt_1/recursive_g2g/src_train_900maxdiv_seeds/logp04/logp'+'/figs'
#path_to_fig = '/tigress/fdamani/mol-edit-data/results_pt_1/recursive_g2g/src_train_900maxdiv_seeds/logp04/logp'+'/figs'
path_to_fig = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5/qed' +'/figs'
if not os.path.exists(path_to_fig):
	os.mkdir(path_to_fig)
img.save(path_to_fig+'/top30qed.png')
embed()
