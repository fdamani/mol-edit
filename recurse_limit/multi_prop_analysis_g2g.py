'''given csv file in spaced out selfies, compute population level qed

1. unconstrained mean logp (avg logp of max(cands/seed) across all compounds)
	- only condition is value must be greater than seed
2. constrained mean logp (tanimoto similarity > 0.4)
	- value must be greater than seed
	- tanimoto similarity must be greater than 0.4
	- plot number of compounds that meet threshold as a function of time
3. optimal logp (with constraint)
	- best logp compound for each seed up to that iteration (e.g. as recursive iterations increase, does the best compound improve?).

	- best logp compound in entire set! 

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import os
import seaborn as sns

def remove_spaces(x):
	return x.replace(" ", "")

#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/softmax_temp1/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/softmax_randtop10/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5/toplogp'

#xdir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/src_train_900maxdiv_seeds'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/jin_test'
xdir = '/tigress/fdamani/mol-edit-data/jin_test_qed/qed'


#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_valid/beam20/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/greedy'
#types=['softmax_randtop5']
#labels=['Rnd Top 5']
#types = ['beam20', 'softmax_randtop2', 'softmax_randtop3', 'softmax_randtop4', 'softmax_randtop5']
#types = ['beam']#, 'softmax_randtop2', 'softmax_randtop5']
#labels = ['Beam']#, 'Rnd Top 2', 'Rnd Top 5']
types = ['beam']
labels = {'beam': 'Greedy'}

rank_types=['logp', 'qed']

target='qed'
#target='LogP'
#rank_types=['logp', 'maxdeltasim']
# compute baseline improvement in logp from qed data

# training_file="/tigress/fdamani/mol-edit-data/data/qed/train_pairs.txt"
# train_x = pd.read_csv(training_file, sep=' ', header=None)
# logp_a, logp_b = [], []
# qed_a, qed_b = [], []

# for i in range(train_x.shape[0]):
# 	a, b = train_x.iloc[i][0], train_x.iloc[i][1]
# 	logp_a.append(properties.penalized_logp(a))
# 	logp_b.append(properties.penalized_logp(b))

# 	qed_a.append(properties.qed(a))
# 	qed_b.append(properties.qed(b))
# 	#delta = properties.penalized_logp(b) - properties.penalized_logp(a)
# 	#logp_delta.append(delta)
# 	print(i)
# 	if i == 1000:
# 		break
# import scipy
# from scipy import stats
# print
seeds = pd.read_csv(xdir+'/'+target+'/seeds_0'+str(0)+'.csv', header=None, skip_blank_lines=False)

type_logp_means = {}
type_logp_stds = {}
type_max_logp_iter = {}
for tp in types:
	type_logp_means[tp] = {}
	type_logp_stds[tp] = {}
	type_max_logp_iter[tp] = {}
for tp in types:
	for rt in rank_types:
		print(tp, rt)
		dr = xdir+'/'+rt
		logp_means = []
		logp_stds = []
		prop_valid = []
		num_samples = []
		start = 0
		end = 5 # instead of 7
		filenums = np.arange(start, end)
		optimal_logp = []
		optimal_logp_iter = []
		optimal_logp_iter_std = []
		max_logp = 0
		max_logp_iter = []
		smiles_dict = {}
		for num in filenums:
			X = pd.read_csv(dr+'/best_cmpds_0'+str(num)+'.csv', header=None, skip_blank_lines=False)
			smiles = []
			local_logp=[]
			for i in range(X.shape[0]):
				sx = X.iloc[i].values[0]
				x_seed = seeds.iloc[i].values[0]
				#sx = decoder(remove_spaces(''.join(X.iloc[i])))
				#x_seed = decoder(remove_spaces(''.join(seeds.iloc[i])))
				try:
					if target=='qed':
						val = properties.qed(sx)
					else:
						val = properties.penalized_logp(sx)
				except:
					val = -100

				if target=='qed':
					x_seed_val = properties.qed(x_seed)
				else:
					x_seed_val = properties.penalized_logp(x_seed)
				if num == 0:
					local_logp.append(val)
					optimal_logp.append(val)
					if val > max_logp:
						max_logp = val
				else:
					if val > max_logp:
						max_logp = val
					#sim = mmpa.similarity(sx, x_seed)
					# if compound has improved property value
					if val > x_seed_val:
						local_logp.append(val)
						# if curr val is better than curr optimal val
						if val > optimal_logp[i]:
							optimal_logp[i] = val
				smiles.append(sx)
			max_logp_iter.append(max_logp)
			optimal_logp_iter.append(np.mean(optimal_logp))
			optimal_logp_iter_std.append(np.std(optimal_logp))
			logp_means.append(np.mean(local_logp))
			logp_stds.append(np.std(local_logp))
			prop_valid.append(len(local_logp)/float(X.shape[0]))
			smiles_dict[num] = smiles
			print(num, max_logp, np.mean(local_logp), np.std(local_logp))
	

	# save to dict
		type_max_logp_iter[tp][rt] = np.max(np.array(max_logp_iter))
		type_logp_means[tp][rt] = np.array(logp_means)
		type_logp_stds[tp][rt] = np.array(logp_stds)
embed()
max_logp_dict = {'logp': 0, 'qed': 0}
for tp in types:
	for rt in rank_types:
		if type_max_logp_iter[tp][rt] > max_logp_dict[rt]:
			max_logp_dict[rt] = type_max_logp_iter[tp][rt]
rt_labels = {'logp': 'LogP', 'qed': 'QED'}

path_to_figs=xdir+'/figs'
if not os.path.exists(path_to_figs):
	os.mkdir(path_to_figs)
dotted_style = {'logp': 'dotted', 'qed': 'solid'}
#color = {'beam': 'r', 'softmax_randtop2': 'b', 'softmax_randtop5': 'g'}
color = {'beam': sns.xkcd_rgb["pale red"], 'softmax_randtop2': sns.xkcd_rgb["medium green"], 'softmax_randtop5': sns.xkcd_rgb["denim blue"]}

for tp in types:
	for rt in rank_types:
		plt.errorbar(x=np.arange(start, end), y=type_logp_means[tp][rt], yerr=type_logp_stds[tp][rt], color=color[tp], linestyle=dotted_style[rt], label=labels[tp]+', '+rt_labels[rt])
for rt in rank_types:
	plt.axhline(y=max_logp_dict[rt], color='black', linestyle=dotted_style[rt], label='Max, ' + rt_labels[rt])

plt.legend(loc='lower right')
plt.xlabel("Iteration")
if target=='qed':
	plt.ylabel("QED")
else:
	plt.ylabel("LogP")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    labelbottom=True)         # ticks along the top edge are off
if target=='qed':
	plt.savefig(path_to_figs+'/multiprop_optqed_auxlogp_qedresults.png', format='png', dpi=600)
else:
	plt.savefig(path_to_figs+'/multiprop_optqed_auxlogp_logpresults.png', format='png', dpi=600)
