'''given csv file in spaced out selfies, compute pop diversity'''

import pandas as pd
import selfies
import numpy as np
from selfies import encoder, decoder
from IPython import embed
import sys
sys.path.insert(0, '../data_analysis')
import mmpa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def remove_spaces(x):
	return x.replace(" ", "")

#xdir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/src_train_900maxdiv_seeds/'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/jin_test/'
xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/jin_test/'

#types = ['beam20', 'softmax_randtop2', 'softmax_randtop3', 'softmax_randtop4', 'softmax_randtop5']
#labels = ['Beam', 'Rnd Top 2', 'Rnd Top 3', 'Rnd Top 4', 'Rnd Top 5']
#types = ['beam', 'softmax_randtop2', 'softmax_randtop5']
types = ['beam', 'softmax_randtop2', 'softmax_randtop5']
#labels = ['Rnd Top 5']
labels = ['Beam', 'Rnd Top 2', 'Rnd Top 5']
rank_types=['qed', 'logp', 'maxdeltasim', 'maxseedsim', 'mindeltasim', 'minmolwt']
rt = rank_types[1]
type_div_means, type_div_stds = {}, {}
for lk in range(2):
	if lk == 0:
		xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/jin_test/'
		rt = 'logp'
		type_div_means[lk] = {}
		type_div_stds[lk] = {}
	else:
		xdir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/jin_test/'
		rt = 'qed'
		type_div_means[lk] = {}
		type_div_stds[lk] = {}
	for tp in types:
		dr = xdir+tp+'/'+rt
		div_means = []
		div_stds = []
		num_samples = []
		start = 0
		end = 25
		cutoff = 10
		num_outputs_per_seed = 20 if tp=='beam' else 100
		filenums = np.arange(start, end)
		for num in filenums:
			if num == 0:
				X_seed = pd.read_csv(dr+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
				X_seed = X_seed.iloc[0:cutoff]
			X = pd.read_csv(dr+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
			X = X.iloc[0:cutoff]
			if num != 0:
				X = pd.read_csv(dr+'/cands/'+str(num)+'.csv', header=None, skip_blank_lines=False)
				X = X.iloc[0:int(num_outputs_per_seed*cutoff)]
			num_samples = int(X.shape[0]/X_seed.shape[0])
			smiles = []
			for j in range(0, X.shape[0], num_samples):
				for k in range(0, num_samples):
					itr = j + k
					try:
						sx = decoder(remove_spaces(''.join(X.iloc[itr].values[0])))
						smiles.append(sx)
					except:
						continue
			unique_smiles = list(set(smiles))
			if num == 0:
				rand_smiles = unique_smiles
			else:
				rand_smiles = np.random.choice(unique_smiles, size=num_outputs_per_seed, replace=False)
			
				# compute approx avg pairwise similarity by picking 100 random compounds
			div = mmpa.population_diversity(rand_smiles)
			div_means.append(np.mean(div))
			div_stds.append(np.std(div))
			print(num, np.mean(div), np.std(div))
		try:
			type_div_means[lk][tp] = np.array(div_means)
		except:
			embed()
		type_div_stds[lk][tp] = np.array(div_stds)
color = {'beam': sns.xkcd_rgb["pale red"], 'softmax_randtop2': sns.xkcd_rgb["medium green"], 'softmax_randtop5': sns.xkcd_rgb["denim blue"]}
# average pop diversity
for lk in range(2):
	plt.cla()
	if lk == 0:
		rt = 'logP'
	else:
		rt = 'QED'
	for i in range(len(types)):
		tp = types[i]
		lab = labels[i]
		#plt.ylim(0.2, 1.0)
		plt.errorbar(np.arange(start,end), type_div_means[lk][tp], color=color[tp], yerr=type_div_stds[lk][tp], label=lab)
	plt.xlabel("Iteration")
	plt.ylabel("Mean Pairwise Diversity")
	plt.ylim(0.4, 1.0)
	plt.legend(loc="lower right")
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
	if lk == 0:
		plt.savefig('/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/jin_test/figs/entropyfig_mean_pop_div_'+rt+'.png', dpi=1200)
	else:
		plt.savefig('/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/jin_test/figs/entropyfig_mean_pop_div_'+rt+'.png', dpi=1200)