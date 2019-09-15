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

def remove_spaces(x):
	return x.replace(" ", "")
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/greedy'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop3/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/beam20/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/beam20/toplogp'
xdir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/src_train_900maxdiv_seeds/'
#types = ['beam20', 'softmax_randtop2', 'softmax_randtop3', 'softmax_randtop4', 'softmax_randtop5']
#labels = ['Beam', 'Rnd Top 2', 'Rnd Top 3', 'Rnd Top 4', 'Rnd Top 5']
types = ['beam', 'softmax_randtop2', 'softmax_randtop5']
labels = ['Beam', 'Rnd Top 2', 'Rnd Top 5']
rank_types=['logp', 'maxdeltasim', 'maxseedsim', 'mindeltasim', 'minmolwt']
rt = rank_types[0]
type_div_means, type_div_stds, type_num_samples = {}, {}, {}

for tp in types:
	dr = xdir+tp+'/'+rt
	div_means = []
	div_stds = []
	num_samples = []
	start = 0
	end = 21
	cutoff = 100
	filenums = np.arange(start, end)
	for num in filenums:
		X = pd.read_csv(dr+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
		smiles = []
		for i in range(X.shape[0]):
			smiles.append(decoder(remove_spaces(''.join(X.iloc[i]))))
		# approximate diversity with first 100 compounds
		div = mmpa.population_diversity(smiles[0:cutoff])
		div = np.array(div)
		div_means.append(np.mean(div))
		div_stds.append(np.std(div))
		num_samples.append(div.shape[0])
		print(num, np.mean(div))

	type_div_means[tp] = np.array(div_means)
	type_div_stds[tp] = np.array(div_stds)
	type_num_samples[tp] = np.array(num_samples)

# average pop diversity
plt.cla()
for i in range(len(types)):
	tp = types[i]
	lab = labels[i]
	plt.errorbar(np.arange(start,end), type_div_means[tp], yerr=type_div_stds[tp], label=lab)
plt.xlabel("Iteration")
plt.ylabel("Mean Pairwise Diversity")
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
plt.savefig(xdir+'figs/mean_pop_div_'+rt+'.png')