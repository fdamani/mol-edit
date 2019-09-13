'''given csv file in spaced out selfies, compute delta_div
delta_div is pairwise diversity between each time point pairs'''

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
import os

def remove_spaces(x):
	return x.replace(" ", "")
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/greedy'
xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5'

#rank_types = ["maxdeltasim", "logp", "mindeltasim", "minmolwt", "maxseedsim"]
#labels = ['Max Delta Sim', "LogP", "Min Delta Sim", "Min Mol Wt", "Max Seed Sim"]
rank_types=['logp', 'maxdeltasim', 'maxseedsim', 'mindeltasim', 'minmolwt']
labels=["LogP", "Max Delta Sim", "Max Seed Sim", "Min Delta Sim", "Min Mol Wt"]
#rank_types=["maxdeltasim", "logp"]
#labels = ['Max Iter Sim', "LogP"]
global_delta_means = {}
global_delta_stds = {}
for rt in rank_types:
	rdir = xdir+'/'+rt

	delta_means = []
	delta_stds = []
	delta_num_samples = []
	start = 0
	end = 21
	filenums = np.arange(start, end)
	for num in filenums:
		X = pd.read_csv(rdir+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
		X_next = pd.read_csv(rdir+'/'+str(num+1)+'.csv', header=None, skip_blank_lines=False)
		smiles = []
		nextX_smiles = []
		local_sim = []
		for i in range(X.shape[0]):
			x = decoder(remove_spaces(''.join(X.iloc[i])))
			x_next = decoder(remove_spaces(''.join(X_next.iloc[i])))
			sim = mmpa.similarity(x, x_next)
			if sim != 0.0:
				local_sim.append(sim)
		avg_sim = np.mean(local_sim)
		sd_sim = np.std(local_sim)
		delta_means.append(avg_sim)
		delta_stds.append(sd_sim)

		delta_num_samples.append(len(local_sim))
		print(num)
	global_delta_means[rt] = np.array(delta_means)
	global_delta_stds[rt] = np.array(delta_stds)
	print(delta_means)
plt.cla()
for i in range(len(rank_types)):
	rt = rank_types[i]
	plt.errorbar(np.arange(start, end), global_delta_means[rt], yerr=global_delta_stds[rt], label=labels[i])
plt.xlabel("Iteration")
plt.ylabel("Avg Similarity Between Pairwise Iters")
plt.legend(loc = 'lower right')
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
path_to_figs=xdir+'/figs'
if not os.path.exists(path_to_figs):
	os.mkdir(path_to_figs)
plt.savefig(path_to_figs+'/delta_sim.png')

