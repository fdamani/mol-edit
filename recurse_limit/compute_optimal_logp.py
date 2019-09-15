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

def remove_spaces(x):
	return x.replace(" ", "")

#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/softmax_temp1/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/softmax_randtop10/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5/toplogp'
xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_valid/beam20/toplogp'
#xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/greedy'
#types=['softmax_randtop5']
#labels=['Rnd Top 5']
#types = ['beam20', 'softmax_randtop2', 'softmax_randtop3', 'softmax_randtop4', 'softmax_randtop5']
#types = ['beam']#, 'softmax_randtop2', 'softmax_randtop5']
#labels = ['Beam']#, 'Rnd Top 2', 'Rnd Top 5']
types = ['beam']
labels = ['Rnd Top 2']

rank_types=['logp', 'maxdeltasim', 'maxseedsim', 'mindeltasim', 'minmolwt', 'qed']
#rank_types=['logp', 'maxdeltasim']
rt = rank_types[5]

type_logp_means = {}
type_logp_stds = {}
type_prop_valid = {}
type_num_samples = {}
type_optimal_logp = {}
type_optimal_logp_iter = {}
type_optimal_logp_iter_std = {}
type_max_logp_iter = {}
for tp in types:
	dr = xdir+'/'+tp+'/'+rt
	logp_means = []
	logp_stds = []
	prop_valid = []
	num_samples = []
	start = 0
	end = 22
	filenums = np.arange(start, end)
	seeds = pd.read_csv(dr+'/'+str(0)+'.csv', header=None, skip_blank_lines=False)
	optimal_logp = []
	optimal_logp_iter = []
	optimal_logp_iter_std = []
	max_logp = 0
	max_logp_iter = []
	smiles_dict = {}
	for num in filenums:
		X = pd.read_csv(dr+'/'+str(num)+'.csv', header=None, skip_blank_lines=False)
		smiles = []
		local_logp=[]
		for i in range(X.shape[0]):
			sx = decoder(remove_spaces(''.join(X.iloc[i])))
			x_seed = decoder(remove_spaces(''.join(seeds.iloc[i])))
			try:
				val = properties.penalized_logp(sx)
			except:
				val = -100
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
	
	embed()
	# save top 10 compounds
	inds = np.argsort(optimal_logp)[-10:]
	for ind in inds:
		logpvals = []
		mols = []
		for i in range(start, end):
			sm = smiles_dict[i][ind]
			logpvals.append(properties.penalized_logp(sm))
			mx = Chem.MolFromSmiles(sm)
			mols.append(mx)
		maxind = np.argmax(logpvals)
		relevant_mols = mols[0:maxind+1]
		logpvals_strs = []
		for i in range(0, maxind+1):
			logpvals_strs.append("LogP="+"{:.2f}".format(logpvals[i]))
		img=Draw.MolsToGridImage(relevant_mols, molsPerRow=4, subImgSize=(400,400), legends=logpvals_strs)
		path_to_figs = xdir+'/'+tp+'/'+rt+'/figs'
		if not os.path.exists(path_to_figs):
			os.mkdir(path_to_figs)
		if not os.path.exists(path_to_figs+'/cmpds'):
			os.mkdir(path_to_figs+'/cmpds')
		img.save(path_to_figs+'/cmpds/'+str(ind)+'.png')
	
	# save to dict
	type_max_logp_iter[tp] = np.array(max_logp_iter)
	type_optimal_logp_iter[tp] = np.array(optimal_logp_iter)
	type_optimal_logp_iter_std[tp] = np.array(optimal_logp_iter_std)
	type_logp_means[tp] = np.array(logp_means)
	type_logp_stds[tp] = np.array(logp_stds)
	type_prop_valid[tp] = np.array(prop_valid)

###################################################################
path_to_figs=xdir+'/figs'
if not os.path.exists(path_to_figs):
	os.mkdir(path_to_figs)
# Max LogP
plt.cla()
for i in range(len(types)):
	tp = types[i]
	lab = labels[i]
	plt.plot(np.arange(start,end), type_max_logp_iter[tp], label=lab)
plt.xlabel("Iteration")
plt.ylabel("Max LogP")
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
plt.savefig(path_to_figs+'/max_logp_iter_'+rt+'.png')

###################################################################
# Average LogP of Iteration
plt.cla()
for i in range(len(types)):
	tp = types[i]
	lab = labels[i]
	plt.errorbar(np.arange(start,end), type_logp_means[tp], yerr=type_logp_stds[tp], label=lab)
plt.xlabel("Iteration")
plt.ylabel("Mean LogP")
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
plt.savefig(path_to_figs+'/avg_logp_'+rt+'.png')

#####################################################################
# Average LogP of best compound seen so far
plt.cla()
for i in range(len(types)):
	tp = types[i]
	lab = labels[i]
	plt.errorbar(np.arange(start,end), type_optimal_logp_iter[tp], yerr=type_optimal_logp_iter_std[tp], label=lab)
plt.xlabel("Iteration")
plt.ylabel("Mean Optimal LogP")
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
plt.savefig(path_to_figs+'/avg_optimal_logp_'+rt+'.png')