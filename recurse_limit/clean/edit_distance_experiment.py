"""
what is the average edit distance between smiles strings for logp score with logp opt and max pairwise sim?
we already have the data collected.
"""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../data_analysis')
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
import translate
from translate import translate
import props
from props import drd2_scorer



from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import editdistance


def return_random_subset(X, num):
	return np.random.choice(X.flatten(), size=num).reshape(-1,1)

def return_diverse_subset(X, num, max_diverse=True):
	fps = []
	for i in range(10000):
		fps.append(mmpa.mgn_fgpt(clean(X[i][0])))
	def distij(i,j,fps=fps):
		'''set "dist" to be similarity'''
		if max_diverse:
			return 1.0 - DataStructs.DiceSimilarity(fps[i],fps[j])
		else:
			return DataStructs.DiceSimilarity(fps[i],fps[j])

	nfps = len(fps)
	picker = MaxMinPicker()
	cmpds = []
	pickIndices = picker.LazyPick(distij,nfps,num,seed=i)
	picks = [X[x] for x in pickIndices]
	cmpds.extend(picks)
	div = mmpa.population_diversity(clean_array(cmpds))
	print(np.mean(div))
	return cmpds

def remove_spaces(x):
	return x.replace(" ", "")

def smiles_to_selfies(x, token_sep=True):
	"""smiles to selfies
	if token_sep=True -> return spaces between each token"""
	output = []
	for i in range(x.shape[0]):
		ax = encoder(x[i])
		if ax != -1:
			if token_sep:
				sx = re.findall(r"\[[^\]]*\]", ax)
				ax = ' '.join(sx)
			output.append(ax)
		else:
			output.append('NaN')
	return output
def selfies_to_smiles(x):
	return decoder(x)

def logp(x):
	return properties.penalized_logp(x)

def drd2(x):
	return drd2_scorer.get_score(x)

def prop_array(x, prop='logp04', prev_x=None, seed_sim=None):
	vals = []

	for i in range(len(x)):
		if prop == 'logp04':
			try:
				px = logp(x[i])
			except:
				px = None
		elif prop == 'drd2':
			try:
				px = drd2(x[i])
			except:
				px = None
		elif prop == 'maxsim':
			if prev_x:
				if score_dict[prop](x[i]) > score_dict[prop](prev_x):
					px = mmpa.similarity(x[i], prev_x)
				else:
					px = None
			else:
				print('error')
				sys.exit(0)
		elif prop == 'minmolwt':
			if score_dict[prop](x[i]) > score_dict[prop](prev_x):
				px = -rdkit.Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(sx))
			else:
				px = None
		elif prop == 'maxseedsim':
			if seed_sim:
				if score_dict[prop](x[i]) > score_dict[prop](seed_sim):
					px = mmpa.similarity(x[i], seed_sim)
				else:
					px = None
			else:
				px = None
		vals.append(px)
	return vals
def sim(a,b):
	return mmpa.similarity(a,b)

def clean(x):
	return selfies_to_smiles(remove_spaces(x))

def clean_array(x):
	sx = []
	for i in range(len(x)):
		if len(x[i])==1:
			sx.append(clean(x[i][0]))
		else:
			sx.append(clean(x[i]))
	return sx

def rank_by_prop(sx, prop):
	return 1

# def beam_search(model, src, n_best):
# 	_,preds=translate(model=model, src=src, tgt=None, n_best=n_best, beam_size=20)
# 	return np.array(preds).T

# def sd_topk(model, src, k, n_best):
# 	"""simplest method
# 	for each seed 1 to n, make k copies concatenate k*n then pass through translate
# 	then collapse back to n x k, rank by k return top 1 in 1 to k 
# 	"""
# 	total_preds=[]
# 	for i in range(n_best):
# 		print(i)
# 		_,preds=translate(model=model, src=src, tgt=None, n_best=1, beam_size=1, random_sampling_topk=k, seed=i)
# 		total_preds.append(preds)
# 	return np.array(total_preds).squeeze(-1)

# def rank(x, n_best, num_samples, prop, prev_x=None):
# 	top_cands = []
# 	for i in range(num_samples):
# 		props = prop_array(clean_array(x[:, i]), prop=prop, prev_x=prev_x[i])
# 		topind, topval = argmax_with_nones(props)
# 		top_cands.append([x[topind, i]])
# 	return top_cands

def argmax_with_nones(x):
	top_ind, top_val = 0, -1000
	for i in range(len(x)):
		if x[i]:
			if x[i] > top_val:
				top_val = x[i]
				top_ind = i
	return top_ind, top_val

def remove_empty_strings(x):
	return [sx for sx in x if len(sx[0])>0]

def remove_none(x):
	return [sx for sx in x if sx is not None]
def levenshtein(s1, s2):
    return float(editdistance.eval(s1, s2)) / max(len(s1), len(s2))

decoding_types = ['softmax_randtop2', 'softmax_randtop5']
score_funcs = ['logp', 'maxdeltasim']
input_dir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds'
for dc in decoding_types:
	for sc in score_funcs:
		dr = input_dir+'/'+dc+'/'+sc
		num_files=25
		sx = []
		for i in range(num_files):
			sx.append(clean_array(pd.read_csv(dr+'/'+str(i)+'.csv',header=None).values))
		sx = np.array(sx)
		edit_distances = np.zeros((sx.shape[0]-1, sx.shape[1]))
		for i in range(900):
			print(i)
			for j in range(num_files-1):
				edit_distances[j,i] = levenshtein(sx[j,i], sx[j+1, i])
		plt.errorbar(np.arange(1,num_files), np.mean(edit_distances,axis=1), yerr=np.std(edit_distances,axis=1)/np.sqrt(900), label=str(dc)+', '+str(sc))

plt.xlabel("Iteration")
plt.ylabel("Edit Distance")
plt.legend(loc="upper right")
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
path_to_figs='/tigress/fdamani/mol-edit-output/paper_figs'
plt.savefig(path_to_figs+'/edit_distance.png', dpi=1200)