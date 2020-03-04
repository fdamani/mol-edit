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


def return_value_subset(X, X_smiles, num, tp):
	"""
	return 
		random 100 compounds from 
			bottom 25 percentile
			25-75 percentile
			75 to 100 percentile
	"""
	X = X[0:10000]
	vals = np.array(prop_array(X_smiles, prop='logp04'))
	if tp == 'high':
		X = X[vals>1]
		vals=vals[vals>1]
	elif tp == 'med':
		X = X[(vals>-1)*(vals<1)]
		vals = vals[(vals>-1)*(vals<1)]
	elif tp == 'low':
		X = X[vals<-1]
		vals=vals[vals<-1]
	else:
		print('error')
		sys.exit(0)
	return np.random.choice(X.flatten(), size=num).reshape(-1,1)

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


# def translate_and_rank(translate_type, model, src, k, n_best, num_samples, score_func):
# 	if translate_type=='sd':
# 		preds = sd_topk(model=model, src=src, k=k, n_best=n_best)
# 	elif translate_type=='beam':
# 		preds=beam_search(model=model, src=src, n_best=n_best)

# 	preds = rank(preds, n_best, num_samples, score_func, clean_array(src))
# 	return preds

# def bbrt(translate_type, num_iters, model, src, k, n_best, num_samples, score_func):
# 	total_preds = []
# 	for i in range(num_iters):
# 		preds = translate_and_rank(translate_type, model, src, k, n_best, num_samples, score_func)
# 		total_preds.append(preds)
# 		src = remove_empty_strings(preds)
# 		preds_smiles = clean_array(src)
# 		drd2_prop = prop_array(preds_smiles,prop='drd2')
# 		mean_drd2_prop.append((np.mean(drd2_prop), np.std(drd2_prop)))

# 	return mean_drd2_prop, total_preds

class BBRT:
	def __init__(self, translate_type, num_iters, model, src, k, n_best, num_samples, score_func, beam_size=20):
		self.translate_type = translate_type
		self.num_iters = num_iters
		self.model = model
		self.src = src
		self.k = k
		self.n_best = n_best
		self.num_samples = num_samples
		self.score_func = score_func
		self.total_preds = []
		self.mean_pop = []
		self.std_dev_pop = []
		self.max_pop = []
		self.beam_size = beam_size

		props = prop_array(clean_array(self.src), prop=self.score_func)
		self.mean_pop.append(np.mean(props))
		self.std_dev_pop.append(np.std(props))
		self.max_pop.append(np.max(props))

		self.max_so_far = []

	def run(self):
		intermed_src = self.src
		for i in range(self.num_iters):
			print('iter: ', i)
			preds = self.translate_and_rank(intermed_src)
			
			self.total_preds.append(preds)
			intermed_src = remove_empty_strings(preds)
			preds_smiles = clean_array(intermed_src)
			prop = prop_array(preds_smiles,prop=self.score_func)
			prop = remove_none(prop)
			self.mean_pop.append(np.mean(prop))
			self.std_dev_pop.append(np.std(prop))
			self.max_pop.append(np.max(prop))
		
		maxSoFar = self.max_pop[0]
		for i in range(len(self.max_pop)):
			if i ==0: continue
			if self.max_pop[i] > maxSoFar:
				maxSoFar = self.max_pop[i]
			self.max_so_far.append(maxSoFar)

	def translate_and_rank(self, intermed_src):
		if self.translate_type=='sd':
			preds = self.sd_topk(intermed_src)
		elif self.translate_type=='beam':
			preds = self.beam_search(src=intermed_src)

		preds = self.rank(preds, clean_array(intermed_src))
		return preds

	def rank(self, x, prev_x=None):
		top_cands = []
		for i in range(num_samples):
			props = prop_array(clean_array(x[:, i]), prop=self.score_func, prev_x=prev_x[i])
			topind, topval = argmax_with_nones(props)
			top_cands.append([x[topind, i]])
		return top_cands

	def beam_search(self, src):
		_,preds=translate(model=self.model, src=src, tgt=None, n_best=self.n_best, beam_size=self.beam_size)
		return np.array(preds).T

	def sd_topk(self, src):
		"""simplest method
		for each seed 1 to n, make k copies concatenate k*n then pass through translate
		then collapse back to n x k, rank by k return top 1 in 1 to k 
		"""
		total_preds=[]
		for i in range(self.n_best):
			_,preds=translate(model=self.model, src=src, tgt=None, n_best=1, beam_size=1, random_sampling_topk=self.k, seed=i)
			total_preds.append(preds)
		return np.array(total_preds).squeeze(-1)

output_dir = '/tigress/fdamani/mol-edit-output/onmt-logp04/output'

prop='logp04'
seed_type= 'complete_dataset_selfies.csv'

logp_model = '/tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt'
#drd2_model = '/tigress/fdamani/mol-edit-output/onmt-drd2/checkpoints/model-mlpattention/model_step_100000.pt'
drd2_model = '/tigress/fdamani/mol-edit-output/onmt-drd2_short/checkpoints/model-mlpattention/model_step_100000.pt'
#seed_file = '/tigress/fdamani/mol-edit-data/data/logp04/test_sets/selfies/src_train_900maxdiv_seeds.csv'
seed_file = '/tigress/fdamani/mol-edit-data/data/logp04/complete_dataset_selfies.csv'
#data
X = pd.read_csv(seed_file, header=None, skip_blank_lines=False).values
X_smiles = clean_array(X[0:10000])
#params
if prop=='logp04':
	model = logp_model
if prop=='drd2':
	model = drd2_model
score = ['drd2', 'logp04', 'maxsim', 'qed']
score_dict = {'drd2': drd2, 'log04': logp, 'drd2': drd2}
translate_types = ['sd', 'beam']
score_func = score[1]
translate_type = translate_types[0]
k=5
num_samples=100
# src = x_div
n_best=100
num_iters=10
lowvalue_so_far_across_iters, medvalue_so_far_across_iters, highvalue_so_far_across_iters = [], [], []
for i in range(10):
	print('iter: ', i)
	x_lowvalue = return_value_subset(X, X_smiles, num_samples, 'low')
	x_medvalue = return_value_subset(X, X_smiles, num_samples, 'med')
	x_highvalue = return_value_subset(X, X_smiles, num_samples, 'high')

	bbrt_lowvalue = BBRT(translate_type, num_iters, model, x_lowvalue, k, n_best, num_samples, score_func)
	bbrt_lowvalue.run()
	lowvalue_so_far_across_iters.append(bbrt_lowvalue.max_so_far)

	bbrt_medvalue = BBRT(translate_type, num_iters, model, x_medvalue, k, n_best, num_samples, score_func)
	bbrt_medvalue.run()
	medvalue_so_far_across_iters.append(bbrt_medvalue.max_so_far)


	bbrt_highvalue = BBRT(translate_type, num_iters, model, x_highvalue, k, n_best, num_samples, score_func)
	bbrt_highvalue.run()
	highvalue_so_far_across_iters.append(bbrt_highvalue.max_so_far)

# Average LogP of Iteration
plt.cla()
plt.errorbar(np.arange(1,num_iters+1), np.mean(lowvalue_so_far_across_iters,axis=0), yerr=np.std(lowvalue_so_far_across_iters,axis=0)/np.sqrt(100), label='low')
plt.errorbar(np.arange(1,num_iters+1), np.mean(medvalue_so_far_across_iters,axis=0), yerr=np.std(medvalue_so_far_across_iters,axis=0)/np.sqrt(100), label='medium')
plt.errorbar(np.arange(1,num_iters+1), np.mean(highvalue_so_far_across_iters,axis=0), yerr=np.std(highvalue_so_far_across_iters,axis=0)/np.sqrt(100), label='high')

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
path_to_figs='/tigress/fdamani/mol-edit-output/paper_figs'
plt.savefig(path_to_figs+'/seed_value_comparison.png', dpi=1200)