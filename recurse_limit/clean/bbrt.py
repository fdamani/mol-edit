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

def remove_spaces(x):
	return x.replace(" ", "")

def selfies_to_smiles(x):
	return decoder(x)
def logp(x):
	return properties.penalized_logp(x)

def drd2(x):
	return drd2_scorer.get_score(x)

def prop_array(x, prop='logp04'):
	vals = []
	for i in range(len(x)):
		if prop=='logp04':
			px = logp(x[i])
		elif prop =='drd2':
			px = drd2(x[i])
		vals.append(px)
	return vals
def sim(a,b):
	return mmpa.similarity(a,b)

def clean(x):
	return selfies_to_smiles(remove_spaces(x))

def clean_array(x):
	sx = []
	for i in range(len(x)):
		sx.append(clean(x[i][0]))
	return sx

def rank_by_prop(sx, prop):
	return 1


def beam_search(model, src, n_best):
	_,preds=translate(model=model, src=src, tgt=None, n_best=n_best, beam_size=20)
	return preds

def sd_topk(model, src, k, n_best):
	"""simplest method
	for each seed 1 to n, make k copies concatenate k*n then pass through translate
	then collapse back to n x k, rank by k return top 1 in 1 to k 
	"""
	for i in range(n_best):
		embed()
		_,preds=translate(model=model, src=src, tgt=None, n_best=1, beam_size=1, random_sampling_topk=k, seed=i)


#prop='logp04' # qed
prop='drd2'
seed_type= 'src_train_900maxdiv_seeds'

logp_model = '/tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt'
drd2_model = '/tigress/fdamani/mol-edit-output/onmt-drd2/checkpoints/model-mlpattention/model_step_9000.pt'

seed_file = '/tigress/fdamani/mol-edit-data/data/logp04/test_sets/selfies/src_train_900maxdiv_seeds.csv'

if prop=='logp04':
	model = logp_model
if prop=='drd2':
	model = drd2_model

x = pd.read_csv(seed_file, header=None, skip_blank_lines=False).values

num_samples = 10
src = x[0:num_samples]
mean_prop = []
num_iters = 10
mean_prop.append(np.mean(prop_array(clean_array(src), prop=prop)))
embed()
for i in range(num_iters):
	preds = sd_topk(model, src, 2, 10)
	translate(model=model, src=src, tgt=None, n_best=10, beam_size=1, random_sampling_topk=50)
	# _,preds=translate(model=model, src=src, tgt=None, n_best=1, beam_size=20)

	embed()
	src = preds
	preds_smiles = clean_array(preds)

	pred_prop = prop_array(preds_smiles,prop=prop)
	mean_prop.append(np.mean(pred_prop))
	print(i+1, np.mean(pred_prop))
embed()
	