'''
analysis for unconstrained optimization
given a directory, search through all generated candidates across
a) different init seeds 
b) decoding methods 
c) ranking methods


- compute top 100 unique logp compounds
- compute pairwise tanimoto diversity of this set.
- histogram of top 100 unique logp compounds
- compare to top 100 in training data.
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
import torch

def remove_spaces(x):
	return x.replace(" ", "")


# compute baselines with training data
target_prop = 'qed'
#target_prop = 'qed'
def property_func(x, prop=target_prop):
	if prop=='logp04':
		try:
			return properties.penalized_logp(x)
		# catch sanitization errors
		except:
			return -100.0
	if prop=='qed':
		try:
			return properties.qed(x)
		# catch sanitization errors
		except:
			return 0.0
	print("ERROR. Please specify valid target property.")


train_run = False
if train_run:
	X_train = pd.read_csv('/tigress/fdamani/mol-edit-data/data/' + target_prop + '/train_pairs.txt', sep=' ', header=None)
	X_train = pd.concat([X_train[0], X_train[1]], axis=0)
	train_logp = []
	train_qed = []
	train_smiles = []
	for i in range(X_train.shape[0]):
		sx = X_train.iloc[i]
		if sx not in train_smiles:
			train_logp.append(property_func(sx, prop='logp04'))
			train_qed.append(property_func(sx, prop='qed'))
			train_smiles.append(sx)
		if i % 1000 == 0:
			print(i)
	train_logp = np.array(train_logp)
	train_qed = np.array(train_qed)
	train_smiles = np.array(train_smiles)
	train_sorted_inds = np.argsort(train_logp)[::-1]
	top100_train_logp = train_logp[train_sorted_inds[0:100]]
	top100_train_logp_smiles = train_smiles[train_sorted_inds[0:100]]
	top100_train_logp_div = mmpa.population_diversity(top100_train_logp_smiles)

	train_sorted_inds = np.argsort(train_qed)[::-1]
	top100_train_qed = train_qed[train_sorted_inds[0:100]]
	top100_train_qed_smiles = train_smiles[train_sorted_inds[0:100]]
	top100_train_qed_div = mmpa.population_diversity(top100_train_qed_smiles)

	train_dat = (train_logp, train_smiles, train_sorted_inds, top100_train_logp, top100_train_logp_smiles, top100_train_logp_div, \
		top100_train_qed, top100_train_qed_smiles, top100_train_qed_div)
	torch.save(train_dat, "train_dat.pth")
else:
	train_dat = torch.load("train_dat.pth")
	train_logp, train_smiles, train_sorted_inds, top100_train_logp, top100_train_logp_smiles, top100_train_logp_div, \
		top100_train_qed, top100_train_qed_smiles, top100_train_qed_div = train_dat

init_seeds = ['jin_test']
decoding_methods = ['beam', 'softmax_randtop2', 'softmax_randtop5']
# if target_prop=='qed':
# 	rank_types =['qed']
# if target_prop=='logp04':
# 	rank_types=['logp']
rank_types = ['logp', 'maxdeltasim', 'mindeltasim', 'minmolwt'] #'qed'

# init_seeds = ['src_train_900maxdiv_seeds']
# decoding_methods = ['beam']
# rank_types = ['logp', 'maxdeltasim']


xdir = '/tigress/fdamani/mol-edit-output/onmt-' + target_prop + '/preds/recurse_limit'
smiles = []
prop_val = []
dups = 0
inTraining = 0
total_cands = 0
seed_prop_vals = {}
count=0
seed_prop_smiles = {}
#dr = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/jin_test/softmax_randtop5/norank2'
dr = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/jin_test/softmax_randtop5/norank2'
X_seed = pd.read_csv(dr+'/0.csv', header=None, skip_blank_lines=False)
X = pd.read_csv(dr+'/stochastic_decoding_qed.csv', header=None, skip_blank_lines=False)
num_samples = int(X.shape[0]/800)
count=0
for j in range(0, X.shape[0], num_samples):
	count+=1
	print(count)
	for k in range(0, num_samples):
		itr = j + k
		try:
			sx = decoder(remove_spaces(''.join(X.iloc[itr].values[0])))
		except:
			continue
		sx_seed = decoder(remove_spaces(''.join(X_seed.iloc[int(j/num_samples)].values[0])))		
		if sx_seed not in seed_prop_vals:
			seed_prop_vals[sx_seed] = []
			seed_prop_smiles[sx_seed] = []
			seed_prop_vals[sx_seed].append(property_func(sx_seed))
		# check similarity and improvement constraint
		if mmpa.similarity(sx, sx_seed) is None:
			continue
		if sx in seed_prop_smiles[sx_seed]:
			continue
		if mmpa.similarity(sx, sx_seed) > 0.4 and property_func(sx) > property_func(sx_seed):
			# qed closed subset constraint
			if target_prop=='qed':
				if property_func(sx) < 0.9:
					continue 
			# initialize seed in seed dictionary
			if sx_seed not in seed_prop_vals:
				count+=1
				seed_prop_vals[sx_seed] = []
				seed_prop_smiles[sx_seed] = []
			# add target compound property value to dict
			seed_prop_vals[sx_seed].append(property_func(sx))
			if property_func(sx) == 0.0 or property_func(sx)==-100.0:
				print('error')
				embed()
			seed_prop_smiles[sx_seed].append(sx)
			# check if string is in training data
			if sx in train_smiles:
				inTraining+=1
			total_cands+=1

success = 0
delta = []
max_val = 0
div = []
avg_num_cands = []
count=0
for k,v in seed_prop_vals.items():
	# constrain to seeds with at least 2 translations (first index is seed's property value)
	if len(v) >= 3:
		unique_list = list(set(seed_prop_smiles[k]))
		print(count,len(unique_list))
		if len(unique_list) > 1:
			if len(unique_list) > 50:
				unique_list = np.random.choice(unique_list, size=50, replace=False)
			div.append(np.mean(mmpa.population_diversity(unique_list)))
	#	div.append(np.mean(mmpa.population_diversity(seed_prop_smiles[k])))
	# success if at least one candidate in dict (first index is seed's prop value)
	if len(v) > 1:
		avg_num_cands.append(len(unique_list))
		if target_prop == 'qed':
			if np.max(v) >= 0.9:
				success+=1
				if np.max(v) > max_val:
					max_val = np.max(v)
		else:
			success+=1
		val = np.max(v) - v[0]
		delta.append(np.max(v) - v[0])
	count+=1
print('delta mean: ', np.mean(delta), 'delta sd: ', np.std(delta), 'success: ', success/800.0, 'div= ', np.mean(div), 'avg num cands: ', np.mean(avg_num_cands))
embed()
#  'percent novel: ', 1.0 - (inTraining/len(train_smiles)),
print('percent not unique: ', inTraining/len(smiles))
prop_val = np.array(prop_val)
smiles = np.array(smiles)
sorted_inds = np.argsort(prop_val)[::-1]
top100_logp, top100_smiles = [], []
for srted in sorted_inds:
	cmpd = smiles[srted]
	val = prop_val[srted]
	if cmpd not in train_smiles:
		top100_logp.append(val)
		top100_smiles.append(cmpd)
	if len(top100_smiles)==100: break
print(top100_logp[0:25])
#top100_logp = prop_val[sorted_inds[0:100]]
#print(top100_logp[0:25])
#top100_smiles = smiles[sorted_inds[0:100]]
num_uniquetop100 = 0
# check for num unique strings in top 100
for sm in smiles[sorted_inds[0:100]]:
	if sm not in train_smiles:
		num_uniquetop100+=1
print('num unique in top 100: ', num_uniquetop100)
top100_div = mmpa.population_diversity(top100_smiles)
embed()
#######################
color = [sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"], sns.xkcd_rgb["denim blue"]]
path_to_figs = xdir+'/figs/'
if not os.path.exists(path_to_figs):
	os.mkdir(path_to_figs)
# pop div plot
plt.cla()
plt.hist(top100_div, color=color[2], bins=40, label="Candidates")
plt.hist(top100_train_div, color=color[0], bins=40, label="Training")
plt.legend(loc='upper left')
plt.savefig(xdir+'/figs/top100'+target_prop+'_pop_div.png', dpi=1200)
########################
# top 100 logp plot
# compute logp of top 100 from ZINC
plt.cla()
plt.hist(top100_logp, color=color[2], bins=40, label="Candidates")
plt.hist(top100_train_logp, color=color[0], bins=40, label="Top 100 Training")
#plt.hist(train_logp, color=color[1], label="Full Training Dist.")
plt.legend(loc='upper left')
plt.savefig(xdir+'/figs/top100' + target_prop +'_hist.png', dpi=1200)
