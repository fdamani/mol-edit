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

def remove_spaces(x):
	return x.replace(" ", "")


# compute baselines with training data
target_prop = 'logp04'
#target_prop = 'qed'
def property_func(x):
	if target_prop=='logp04':
		try:
			return properties.penalized_logp(x)
		# catch sanitization errors
		except:
			return -100.0
	if target_prop=='qed':
		try:
			return properties.qed(x)
		# catch sanitization errors
		except:
			return 0.0
	print("ERROR. Please specify valid target property.")

X_train = pd.read_csv('/tigress/fdamani/mol-edit-data/data/' + target_prop + '/train_pairs.txt', sep=' ', header=None)
X_train = X_train[1]
train_logp = []
train_smiles = []
for i in range(X_train.shape[0]):
	sx = X_train.iloc[i]
	if sx not in train_smiles:
		train_logp.append(property_func(sx))
		train_smiles.append(sx)
train_logp = np.array(train_logp)
train_smiles = np.array(train_smiles)
train_sorted_inds = np.argsort(train_logp)[::-1]
top100_train_logp = train_logp[train_sorted_inds[0:100]]
top100_train_smiles = train_smiles[train_sorted_inds[0:100]]
top100_train_div = mmpa.population_diversity(top100_train_smiles)


init_seeds = ['jin_test', 'src_train_900maxdiv_seeds']
decoding_methods = ['beam', 'softmax_randtop2', 'softmax_randtop5']
rank_types = ['logp', 'maxdeltasim', 'mindeltasim', 'minmolwt']

# init_seeds = ['src_train_900maxdiv_seeds']
# decoding_methods = ['beam']
# rank_types = ['logp', 'maxdeltasim']


xdir = '/tigress/fdamani/mol-edit-output/onmt-' + target_prop + '/preds/recurse_limit'
num_files = 3
smiles = []
prop_val = []
dups = 0
inTraining = 0
for seed in init_seeds:
	print(seed)
	for dec in decoding_methods:
		print(dec)
		for rt in rank_types:
			print(rt)
			dr = xdir+'/'+seed+'/'+dec+'/'+rt
			for i in range(num_files):
				X = pd.read_csv(dr+'/'+str(i)+'.csv', header=None, skip_blank_lines=False)
				for j in range(X.shape[0]):
					sx = decoder(remove_spaces(''.join(X.iloc[j].values[0])))
					if sx not in smiles:
						prop_val.append(property_func(sx))
						smiles.append(sx)
						# of the non-duplicate strings, how many are in the training data?
						if sx in train_smiles:
							inTraining+=1
					else:
						dups+=1

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