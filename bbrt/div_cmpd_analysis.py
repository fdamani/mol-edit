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
def remove_spaces(x):
	return x.replace(" ", "")

def selfies_to_smiles(x):
	return decoder(x)
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

"""
	want to trace the mean logp of a set of seed compounds
	pick 10 sets of min diverse
	pick 10 sets of max diverse
	pick 10 sets of median diverse

	for each plot the sd of the max found so far. 
	for each plot the sd of the mean found so far.
"""
X = pd.read_csv(sys.argv[1], sep=' ', header=None, skip_blank_lines=False).values.flatten()
# X = pd.concat([X[0], X[1]])
# output_dir = sys.argv[2]

fps = []
for i in range(X.shape[0]):
	fps.append(mmpa.mgn_fgpt(X[i]))
	#fps.append(mmpa.mgn_fgpt(X.iloc[i].values[0]))
def distij(i,j,fps=fps):
	'''set "dist" to be similarity'''
	# return 1.0 - DataStructs.DiceSimilarity(fps[i],fps[j])
	return DataStructs.DiceSimilarity(fps[i],fps[j])
nfps = len(fps)
picker = MaxMinPicker()
cmpds = []
for i in range(1):
	print(i)
	pickIndices = picker.LazyPick(distij,nfps,100,seed=i)
	picks = [X[x] for x in pickIndices]
	cmpds.extend(picks)
embed()
div = mmpa.population_diversity(cmpds)
print(np.mean(div))
pd.DataFrame(cmpds).to_csv(output_dir+'/src_train_1kdiv_seeds.csv',header=None,index=None)
embed()
# baseline
baseline = [X.iloc[i].values[0] for i in range(100)]
div_baseline = mmpa.population_diversity(baseline)
print(np.mean(div), np.std(div), np.mean(div_baseline), np.std(div_baseline))
embed()