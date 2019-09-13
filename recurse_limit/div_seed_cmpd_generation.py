'''
start with the top 90 pct compounds
pick a small subset of similar molecules from a larger set

compute morganfingerprints, and DiceSimilarity to calculate distance between objects

then use the MaxMin algorithm:
Ashton, M. et al. “Identification of Diverse Database Subsets using Property-Based and Fragment-Based Molecular Descriptions.” Quantitative Structure-Activity Relationships 21:598-604 (2002).

compute morgan fingerprint
compute 2d pca
take 100 compounds that are clustered closely together
'''

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../data_analysis')
import properties
import mmpa
from IPython import embed

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


X = pd.read_csv(sys.argv[1], sep=' ', header=None, skip_blank_lines=False)
X = pd.concat([X[0], X[1]])
output_dir = sys.argv[2]

fps = []
for i in range(X.shape[0]):
	fps.append(mmpa.mgn_fgpt(X.iloc[i]))
	#fps.append(mmpa.mgn_fgpt(X.iloc[i].values[0]))
def distij(i,j,fps=fps):
	'''set "dist" to be similarity'''
	return 1.0 - DataStructs.DiceSimilarity(fps[i],fps[j])
nfps = len(fps)
picker = MaxMinPicker()
cmpds = []
for i in range(1,10):
	print(i)
	pickIndices = picker.LazyPick(distij,nfps,100,seed=i)
	picks = [X.iloc[x] for x in pickIndices]
	cmpds.extend(picks)
div = mmpa.population_diversity(cmpds)
print(np.mean(div))
pd.DataFrame(cmpds).to_csv(output_dir+'/src_train_1kdiv_seeds.csv',header=None,index=None)
embed()
# baseline
baseline = [X.iloc[i].values[0] for i in range(100)]
div_baseline = mmpa.population_diversity(baseline)
print(np.mean(div), np.std(div), np.mean(div_baseline), np.std(div_baseline))
embed()