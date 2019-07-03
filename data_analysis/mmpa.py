'''
Generate training pairs using Matched molecular pair analysis
Find all pairs of compounds with tanimoto similarity > x and large differences in potency values
'''
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import pandas as pd
import numpy as np

from IPython import embed

def similarity(a, b):
	'''https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py'''

	if a is None or b is None: 
		return 0.0
	amol = Chem.MolFromSmiles(a)
	bmol = Chem.MolFromSmiles(b)
	if amol is None or bmol is None:
		return 0.0

	fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
	fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
	return DataStructs.TanimotoSimilarity(fp1, fp2)

if __name__ == "__main__":
	dat = pd.read_csv('/data/potency_dataset_with_props.csv')
	dat = dat.sample(frac=1).reset_index(drop=True)
	dat = dat.head(1000)

	structure = dat['Structure']
	pot = -np.log10(dat['pot_uv'])
	similarity_scores = []
	inactive_structs = structure[pot == -3.]
	active_structs = structure[pot > -3.]
	for sx in inactive_structs:
		print(sx)
		for ax in active_structs:
			similarity_scores.append((sx, ax, similarity(sx, ax)))
	similarity_scores = pd.DataFrame(similarity_scores)
	embed()