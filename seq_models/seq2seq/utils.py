import torch
import rdkit
import rdkit.Chem.QED as QED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

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