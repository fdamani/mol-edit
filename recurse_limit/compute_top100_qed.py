'''compute top 100 qed from vishnu's file'''

import pandas as pd
import selfies
import numpy as np
from selfies import encoder, decoder
from IPython import embed
import sys
sys.path.insert(0, '../data_analysis')
import mmpa
import properties
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import torch

x = pd.read_csv(sys.argv[1], header=None, sep=' ')
x=x[1]
qed_vals = []
smiles = []
for i in range(x.shape[0]):
	# if i % 100000 == 0 and i > 0:
	# 	print(i/x.shape[0], np.max(qed_vals))
	# 	torch.save(qed_vals, "/tigress/fdamani/mol-edit-data/SELFIES_seq2seq/6600_decode_logp_vals.pth")
	# 	torch.save(smiles, "/tigress/fdamani/mol-edit-data/SELFIES_seq2seq/6600_decode_logp_smiles.pth")
	sx = x.iloc[i]
	if sx is None:
		continue
	if sx in smiles:
		continue
	try:
		qed_vals.append(properties.qed(sx))		
		smiles.append(sx)
	except:
		continue
qed_vals = np.array(qed_vals)
smiles = np.array(smiles)
sorted_inds = np.argsort(qed_vals)[::-1]
top100_qed_vals = qed_vals[sorted_inds[0:100]]
top100_qed_smiles = smiles[sorted_inds[0:100]]
embed()