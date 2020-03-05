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
import utils_rebut as ut


def fp_array(x):
	dat = []
	for i in range(len(x)):
		a,b=ut.morgan_fpt(x[i])
		dat.append(list(b))
	return dat


input_dir = '/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5/qed/'
path_to_figs='/tigress/fdamani/mol-edit-output/paper_figs'
num_files = 5
top_qed_init, top_qed = [], []
total_dats = []
for i in range(0, num_files):
	if i == 0: continue
	x = pd.read_csv(input_dir+str(i)+'.csv',header=None).values
	x_smi = ut.clean_array(x)
	x_qed = np.array(ut.prop_array(x_smi,prop='qed'))
	descript = np.array(fp_array(x_smi))
	if i > 1: total_dats.extend(descript)

	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(descript)
	if i == 1:
		plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1],color='b')
		plt.savefig(path_to_figs+'/qed_tsne.png')
	embed()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(np.array(total_dats))
plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1],color='r')
plt.savefig(path_to_figs+'/qed_tsne.png')
embed()
	# sx = torch.load('/tigress/fdamani/mol-edit-data/SELFIES_seq2seq/jin_test_qed_6600_decode_qed_smiles.pth')