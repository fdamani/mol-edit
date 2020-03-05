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
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r_graph2graph = torch.load("r_graph2graph_top100_smiles.pth")
r_seq2seq = torch.load("r_seq2seq_top100_smiles.pth")
top100_train_smiles = torch.load("train_top100_smiles.pth")
# seq2seq_div = mmpa.population_diversity(r_seq2seq)
# graph2graph_div = mmpa.population_diversity(r_graph2graph)

color = [sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"], sns.xkcd_rgb["denim blue"]]

target_prop='logp04'
r_graph2graph_molwt, r_seq2seq_molwt, top100_train_smiles_molwt = [], [], []
for i in range(r_graph2graph.shape[0]):
	r_graph2graph_molwt.append(properties.molwt(r_graph2graph[i]))
	r_seq2seq_molwt.append(properties.molwt(r_seq2seq[i]))
	top100_train_smiles_molwt.append(properties.molwt(top100_train_smiles[i]))


plt.cla()
plt.hist(r_seq2seq_molwt, color=color[2], alpha=.6, bins=20, label='R-Seq2Seq')
plt.hist(r_graph2graph_molwt, color=color[1], alpha=.6,bins=20, label='R-Graph2Graph')
plt.hist(top100_train_smiles_molwt, color=color[0], alpha=.6, bins=20, label='Training')
plt.legend(loc='upper left')
plt.savefig('comparison_molwt.png', dpi=1200)
embed()


# X_train = pd.read_csv('/tigress/fdamani/mol-edit-data/data/' + target_prop + '/train_pairs.txt', sep=' ', header=None)
# X_train = X_train[1]
# train_smiles, train_logp = [], []
# for i in range(X_train.shape[0]):
# 	if X_train.iloc[i] not in train_smiles:
# 		train_smiles.append(X_train.iloc[i])
# 		train_logp.append(properties.penalized_logp(X_train.iloc[i]))

# train_logp = np.array(train_logp)
# sorted_inds = np.argsort(train_logp)[::-1]
# train_smiles = np.array(train_smiles)
# top100_train_smiles = train_smiles[sorted_inds][0:100]
# top100_train_smiles = torch.load("train_top100_smiles.pth")
# train_div = mmpa.population_diversity(top100_train_smiles)
embed()
#torch.save(top100_train_smiles, "train_top100_smiles.pth")
# top100_train_smiles = torch.load("train_top100_smiles.pth")
# train_div = mmpa.population_diversity(top100_train_smiles)
plt.cla()
plt.hist(seq2seq_div, color=color[2], alpha=.6, bins=20, label='R-Seq2Seq')
plt.hist(graph2graph_div, color=color[1], alpha=.6,bins=20, label='R-Graph2Graph')
plt.hist(train_div, color=color[0], alpha=.6, bins=20, label='Training')
plt.legend(loc='upper left')
plt.savefig('div_comparison_g2g_s2s.png', dpi=1200)