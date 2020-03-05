'''
take as input
1) seq2seq with 6k samples
2) rseq2seq

histogram of top100 compounds for each
histogram of diversity of both.'''

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


# for r-seq2seq, compute top100 logpvals and avg pairwise div
r_smiles = torch.load('r_seq2seq_top100_smiles.pth')
r_logpvals = []
for i in range(len(r_smiles)):
	r_logpvals.append(properties.penalized_logp(r_smiles[i]))
r_div = mmpa.population_diversity(r_smiles)

# repeat above analysis for seq2seq
seed_to_logps = torch.load('logp_sd_seed_prop_vals.pth')
seed_to_smiles = torch.load('logp_sd_seed_prop_smiles.pth')
train_top100_smiles = torch.load('train_top100_smiles.pth')
rgraph_top100_smiles = torch.load('r_graph2graph_top100_smiles.pth')
graph_logp_vals = torch.load('/tigress/fdamani/mol-edit-data/SELFIES_seq2seq/6600_decode_logp_vals.pth')
graph_logp_vals = np.sort(graph_logp_vals)[::-1]
graph_logp_top100 = graph_logp_vals[0:100]
total_smiles = []
total_logp = []
count=0
input_smiles = []

input_logp = []
for k,v in seed_to_logps.items():
	#input_smiles.append(k)
	input_logp.append(properties.penalized_logp(k))
	for i in range(1, len(v)):
		total_smiles.append(seed_to_smiles[k][i-1])
		total_logp.append(seed_to_logps[k][i])
	count+=1
	print(count)
total_smiles = np.array(total_smiles)
total_logp = np.array(total_logp)
sorted_inds = np.argsort(total_logp)[::-1]
top100_smiles = total_smiles[sorted_inds[0:100]]
top100_logp = total_logp[sorted_inds[0:100]]
# top100_div = mmpa.population_diversity(top100_smiles)
# input_div = mmpa.population_diversity(train_top100_smiles)
# rgraph_div = mmpa.population_diversity(rgraph_top100_smiles)
rgraph_logp = []
for i in range(len(rgraph_top100_smiles)):
	rgraph_logp.append(properties.penalized_logp(rgraph_top100_smiles[i]))
embed()
color = [sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"], sns.xkcd_rgb["denim blue"]]
plt.cla()
#plt.hist(input_logp, color=color[1], alpha=.5, bins=10, label='Input')
plt.hist(graph_logp_top100, color=color[0], alpha=.5, bins=10, label='SD+Graph2Graph')
plt.hist(rgraph_logp, color=color[2], alpha=.5, bins=10, label='BBRT+Graph2Graph')
plt.legend(loc='upper left')
plt.savefig('bbrt_to_sd_graph2graph_logp.png', dpi=1200)




# plt.cla()
# color = [sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"], sns.xkcd_rgb["denim blue"]]
# plt.hist(input_div, color=color[0], alpha=.5, bins=40, label='Top 100 Training')
# plt.hist(rgraph_div, color=color[1], alpha=.5, bins=40, label='BBRT+Graph2Graph')
# plt.hist(r_div, color=color[2], alpha=.5, bins=40, label='BBRT+Seq2Seq')
# plt.legend(loc='upper left')
# plt.savefig('bbrt_to_sd_seq2seq_div.png', dpi=1200)