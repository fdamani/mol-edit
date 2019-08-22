'''read in test-src, test-tgt, and test-tg
python evaluate.py /tigress/fdamani/mol-edit-data/data/qed/src_test.csv /tigress/fdamani/mol-edit-data/data/qed/tgt_test.csv /tigress/fdamani/mol-edit-output/onmt-qed/preds/output-model-rnnenc-rnndec-1layer-500wordembed-500embed-adamoptim_step_5000.csv


python evaluate.py /tigress/fdamani/mol-edit-data/data/qed/src_test.csv /tigress/fdamani/mol-edit-data/data/qed/tgt_test.csv /tigress/fdamani/mol-edit-output/onmt-qed/preds/output-5beam-model-rnnenc-rnndec-1layer-500wordembed-500embed-dotattention-adamoptim_step_100000.csv



'''
import sys
sys.path.insert(0, '../seq2seq')
sys.path.insert(0, '../../data_analysis')
import torch
import pandas as pd
from IPython import embed
import re
import selfies
import utils
import properties
import mmpa
from selfies import encoder, decoder
from utils import similarity
import rdkit
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

def remove_spaces(x):
	return x.replace(" ", "")

def molwt(x):
	'''x is a smiles string'''
	return rdkit.Chem.Descriptors.ExactMolWt(rdkit.Chem.MolFromSmiles(x))

def penalized_logp(x):
	'''x is a smiles string'''
	return properties.penalized_logp(x)
	#return rdkit.Chem.Crippen.MolLogP(rdkit.Chem.MolFromSmiles(x))

src = pd.read_csv(sys.argv[1], header=None)
tgt = pd.read_csv(sys.argv[2], header=None)
preds = pd.read_csv(sys.argv[3], header=None)

num_evaluate = src.shape[0]
valid = []
input_strs, tgt_strs, preds_strs = [], [], []
preds_similarity, tgt_similarity = [], []
src_list_qed, tgt_list_qed, preds_list_qed = [], [], []

# distributional stats
preds_string_length, tgt_string_length = [], []
preds_molwt, tgt_molwt = [], []
src_logp, preds_logp, tgt_logp = [], [], []
preds_diversity, tgt_diversity = [], []
for i in range(num_evaluate):
	print(i)
	src_i = decoder(remove_spaces(''.join(src.iloc[i])))
	tgt_i = decoder(remove_spaces(''.join(tgt.iloc[i])))
	preds_i = decoder(remove_spaces(''.join(preds.iloc[i])))

	if preds_i == -1:
		valid.append(0)
		continue

	input_qed = properties.qed(src_i)
	tgt_qed = properties.qed(tgt_i)
	pred_qed = properties.qed(preds_i)

	if pred_qed == 0.0:
		valid.append(0)
		continue
	src_list_qed.append(input_qed)
	tgt_list_qed.append(tgt_qed)
	preds_list_qed.append(pred_qed)
	preds_similarity.append(mmpa.similarity(src_i, preds_i))
	tgt_similarity.append(mmpa.similarity(src_i, tgt_i))
	input_strs.append(src_i), tgt_strs.append(tgt_i), preds_strs.append(preds_i), valid.append(1)


	preds_string_length.append(len(preds_i)), tgt_string_length.append(len(tgt_i))
	preds_molwt.append(molwt(preds_i)), tgt_molwt.append(molwt(tgt_i))
	preds_logp.append(penalized_logp(preds_i)), src_logp.append(penalized_logp(src_i)), tgt_logp.append(penalized_logp(tgt_i))


# cast to numpy array
preds_similarity = np.array(preds_similarity)
tgt_similarity = np.array(tgt_similarity)
valid = np.array(valid)
src_list_qed = np.array(src_list_qed)
preds_list_qed = np.array(preds_list_qed)
tgt_list_qed = np.array(tgt_list_qed)
preds_string_length = np.array(preds_string_length)
tgt_string_length = np.array(tgt_string_length)
preds_logp = np.array(preds_logp)
tgt_logp = np.array(tgt_logp)
src_logp = np.array(src_logp)
embed()
cands = preds_similarity[preds_list_qed > 0.9]
percent_success = len(cands[cands>.4]) / float(num_evaluate)


#### do this for beam search candidates - compute diversity of that set.
# preds_diversity = mmpa.population_diversity(preds_strs)
# tgt_diversity = mmpa.population_diversity(tgt_strs)


print('prediction mean sim: ', np.mean(preds_similarity), \
		'true mean sim: ', np.mean(tgt_similarity), \
		'percent valid: ', np.mean(valid), \
		'src mean qed: ', np.mean(src_list_qed), \
		'pred mean qed: ', np.mean(preds_list_qed), \
		'tgt mean qed: ', np.mean(tgt_list_qed), \
		'percent greater than .4 sim: ', preds_similarity[preds_similarity>.4].shape[0] / len(preds_similarity),\
		'percent success: ', percent_success,\
		'preds delta logp: ', np.mean(preds_logp-src_logp),\
		'tgt delta logp: ', np.mean(tgt_logp-src_logp))

output_dir = '/tigress/fdamani/mol-edit-output/onmt-qed/output/'+sys.argv[3].split('/')[-1].split('.csv')[0]

if os.path.exists(output_dir):
	print("DIRECTORY EXISTS. Continue to delete.")
	shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir+'/figs')
plt.cla()
plt.hist(np.array(tgt_similarity), bins=40, range=[0,1], alpha=0.5, label='Emp Dist')
plt.hist(np.array(preds_similarity), bins=40, range=[0,1], alpha=0.5, label='Preds')
plt.xlabel("Tanimoto Similarity")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/hist_tm_sim.png')

plt.cla()
plt.hist(src_list_qed, bins=80, range=[0,1], alpha=0.5, label='Input')
plt.hist(tgt_list_qed, bins=80, range=[0,1], alpha=0.5, label='Tgts')
plt.hist(preds_list_qed, bins=80, range=[0,1], alpha=0.5, label='Preds')
plt.xlabel("QED")
plt.ylabel("Count")
plt.xlim([.6, 1.0])
plt.legend()
plt.savefig(output_dir+'/figs/hist_qed.png')

plt.cla()
plt.hist(preds_string_length, range=[10,80], bins=40, alpha=0.5, label='Preds')
plt.hist(tgt_string_length, range=[10,80], bins=40, alpha=0.5, label='Tgts')
plt.xlabel("Smiles Length")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/smiles_length.png')

plt.cla()
plt.hist(preds_logp, bins=40, range=[0,8], alpha=0.5, label='Preds')
plt.hist(tgt_logp, bins=40, range=[0,8], alpha=0.5, label='Tgts')
plt.xlabel("LogP")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/logp.png')

plt.cla()
plt.hist(preds_molwt, bins=40, range=[200,400], alpha=0.5, label='Preds')
plt.hist(tgt_molwt, bins=40, range=[200,400],alpha=0.5, label='Tgts')
plt.xlabel("MolWt")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/molwt.png')