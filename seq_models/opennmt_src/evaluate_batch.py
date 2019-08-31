'''
evaluate set of K candidates for each sentence
read in test-src, test-tgt, and test-tg
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
import scipy
from scipy import stats

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

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
training_src = pd.read_csv(sys.argv[2], header=None)
training_tgt = pd.read_csv(sys.argv[3], header=None)
preds = pd.read_csv(sys.argv[4], header=None, skip_blank_lines=False)
n_best = 20


num_evaluate = src.shape[0]
valid = []
input_strs, tgt_strs, preds_strs = [], [], []
preds_similarity, tgt_similarity = [], []
src_list_qed, tgt_list_qed, preds_list_qed = [], [], []
scaffold_div_train, scaffold_div_preds = [], [] # compute scaffold div using new metric
scaffold_div_train_tm, scaffold_div_preds_tm = [], [] # compute scaffold div using just tm sim w/o scaffold
scaffold_tm_agreement = []
# distributional stats
preds_string_length, tgt_string_length = [], []
preds_molwt, tgt_molwt = [], []
src_logp, preds_logp, tgt_logp = [], [], []
preds_diversity, tgt_diversity = [], []
batch_preds_logp = []
training_src_logp = []
best_sim = []
errors=0
logp_eval = True
save = True
isCandScaffold = False
scaffold_val=-1
src_selfies=True
if logp_eval:
	output_dir = '/tigress/fdamani/mol-edit-output/onmt-logp04/output/'+sys.argv[4].split('/')[-1].split('.csv')[0]
else:
	output_dir = '/tigress/fdamani/mol-edit-output/onmt-qed/output/train_valid_360/'+sys.argv[4].split('/')[-1].split('.csv')[0]

if os.path.exists(output_dir):
	print("DIRECTORY EXISTS. Continue to delete.")
	if os.path.exists(output_dir+'/figs'): 
		shutil.rmtree(output_dir+'/figs')
	if os.path.exists(output_dir+'/data'): shutil.rmtree(output_dir+'/data')
else:
	os.mkdir(output_dir)
os.mkdir(output_dir+'/figs')
os.mkdir(output_dir+'/data')

for i in range(0, n_best*num_evaluate, n_best):
	batch = []
	batch_preds_qed, batch_preds_sim, batch_preds_logp = [], [], []
	isError=False
	for j in range(0, n_best):
		itr = i+j
		if len(preds.iloc[itr].dropna()) == 0:
			#print('nan error')
			continue
		try:
			preds_i = decoder(remove_spaces(''.join(preds.iloc[itr])))
		except:
			embed()
			print ('error')
			if not isError:
				errors+=1
			isError=True
			continue

		if src_selfies:
			src_i = remove_spaces(''.join(src.iloc[int(i/n_best)]))
		else:
			src_i = decoder(remove_spaces(''.join(src.iloc[int(i/n_best)])))
		
		try:
			batch_preds_logp.append(properties.penalized_logp(preds_i))
		except:
			print('cannot compute logp property.')
			embed()
			continue
		batch_preds_qed.append(properties.qed(preds_i))
		batch_preds_sim.append(mmpa.similarity(src_i, preds_i))
		batch.append((src_i, preds_i))
	batch_preds_qed = np.array(batch_preds_qed)
	batch_preds_logp = np.array(batch_preds_logp)
	batch_preds_sim = np.array(batch_preds_sim)
	best_sim.append(np.max(batch_preds_sim))
	print(np.mean(batch_preds_sim), np.max(batch_preds_sim), len(batch_preds_sim))
	cand_sim_inds = np.where(batch_preds_sim > 0.4)[0]
	if len(cand_sim_inds) == 0:
		continue
	try:
		if logp_eval:
			final_ind = cand_sim_inds[np.argmax(batch_preds_logp[cand_sim_inds])]
		else:
			final_ind = cand_sim_inds[np.argmax(batch_preds_qed[cand_sim_inds])]
	except:
		embed()
	src_i, preds_i = batch[final_ind]
	if preds_i == -1:
		valid.append(0)
		print('invalid prediction via selfies decode.')
		continue

	input_qed = properties.qed(src_i)
	pred_qed = properties.qed(preds_i)

	if pred_qed == 0.0:
		valid.append(0)
		print('invalid prediction via qed')
		continue
	src_list_qed.append(input_qed)
	preds_list_qed.append(pred_qed)
	preds_similarity.append(mmpa.similarity(src_i, preds_i))

	input_strs.append(src_i), preds_strs.append(preds_i), valid.append(1)

	preds_string_length.append(len(preds_i))
	preds_molwt.append(molwt(preds_i))
	preds_logp.append(penalized_logp(preds_i)), src_logp.append(penalized_logp(src_i))

	cand_div = []
	cand_div_tm = []
	for cand in cand_sim_inds:
		s,t = batch[cand]
		s_logp = properties.penalized_logp(s)
		t_logp = properties.penalized_logp(t)
		# restrict to candidates where pred is better than s
		if properties.penalized_logp(t) > properties.penalized_logp(s):
			s_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
			t_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(t)))
			cand_div.append(mmpa.similarity(s_core, t_core))
			cand_div_tm.append(mmpa.similarity(s, t))
	cand_div = np.array(cand_div)
	cand_div_tm = np.array(cand_div_tm)
	if len(cand_div) !=0:
		scaffold_div_preds.append(np.min(cand_div))
		scaffold_div_preds_tm.append(np.min(cand_div_tm))
		scaffold_tm_agreement.append(np.argmin(cand_div)==np.argmin(cand_div_tm))
		if np.min(cand_div) < 0.2:
			isCandScaffold = True
			scaffold_val = '%.3f'%(np.min(cand_div))

	if len(cand_sim_inds) >= 1 or isCandScaffold:
		if save:
			smiles_to_save = [] # save src, tgt1, tgt2, ...
			smiles_to_save.append(batch[0][0])
			for cand in cand_sim_inds:
				s, t = batch[cand]
				s_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
				t_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(t)))
				smiles_to_save.append(t)
			if isCandScaffold:
				save_path = output_dir+'/data/'+str(i)+'_diverse_'+str(scaffold_val)+'_translation_example.csv'
			else:
				save_path = output_dir+'/data/'+str(i)+'_translation_example.csv'

			pd.DataFrame(smiles_to_save).to_csv(save_path, header=None, index=None)
		isCandScaffold = False

# compute empirical distribution over target data
end = training_tgt.shape[0]
for i in range(0, end):
	tgt_smiles = decoder(remove_spaces(''.join(training_tgt.iloc[i])))
	src_smiles = decoder(remove_spaces(''.join(training_src.iloc[i])))
	tgt_similarity.append(mmpa.similarity(src_smiles, tgt_smiles))
	tgt_qed = properties.qed(tgt_smiles)
	tgt_list_qed.append(tgt_qed)
	tgt_string_length.append(len(tgt_smiles))
	tgt_molwt.append(molwt(tgt_smiles))
	tgt_logp.append(penalized_logp(tgt_smiles))
	training_src_logp.append(penalized_logp(src_smiles))

	s_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(src_smiles)))
	t_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(tgt_smiles)))
	scaffold_div_train.append(mmpa.similarity(s_core, t_core))
	if i % 500 == 0 and i!=0:
		break

# cast to numpy array
scaffold_div_preds = np.array(scaffold_div_preds)
scaffold_div_train = np.array(scaffold_div_train)
scaffold_tm_agreement = np.array(scaffold_tm_agreement)
scaffold_div_preds_tm = np.array(scaffold_div_preds_tm)
best_sim = np.array(best_sim)
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
if logp_eval:
	percent_success = preds_logp.shape[0] / (float(num_evaluate)-errors)
else:
	cands = preds_similarity[preds_list_qed > 0.9]
	percent_success = len(cands[cands>.4]) / (float(num_evaluate)-errors)

#### do this for beam search candidates - compute diversity of that set.
# preds_diversity = mmpa.population_diversity(preds_strs)
# tgt_diversity = mmpa.population_diversity(tgt_strs)

embed()
print('prediction mean sim: ', np.mean(best_sim), \
		'\ntrue mean sim: ', np.mean(tgt_similarity), \
		'\npercent valid: ', np.mean(valid), \
		'\nsrc mean qed: ', np.mean(src_list_qed), \
		'\npred mean qed: ', np.mean(preds_list_qed), \
		'\ntgt mean qed: ', np.mean(tgt_list_qed), \
		'\npercent greater than .4 sim: ', preds_similarity[preds_similarity>.4].shape[0] / len(preds_similarity),\
		'\npercent success: ', percent_success,\
		'\npreds delta logp: ', np.mean(preds_logp-src_logp), np.std(preds_logp-src_logp), \
		'\ntgt delta logp: ', np.mean(tgt_logp-training_src_logp))
embed()
# we want to compute spearman rank correlation between scaffold_div_preds and scaffold_div_preds_tm
# compute mean of scaffold_tm_agreement
print('scaffold and tm div agreement: ', np.mean(scaffold_tm_agreement))
print('spearman rank corr: ', scipy.stats.spearmanr(scaffold_div_preds, scaffold_div_preds_tm))

plt.cla()
plt.hist(np.array(tgt_similarity), bins=40, alpha=0.5, density=True, label='Training Emp Dist')
plt.hist(np.array(best_sim), bins=40,  alpha=0.5, density=True, label='Test Preds')
plt.xlabel("Tanimoto Similarity")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/hist_tm_sim.png')

plt.cla()
plt.hist(src_list_qed, bins=80, alpha=0.5, density=True, label='Test Input')
plt.hist(tgt_list_qed, bins=80, alpha=0.5, density=True, label='Training Tgts')
plt.hist(preds_list_qed, bins=80, alpha=0.5, density=True, label='Test Preds')
plt.xlabel("QED")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/hist_qed.png')

plt.cla()
plt.hist(preds_string_length,  bins=40, alpha=0.5, density=True, label='Test Preds')
plt.hist(tgt_string_length,  bins=40, alpha=0.5, density=True, label='Training Tgts')
plt.xlabel("Smiles Length")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/smiles_length.png')

plt.cla()
plt.hist((preds_logp-src_logp), bins=40, alpha=0.5, density=True, label='Test Preds')
plt.hist((tgt_logp-training_src_logp), bins=40, alpha=0.5, density=True, label='Training Tgts')
plt.xlabel("Delta LogP")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/logp.png')

plt.cla()
plt.hist(preds_logp, bins=40, alpha=0.5, density=True, label='Preds')
plt.hist(src_logp, bins=40, alpha=0.5, density=True, label='Seeds')
plt.xlabel("Delta LogP")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/logp.png')



plt.cla()
plt.hist(preds_molwt, bins=40, alpha=0.5, density=True, label='Test Preds')
plt.hist(tgt_molwt, bins=40, alpha=0.5, density=True, label='Training Tgts')
plt.xlabel("MolWt")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/molwt.png')

plt.cla()
plt.hist(scaffold_div_preds, bins=40, alpha=0.5, density=True, label='Test Preds')
plt.hist(scaffold_div_train, bins=40, alpha=0.5, density=True, label='Training Tgts')
plt.xlabel("Scaffold Div.")
plt.ylabel("Count")
plt.legend()
plt.savefig(output_dir+'/figs/scaffold_div.png')