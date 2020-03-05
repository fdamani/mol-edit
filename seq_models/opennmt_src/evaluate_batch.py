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
second_iter_seeds = pd.read_csv(sys.argv[5], header=None, skip_blank_lines=False)
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
prop_unique_scaffolds_per_seed = []
prop_rgroup_edits = []
prop_scaffold_edits = []
mean_sa, max_sa = [], []
src_sa= []
training_tgt_sa = []
second_iter_seeds_sa= []

mean_src_num_h_acceptors, mean_tgt_num_h_acceptors, mean_src_num_h_donors, \
	mean_tgt_num_h_donors, mean_src_num_rotatable, mean_tgt_num_rotatable = [], [], [], [], [], []

if logp_eval:
	output_dir = '/tigress/fdamani/mol-edit-output/onmt-logp04/output/top800_top_90/'+sys.argv[4].split('/')[-1].split('.csv')[0]
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
	unique_set = []
	for j in range(0, n_best):
		itr = i+j
		if len(preds.iloc[itr].dropna()) == 0:
			#print('nan error')
			continue
		try:
			preds_i = decoder(remove_spaces(''.join(preds.iloc[itr])))
			# remove duplicate predictions
			if preds_i not in unique_set:
				unique_set.append(preds_i)
			else:
				continue
		except:
			embed()
			print ('error')
			if not isError:
				errors+=1
			isError=True
			continue
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

	preds_logp.append(penalized_logp(preds_i)), src_logp.append(penalized_logp(src_i))

	# compute other distributional stats, num rotatable bonds, num hbond acceptors, num hbond donors
	src_num_h_acceptors, tgt_num_h_acceptors, src_num_h_donors, tgt_num_h_donors = [], [], [], []
	src_num_rotatable, tgt_num_rotatable = [], []
	local_mean_sa, local_max_sa = [], 0
	local_src_sa = []
	for j in range(len(cand_sim_inds)):
		s,t = batch[cand_sim_inds[j]]
		mol_s = Chem.MolFromSmiles(s)
		mol_t = Chem.MolFromSmiles(t)
		src_num_h_acceptors.append(rdkit.Chem.Lipinski.NumHAcceptors(mol_s))
		tgt_num_h_acceptors.append(rdkit.Chem.Lipinski.NumHAcceptors(mol_t))
		src_num_h_donors.append(rdkit.Chem.Lipinski.NumHDonors(mol_s))
		tgt_num_h_donors.append(rdkit.Chem.Lipinski.NumHDonors(mol_t))
		src_num_rotatable.append(rdkit.Chem.Lipinski.NumRotatableBonds(mol_s))
		tgt_num_rotatable.append(rdkit.Chem.Lipinski.NumRotatableBonds(mol_t))

		try:
			sa_prop = properties.sa(t)
			src_sa_prop = properties.sa(s)
		except:
			continue
		local_src_sa.append(src_sa_prop)
		local_mean_sa.append(sa_prop)
		if sa_prop > local_max_sa:
			local_max_sa = sa_prop

	if local_max_sa != 0:
		max_sa.append(local_max_sa)
		mean_sa.append(np.mean(local_mean_sa))


	src_sa.append(np.mean(local_src_sa))

	mean_src_num_h_acceptors.append(np.mean(src_num_h_acceptors))
	mean_tgt_num_h_acceptors.append(np.mean(tgt_num_h_acceptors))
	mean_src_num_h_donors.append(np.mean(src_num_h_donors))
	mean_tgt_num_h_donors.append(np.mean(tgt_num_h_donors))
	mean_src_num_rotatable.append(np.mean(src_num_rotatable))
	mean_tgt_num_rotatable.append(np.mean(tgt_num_rotatable))

	# entropy of candidate set: pairwise tm sim between scaffolds 
	unique_scaffolds = []
	num_rgroup_edits = 0
	num_scaffold_edits = 0
	for j in range(len(cand_sim_inds)):
		s,t = batch[cand_sim_inds[j]]
		t_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(t)))
		s_core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
		if t_core not in unique_scaffolds:
			unique_scaffolds.append(t_core)
		if t_core == s_core:
			num_rgroup_edits += 1
		else:
			num_scaffold_edits += 1
	prop_unique_scaffolds = len(unique_scaffolds)/float(len(cand_sim_inds))
	if len(cand_sim_inds) > 4:
		prop_unique_scaffolds_per_seed.append(prop_unique_scaffolds)
		prop_rgroup_edits.append(num_rgroup_edits / float(len(cand_sim_inds)))
		prop_scaffold_edits.append(num_scaffold_edits / float(len(cand_sim_inds)))

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
	
			preds_molwt.append(molwt(t))
			preds_string_length.append(len(t))

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
	training_tgt_sa.append(properties.sa(tgt_smiles))
	if i == 5000:
		break

end = second_iter_seeds.shape[0]
for i in range(0, end):
	try:
		dx = decoder(remove_spaces(''.join(second_iter_seeds.iloc[i])))
		second_iter_seeds_sa.append(properties.sa(dx))
	except:
		continue
embed()
# cast to numpy array
mean_tgt_num_rotatable = np.array(mean_tgt_num_rotatable)
mean_src_num_rotatable = np.array(mean_src_num_rotatable)
mean_tgt_num_h_donors = np.array(mean_tgt_num_h_donors)
mean_src_num_h_donors = np.array(mean_src_num_h_donors)
mean_tgt_num_h_acceptors = np.array(mean_tgt_num_h_acceptors)
mean_src_num_h_acceptors = np.array(mean_src_num_h_acceptors)
max_sa = np.array(max_sa)
mean_sa = np.array(mean_sa)
src_sa = np.array(src_sa)
second_iter_seeds_sa = np.array(second_iter_seeds_sa)
training_tgt_sa = np.array(training_tgt_sa)
prop_unique_scaffolds_per_seed = np.array(prop_unique_scaffolds_per_seed)
prop_rgroup_edits = np.array(prop_rgroup_edits)
prop_scaffold_edits = np.array(prop_scaffold_edits)
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

# we want to compute spearman rank correlation between scaffold_div_preds and scaffold_div_preds_tm
# compute mean of scaffold_tm_agreement
print('scaffold and tm div agreement: ', np.mean(scaffold_tm_agreement))
print('spearman rank corr: ', scipy.stats.spearmanr(scaffold_div_preds, scaffold_div_preds_tm))

plt.cla()
fig,ax = plt.subplots(1)
plt.hist(np.array(tgt_similarity), bins=40, alpha=0.5, density=True, label='Training Emp Dist')
plt.hist(np.array(best_sim), bins=40,  alpha=0.5, density=True, label='Test Preds Best Sim')
plt.xlabel("Tanimoto Similarity Max")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/hist_tm_simmax.png')

plt.cla()
fig,ax=plt.subplots(1)
plt.hist(np.array(tgt_similarity), bins=40, alpha=0.5, density=True, label='Training Emp Dist')
plt.hist(np.array(preds_similarity), bins=40,  alpha=0.5, density=True, label='Test Preds')
plt.xlabel("Tanimoto Similarity")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
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
plt.savefig(output_dir+'/figs/deltalogp.png')

plt.cla()
fig,ax=plt.subplots(1)
plt.hist(preds_logp, alpha=0.5, density=True, label='Preds')
plt.hist(src_logp, alpha=0.5, density=True, label='Seeds')
plt.xlabel("Penalized LogP")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/logp.png')

plt.cla()
fig,ax=plt.subplots(1)
plt.hist(preds_molwt, bins=40, alpha=0.5, density=True, label='Test Preds')
plt.hist(tgt_molwt, bins=40, alpha=0.5, density=True, label='Training Tgts')
plt.xlabel("MolWt")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/molwt.png')

plt.cla()
fig,ax=plt.subplots(1)
plt.hist(scaffold_div_preds, bins=40, alpha=0.5, density=True, label='Test Preds')
plt.hist(scaffold_div_train, bins=40, alpha=0.5, density=True, label='Training Tgts')
plt.xlabel("Scaffold Div.")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/scaffold_div.png')


plt.cla()
fig,ax=plt.subplots(1)
plt.hist(prop_unique_scaffolds_per_seed, alpha=0.5, density=True, label='Test Preds')
plt.xlabel("Prop. Unique Scaffolds Per Seed")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/entropy_prop_unique_scaffolds.png')


plt.cla()
fig,ax=plt.subplots(1)
plt.hist(prop_rgroup_edits, alpha=0.5, density=True, label='Test Preds')
plt.xlabel("Prop. R Group Edits Per Seed")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/prop_rgroup_edits.png')


plt.cla()
fig,ax=plt.subplots(1)
plt.hist(prop_scaffold_edits, alpha=0.5, density=True, label='Test Preds')
plt.xlabel("Prop. Scaffold Edits Per Seed")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/prop_scaffold_edits.png')


################################################################
plt.cla()
fig,ax=plt.subplots(1)
plt.hist(mean_tgt_num_rotatable, alpha=0.5, density=True, label='Preds')
plt.hist(mean_src_num_rotatable, alpha=0.5, density=True, label='Seeds')
plt.xlabel("Mean Rotatable Bonds")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/rotatable_bonds.png')


plt.cla()
fig,ax=plt.subplots(1)
plt.hist(mean_tgt_num_h_acceptors, alpha=0.5, density=True, label='Preds')
plt.hist(mean_src_num_h_acceptors, alpha=0.5, density=True, label='Seeds')
plt.xlabel("Mean Num Hydrogen Acceptors")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/hydrogen_acceptors.png')

plt.cla()
fig,ax=plt.subplots(1)
plt.hist(mean_tgt_num_h_donors, alpha=0.5, density=True, label='Preds')
plt.hist(mean_src_num_h_donors, alpha=0.5, density=True, label='Seeds')
plt.xlabel("Mean Num Hydrogen Donors")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/hydrogen_donors.png')

##########################################################################
# SA plots
plt.cla()
fig,ax=plt.subplots(1)
plt.hist(max_sa, alpha=0.5, density=True, label='Preds')
plt.hist(src_sa, alpha=0.5, density=True, label='Seeds')
plt.hist(second_iter_seeds_sa, alpha=0.5, density=True, label='Top SA Seeds')
plt.xlabel("Max SA")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/max_sa.png')

plt.cla()
fig,ax=plt.subplots(1)
plt.hist(mean_sa, alpha=0.5, density=True, label='Preds')
plt.hist(src_sa, alpha=0.5, density=True, label='Seeds')
plt.hist(second_iter_seeds_sa, alpha=0.5, density=True, label='Top SA Seeds')
plt.xlabel("Mean SA")
plt.ylabel("Count")
plt.legend()
ax.set_yticklabels([])
plt.savefig(output_dir+'/figs/mean_sa.png')

embed()