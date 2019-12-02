import sys
sys.path.insert(0, '../../data_analysis')
sys.path.insert(0, '../../data_processing')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import IPython
import selfies
import numpy as np
from IPython import embed
import properties
import mmpa
from selfies import encoder, decoder
import process_data

def remove_spaces(x):
	return x.replace(" ", "")

def selfies_to_smiles(x):
	return decoder(x)
def logp(x):
	return properties.penalized_logp(x)

def sim(a,b):
	return mmpa.similarity(a,b)

def clean(x):
	return selfies_to_smiles(remove_spaces(x))


# src = pd.read_csv('/tigress/fdamani/mol-edit-data/data/logp04/src_train.csv', header=None).values
# tgt = pd.read_csv('/tigress/fdamani/mol-edit-data/data/logp04/tgt_train.csv', header=None).values

pairs = pd.read_csv(sys.argv[1], sep=' ', header=None)
pairs = pairs.sample(frac=1).reset_index(drop=True).values

logp = False
if logp:
	sim_thresh = .80
	sim_pairs = []
	for i in range(len(pairs)):
		if sim(pairs[i][0], pairs[i][1]) > sim_thresh:
			sim_pairs.append(pairs[i])
	pairs = np.array(sim_pairs)

output_dir = sys.argv[2]
num_samples = len(pairs)
num_valid = int(.1*num_samples)
sims = []
src = pairs[:,0]
tgt = pairs[:,1]

src_selfies = process_data.smiles_to_selfies(src)
tgt_selfies = process_data.smiles_to_selfies(tgt)

src_valid = src_selfies[0:num_valid]
tgt_valid = tgt_selfies[0:num_valid]

src_train = src_selfies[num_valid:]
tgt_train = tgt_selfies[num_valid:]

pd.DataFrame(src_valid).to_csv(output_dir+'/src_valid.csv', index=None,header=None)
pd.DataFrame(tgt_valid).to_csv(output_dir+'/tgt_valid.csv', index=None,header=None)

pd.DataFrame(src_train).to_csv(output_dir+'/src_train.csv', index=None,header=None)
pd.DataFrame(tgt_train).to_csv(output_dir+'/tgt_train.csv', index=None,header=None)