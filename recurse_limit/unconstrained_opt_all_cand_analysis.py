'''
given a directory, search through all generated candidates across
a) different init seeds 
b) decoding methods 
c) ranking methods

histogram of all compounds and values
- what percent is valid?
- of the top 100 compounds, what is the entropy of these compounds? (avg pairwise TM sim)
'''
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

init_seeds = ['jin_test', 'src_train_900maxdiv_seeds']
decoding_methods = ['beam', 'softmax_randtop2', 'softmax_randtop5']
rank_types = ['logp', 'maxdeltasim', 'mindeltasim', 'minmolwt']

xdir = '/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/'
num_files = 20
for seed in init_seeds:
	for dec in decoding_methods:
		for rt in rank_types:
			dr = xdir+seed+'/'+dec+'/'+rt
			for i in range(num_files):
				sx = pd.read_csv(dr+'/'+str(i)+'.csv', header=None, skip_blank_lines=False)
				for j in range(sx.shape[0]):
					embed()
					properties.penalized_logp(sx.iloc[j])