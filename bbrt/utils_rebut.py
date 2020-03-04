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


def morgan_fpt(x):
	return properties.morgan_fpt(x)
def remove_spaces(x):
	return x.replace(" ", "")

def smiles_to_selfies(x, token_sep=True):
	"""smiles to selfies
	if token_sep=True -> return spaces between each token"""
	output = []
	for i in range(x.shape[0]):
		ax = encoder(x[i])
		if ax != -1:
			if token_sep:
				sx = re.findall(r"\[[^\]]*\]", ax)
				ax = ' '.join(sx)
			output.append(ax)
		else:
			output.append('NaN')
	return output
def selfies_to_smiles(x):
	return decoder(x)

def qed(x):
	return properties.qed(x)

def logp(x):
	return properties.penalized_logp(x)

def drd2(x):
	return drd2_scorer.get_score(x)

def prop_array(x, prop='logp04', prev_x=None, seed_sim=None):
	vals = []

	for i in range(len(x)):
		if prop == 'logp04':
			try:
				px = logp(x[i])
			except:
				px = None
		elif prop == 'drd2':
			try:
				px = drd2(x[i])
			except:
				px = None
		elif prop == 'maxsim':
			if prev_x:
				if score_dict[prop](x[i]) > score_dict[prop](prev_x):
					px = mmpa.similarity(x[i], prev_x)
				else:
					px = None
			else:
				print('error')
				sys.exit(0)
		elif prop == 'minmolwt':
			if score_dict[prop](x[i]) > score_dict[prop](prev_x):
				px = -rdkit.Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(sx))
			else:
				px = None
		elif prop == 'maxseedsim':
			if seed_sim:
				if score_dict[prop](x[i]) > score_dict[prop](seed_sim):
					px = mmpa.similarity(x[i], seed_sim)
				else:
					px = None
			else:
				px = None
		elif prop == 'qed':
			try:
				px = qed(x[i])
			except:
				px = None
		vals.append(px)
	return vals
def sim(a,b):
	return mmpa.similarity(a,b)

def clean(x):
	return selfies_to_smiles(remove_spaces(x))

def clean_array(x):
	sx = []
	for i in range(len(x)):
		if len(x[i])==1:
			sx.append(clean(x[i][0]))
		else:
			sx.append(clean(x[i]))
	return sx
