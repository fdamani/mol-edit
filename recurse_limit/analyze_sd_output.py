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

max_val = []
top100_vals = 0
#sx = torch.load('logp_sd_seed_prop_vals.pth')
sx = torch.load('qed_sd_seed_prop_vals.pth')

count = 0
for k,v in sx.items():
	count+=1
	print(count)
	max_val.extend(v)
max_val = np.array(max_val)
max_val = np.sort(max_val)[::-1]
embed()