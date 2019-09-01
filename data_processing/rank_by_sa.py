'''take translation output (20 candidates per seed)
return best candidate per seed by SA
'''
import numpy as np
import pandas as pd
import sys
from IPython import embed
sys.path.insert(0, '../data_analysis/')
import properties

dat = pd.read_csv(sys.argv[1], header=None, skip_blank_lines=False)
n_best = 20
num_evaluate = dat.shape[0]
for i in range(0, n_best*num_evaluate, n_best):
	for j in range(0, n_best):
		itr = i+j
		properties.sa(dat.iloc[itr])