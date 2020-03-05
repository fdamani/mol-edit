'''take translation output (20 candidates per seed)
return best candidate per seed by SA
'''
import numpy as np
import pandas as pd
import sys
from IPython import embed
sys.path.insert(0, '../data_analysis/')
import properties
from IPython import embed
import selfies
from selfies import encoder, decoder

def remove_spaces(x):
	return x.replace(" ", "")

dat = pd.read_csv(sys.argv[1], header=None, skip_blank_lines=False)
output_dir = sys.argv[2]
n_best = 20
num_evaluate = int(dat.shape[0] / float(n_best))
top_compd_sa = []
for i in range(0, n_best*num_evaluate, n_best):
	sa_list = []
	for j in range(0, n_best):
		itr = i+j
		try:
			cpd = decoder(remove_spaces(dat.iloc[itr].values[0]))
			sa_cpd = properties.sa(cpd)
		except:
			cpd = ''
			sa_cpd = -100
		if len(cpd) < 100:
			sa_cpd = -100
		sa_list.append(sa_cpd)
	max_sa_ind = np.argmax(sa_list)
	compd = dat.iloc[i + max_sa_ind].values[0]
	top_compd_sa.append(compd)
top_compd_sa = pd.DataFrame(top_compd_sa)
top_compd_sa.to_csv(output_dir+'/firstiter-20beam-20best-rankbysa.csv', index=None, header=None)