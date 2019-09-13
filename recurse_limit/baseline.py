'''take as input top 800 seed compunds
compute population-level diversity measure: mean (1- sim(x,y)), for all x,y
'''
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../data_analysis')
import properties
import mmpa
from IPython import embed


def sim(x,y):
	return mmpa.similarity(x,y)

def pop_div(X):
	'''input list of compounds
	return  mean(1-sim(x,y))'''
	divs = []
	for i in range(X.shape[0]):
		for j in range(X.shape[0]):
			if j > i:
				divs.append(1.0 - sim(X.iloc[i].values[0], X.iloc[j].values[0]))
		print(i)
		if i == 50:
			break
	divs = np.array(divs)
	return np.mean(divs), np.std(divs)

seeds = pd.read_csv(sys.argv[1], header=None, skip_blank_lines=False)
embed()
div_mu, div_std = pop_div(seeds)
embed()