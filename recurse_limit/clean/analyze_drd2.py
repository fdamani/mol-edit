import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../data_analysis')
sys.path.insert(0, '../../props')
import drd2_scorer
# import drd2_scorer

import pandas as pd
import IPython
from IPython import embed

x = pd.read_csv('/tigress/fdamani/mol-edit-data/data/drd2/train_pairs.txt', sep=' ', header=None).values
x = x[:,1]

scores=[]
for i in range(len(x)):
	scores.append(drd2_scorer.get_score(x[i]))
embed()