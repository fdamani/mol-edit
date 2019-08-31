'''take a list of compounds (train_pairs file) and return the top 100 compounds with the highest logp values.
'''

import numpy as np
import pandas as pd
import sys
from IPython import embed
sys.path.insert(0, '../data_analysis/')
import properties

train_pairs = pd.read_csv(sys.argv[1], delimiter=' ', header=None)
output_dir = sys.argv[2]
data = pd.concat([train_pairs[0], train_pairs[1]]).drop_duplicates()
logp_vals = []
topk = 2000
for i in range(len(data)):
	logp_vals.append(properties.penalized_logp(data.iloc[i]))
logp_vals = np.array(logp_vals)
sorted_inds = np.argsort(logp_vals)
sorted_vals = np.sort(logp_vals)

bottom_10_pct_thresh = np.percentile(sorted_vals, 10)
bottom_10_pct_inds = sorted_inds[sorted_vals < bottom_10_pct_thresh]
bottom_10_pct_logp = data.iloc[bottom_10_pct_inds]
bottom_10_pct_logp.to_csv(output_dir+'/bottom_10_pct_logp.csv', header=None, index=None)
embed()