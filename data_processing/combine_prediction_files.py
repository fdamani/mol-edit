'''take multiple csv prediction files and combine into a single file to read into evaluate_batch'''
import numpy as np
import pandas as pd
import sys
from IPython import embed

num_files = int(sys.argv[1])
combined_dat = []
for i in range(2, num_files+2):
	combined_dat.append(pd.read_csv(sys.argv[i], header=None, skip_blank_lines=False))
combined_dat = pd.concat(combined_dat,axis=1).stack(dropna=False)
combined_dat.to_csv(sys.argv[-1]+'/ensemble_stacked_and_firstiterbeam.csv',header=None,index=None)