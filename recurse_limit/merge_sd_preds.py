import numpy as np
import pandas as pd
import sys
from IPython import embed
import os

input_dir1 = sys.argv[1]
input_dir2 = sys.argv[2]
input_dir3 = sys.argv[3]
#input_dir4 = sys.argv[4]

output_dir = sys.argv[4]
combined_dat = []

dirs = [input_dir1, input_dir2, input_dir3]#, input_dir4]
for dr in dirs:
	for file in os.listdir(dr):
		try:
			combined_dat.append(pd.read_csv(dr+'/'+file, header=None, skip_blank_lines=False))
		except:
			print('error')
			continue
combined_dat = pd.concat(combined_dat,axis=1).stack(dropna=False)
combined_dat.to_csv(output_dir+'/stochastic_decoding_qed.csv',header=None,index=None)