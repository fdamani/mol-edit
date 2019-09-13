import numpy as np
import pandas as pd
import sys
from IPython import embed
import os

num_files = int(sys.argv[1])
input_dir = sys.argv[2]
output_dir = sys.argv[3]
file_num = sys.argv[4]
combined_dat = []
for i in range(1, num_files+1):
	combined_dat.append(pd.read_csv(input_dir+'/'+str(i)+'.csv', header=None, skip_blank_lines=False))
combined_dat = pd.concat(combined_dat,axis=1).stack(dropna=False)
combined_dat.to_csv(output_dir+'/'+str(file_num)+'.csv',header=None,index=None)