'''take csv file of smiles and convert to selfies with spaces to be read in translate'''
import pandas as pd
from selfies import encoder, decoder
from IPython import embed
import sys
from process_data import smiles_to_selfies

X = pd.read_csv(sys.argv[1], header=None, skip_blank_lines=False)
output_dir = sys.argv[2]
selfies_output = smiles_to_selfies(X.values.reshape(-1))
pd.DataFrame(selfies_output).to_csv(output_dir+'/jin_test_selfies.csv', index=None, header=None)