'''
seq2seq model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import rnn
import rdkit as rd
import pandas as pd
import IPython
import utils
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
import selfies

from selfies import encoder, decoder
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score
from IPython import embed
from torch import optim
import random

# read and process data
