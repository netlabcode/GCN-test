import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time

import matplotlib.pyplot as plt


A = np.load('lstm_loss.npy', allow_pickle=True)


df1 = pd.DataFrame({"idx":[],"val":[]})

c=0
while(c<10):
    #print(A[0][c].data.cpu().numpy())
    x = float(A[3][c].data.cpu().numpy())
    df2 = pd.DataFrame({"idx":[c],"val":[x]})
    df1 = df1.append(df2, ignore_index=True)
    c=c+1

fig, ax = plt.subplots()
plt.plot(df1.idx,df1.val,  label = 'LSTM')

plt.ylabel('Validation Loss (MSE)', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
# plt.gca().invert_xaxis()
plt.legend(fontsize=14)
plt.grid(True, which='both')

plt.savefig('LSTM-Loss.png', dpi=300, bbox_inches = 'tight', pad_inches=0.1)


