"""Common utilities for CUDA operations in TVMC"""

import torch

NGPU = 1
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
print("Using device: ", DEVICE)
