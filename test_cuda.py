import torch
import sys


print(sys.version)

print('version', torch.__version__)

print('cuda', torch.cuda.is_available())

print('cuda version', torch.version.cuda)