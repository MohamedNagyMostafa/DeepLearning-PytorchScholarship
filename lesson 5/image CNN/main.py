import torch
import torch.nn as nn

#Check CUDA 
train_on_cuda = torch.cuda.is_available()

if not train_on_cuda:
	print('CUDA isn\'t available')
else:
	print('CUDA is available')

