from torch.nn import Module
from torch import nn
import torch.nn.functional as F

class NeuralClassifier(Module):
	def __init__(self):
		super().__init__()
		n_input 	= 784
		n_hidden1 	= 196
		n_hidden2 	= n_hidden1
		n_output	= 10

		self.L1 = nn.Linear(n_input, n_hidden1)
		self.L2 = nn.Linear(n_hidden1, n_hidden2)
		self.L3 = nn.Linear(n_hidden2, n_output)

	def forward(self, images):
		x = images.view(images.shape[0], -1)

		z1 = self.L1(x)
		a1 = F.relu(z1)
		z2 = self.L2(a1)
		a2 = F.relu(z2)
		z3 = self.L3(a2)
		a3 = F.log_softmax(z3, dim=1)

		return a3
