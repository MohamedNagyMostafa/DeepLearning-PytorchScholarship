import torch.nn.functional as F
class NeuralNetwork(nn.Module):
	def __init__(self, n_input, n_hidden, n_output):
		super.__init__()
		#Add bias by default
		self.hidden = nn.Linear(n_input, n_hidden)
		self.output = nn.Linear(n_hidden, n_output)

		#self.sigmoid = nn.Sigmoid()
		#self.softmax = nn.Softmax(dim = 1)

	def forward(self, x):
		x = F.sigmoid(self.hidden(x))
		x = F.softmax(self.output(x), dim=1)
		#x = self.hidden(x)
		#x = self.sigmoid(x)
		#x = self.output(x)
		#x = self.softmax(x)

		return x