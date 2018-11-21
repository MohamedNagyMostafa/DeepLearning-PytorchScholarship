import torch
from torchvision import datasets, transforms

class Data():

	def __init__(self):
		transform = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

		trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, 
			train=True, transform = transform)
		self.trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

		testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, 
			train=False, transform = transform)
		self.testLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

	def trainData(self):
		return self.trainLoader

	def testData(self):
		return self.testLoader