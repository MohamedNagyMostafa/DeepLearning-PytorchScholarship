from torchvision import datasets
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

class Data:
	def __init__(self):
		num_workers	= 0
		batch_size		= 20
		valid_size		= 0.2

		#Transform
		transform 	= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
		#Train & Test data
		train_data 	= datasets.CIFAR10('data',train=True, download=True, transform = transform)
		test_data 	= datasets.CIFAR10('data',train=False, download=True, transform = transform)
		#Determine train indexes
		num_train 	= len(train_data)
		indices		= list(range(num_train))
		np.random.shuffle(indices)
		split 		= int(np.floor(num_train * valid_size))
		train_idx, valid_idx 	= indices[split:], indices[:split]
		#define samplers
		train_sampler 	= SubsetRandomSampler(train_idx)
		valid_sampler 	= SubsetRandomSampler(valid_idx)
		#loaders
		self.train_loader	= torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler= train_sampler, num_workers=num_workers)
		self.valid_loader	= torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler= train_sampler, num_workers=num_workers)
		self.test_loader		= torch.utils.data.DataLoader(test_data,  batch_size = batch_size, num_workers = num_workers)

	def train_loader(self):
		return self.train_loader

	def valid_loader(self):
		return self.valid_loader

	def test_loader(self):
		return self.test_loader