from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

def downloadData():
	transform = transforms.Compose(
		[transforms.ToTensor(), 
		transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		])
	trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
	trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
	return iter(trainLoader).next()

def activation_function (z):
	return 1/(1+torch.exp(-z))

def showImage(image):
	plt.imshow(image.numpy().squeeze(), cmap='Greys_r')
	plt.show()


images, labels = downloadData()

n_input = 784
n_hidden = 256
n_output = 10

#one hot conversion
identity = torch.eye(10)
one_hot_labels = torch.tensor(identity[labels[:]])

#Weights, bias and features
W1 = torch.randn((n_input, n_hidden))
W2 = torch.randn((n_hidden, n_output))

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

features = images.view(64, 784)

#Feed forward
z1 = torch.mm(features, W1) + B1
a1 = activation_function(z1)
z2 = torch.mm(a1, W2) + B2
a2 = activation_function(z2)

print(labels[12])
showImage(features[12].view(28,28))
