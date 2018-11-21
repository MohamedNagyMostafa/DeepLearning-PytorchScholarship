from torchvision import datasets, transforms
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F

def downloadData():
	transform = transforms.Compose(
		[transforms.ToTensor(), 
		transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		])
	trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
	trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
	return trainLoader



def showImage(image):
	plt.imshow(image.numpy().squeeze(), cmap='Greys_r')
	plt.show()

n_input = 784
n_hidden1 = 128
n_hidden2 = 64
n_output = 10
epochs = 5
#initial model
model = nn.Sequential(
	nn.Linear(n_input, n_hidden1),
	nn.ReLU(),
	nn.Linear(n_hidden1, n_hidden2),
	nn.ReLU(),
	nn.Linear(n_hidden2,n_output),
	nn.LogSoftmax(dim=1)
	)

optimizer = optim.SGD(model.parameters(), lr=0.01)

for e in range(epochs):
	training = 0
	for images, labels in downloadData():
		images = images.view(images.shape[0], -1)

		optimizer.zero_grad()
		#criterion = nn.CrossEntropyLoss()
		criterion = nn.NLLLoss()

		logist = model(images)

		loss = criterion(logist, labels)

		loss.backward()
		optimizer.step()

		training += loss.item()
	else:
		print(training/len(downloadData()))		
