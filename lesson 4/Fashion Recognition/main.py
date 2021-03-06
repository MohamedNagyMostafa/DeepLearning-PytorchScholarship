from Data import Data
from NeuralClassifier import NeuralClassifier
import torch
import matplotlib.pyplot as plt

#Load data
data 		= Data()

#Identify Model
epochs		= 10

model 		= NeuralClassifier()
optimizer 	= torch.optim.SGD(model.parameters(), lr=0.01)
criterion 	= torch.nn.NLLLoss()

#Train Model
train_losses = []
test_losses = []
for e in range(epochs):
	train_loss 	= 0
	test_loss 	= 0
	accuracy 	= 0
	for images, labels in data.trainData():
		optimizer.zero_grad()
		a3 		= model(images)
		loss 	= criterion(a3, labels)

		loss.backward()
		optimizer.step()

		train_loss += loss
	else:
		with torch.no_grad():
			model.eval()
			for images, labels in data.testData():

				a3 		= model(images)
				loss	= criterion(a3, labels)

				prediction = torch.exp(a3)
				p_classes, top_classes = prediction.topk(1, dim=1)

				equal = top_classes == labels.view(*top_classes.shape)

				accuracy  += torch.mean(equal.type(torch.FloatTensor))
				test_loss += loss

		train_losses.append(train_loss/len(data.trainData()))
		test_losses.append(test_loss/len(data.testData()))

		model.train()
		print('Epoch {}/{}\n\tTrain Loss: {}\n\tTest Loss: {}\n\tAccuracy: {}'
				.format(e+1, epochs, train_losses[e], test_losses[e], (accuracy * 100)/len(data.testData())))

print(model)

plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Testing loss")
plt.legend(frameon=False)
plt.show()