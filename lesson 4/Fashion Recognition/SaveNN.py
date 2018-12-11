import torch

class SaveNN():
	def __init__(self, name):
		name += '.pth'

	def save(self, n_input, n_output):
		checkpoint = {
			'input_size':n_input,
			'output_size':n_output,
			'hidden_layers':[each.out_features for each in model.hidden_layers],
			'state_dict': model.state_dict()
		}
		torch.save(checkpoint, self.name)

	def load(self, file_path):
		checkpoint = torch.load(file_path)
		return checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], checkpoint['state_dict']
