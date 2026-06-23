import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class SimpleCNNClassifier(nn.Module):
	def __init__(
			self,
			batch_norm:bool,
		):
		super().__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=32,
				kernel_size=3
			),
			nn.BatchNorm2d(num_features=32) if batch_norm else nn.Identity(),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		

		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,
				out_channels=64,
				kernel_size=3
			),
			nn.BatchNorm2d(num_features=64) if batch_norm else nn.Identity(),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		) 

		self.conv3 = nn.Sequential(
			nn.Conv2d(
				in_channels=64,
				out_channels=128,
				kernel_size=3
			),
			nn.BatchNorm2d(num_features=128) if batch_norm else nn.Identity(),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		) 

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(
				in_features=128*31*56, 
				out_features=128
			),
			nn.BatchNorm1d(num_features=128) if batch_norm else nn.Identity(),
			nn.ReLU(),
			nn.Linear(
				in_features=128, 
				out_features=64
			),
			nn.BatchNorm1d(num_features=64) if batch_norm else nn.Identity(),
			nn.ReLU(),
			nn.Linear(
				in_features=64, 
				out_features=16
			),
			nn.BatchNorm1d(num_features=16) if batch_norm else nn.Identity(),
			nn.ReLU(),
			nn.Linear(
				in_features=16, 
				out_features=2
			)
		)


	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.classifier(x)
		return x



if __name__ == "__main__":
	model = SimpleCNNClassifier(batch_norm=True)
	summary(model, input_size=(1, 3, 267, 464))