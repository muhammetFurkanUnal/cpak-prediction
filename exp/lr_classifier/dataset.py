import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split 
from shared import Sample

class _LRClassifierDataset(Dataset):
	def __init__(
			self, 
			dataframe, 
			root_dir, 
			transform=None
		):
		self.df = dataframe
		self.root_dir = root_dir
		self.transform = transform		

	def __len__(self):
		return len(self.df)
	

	def __getitem__(self, index):
		img_name = self.df.iloc[index]["image"]
		img_path = os.path.join(
			self.root_dir,
			img_name
		)

		img = Image.open(img_path).convert("RGB")
		img:torch.Tensor = self.transform(img)

		label = self.df.iloc[index]["label"]
		label_idx = 0 if label == 'l' else 1

		sample = Sample(
			name=img_name,
			path=img_path,
			image=img,
			size=img.size(),
			label=torch.tensor(label_idx, dtype=torch.long)
		)

		# sample = img, label_idx

		return sample
	

def build_dataset(
		csv_path:str,
		root_dir:str,
		seed:int,
		test_size:float,
		train_transform=None,
		test_transform=None
	):

	# splits test and train samples and returns two distict datasets:
	# one for train one for test.

	df = pd.read_csv(csv_path)
	train_df, test_df = train_test_split(
				df, 
				test_size=test_size, 
				random_state=seed, 
				stratify=df["label"]  # distributes labels between train and test in balance
	)

	train_dataset = _LRClassifierDataset(
		train_df,
		root_dir=root_dir,
		transform=train_transform
	)

	test_dataset = _LRClassifierDataset(
		test_df,
		root_dir=root_dir,
		transform=test_transform
	)

	return train_dataset, test_dataset
