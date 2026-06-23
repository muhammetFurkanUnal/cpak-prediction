from typing import TypedDict, Tuple
import torch


class ModelCheckpoint(TypedDict):
	epoch: int
	model_state: any
	optimizer_state: any
	epoch_loss: float


class Sample(TypedDict):
	name:str
	path:str
	image:torch.Tensor
	size:Tuple[int, int]
	label:int