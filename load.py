import utils
import torch
import torch.nn as nn

import numpy as np
from model import FormulaNet

import os

def Test(num_datapoints):
	tokens_to_index = get_token_dict_from_file()

	err_count = 0
	count = 0

	for datapoint in test_dataset():
		conjecture = datapoint.conjecture
		statement = datapoint.statement
		label = datapoint.label

		conjecture_statement = [conjecture, statement]

		prediction_val = F([conjecture_statement])
		_, prediction_label = torch.max(prediction_val, dim = 1)

		if cuda_available:
			prediction_label = prediction_label.cpu()
		prediction_label = prediction_label.numpy()

		if datapoint.label != prediction_label[0]:
			err_count += 1

		count += 1

		if count % 100 == 0:
			print("Count: ",count)

	print("Fraction of Incorrect Test Points: ", err_count / count)

	return err_count / count


cuda_available = torch.cuda.is_available()

num_steps = int(input("Number of Update Iterations:\n"))
F = FormulaNet(num_steps, cuda_available)

# Load Model
MODEL_DIR = os.path.join("..", "models")
file_path = os.path.join(MODEL_DIR, 'last.pth.tar')
utils.load_checkpoint(F, file_path, cuda_available)

print("Model Loaded!")

# Evaluate
F.eval()
err_fract = Test(5000)

print("Test Error: ", err_fract)
