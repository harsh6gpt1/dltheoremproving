import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import FormulaNet
import utils

from dataset import train_dataset, test_dataset, validation_dataset, get_token_dict_from_file


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch Size')
parser.add_argument('--epochs', type = int, default = 5, help = 'Number of training episodes')
parser.add_argument('--num_steps', type = int, default = 0, help = 'Number of update steps for equation 1 or 2')
parser.add_argument('--lr', type = float, default = 1e-3, help = 'Initial learning rate for RMSProp')
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = "Weight decay parameter for RMSProp")
parser.add_argument('--lr_decay', type = float, default = 3., help = 'Multiplicative Factor by which to decay learning rate by after each epoch, > 1')
parser.add_argument('--start_epoch', type = int, default = 0, help = 'Epoch to resume with')
parser.add_argument('--start_batch', type = int, default = 0, help = 'Batch Number to resume with')
parser.add_argument('--load', type = bool, default = False, help = 'True to load model')
parser.add_argument('--model_path', type = bool, default = None, help = '.pth.tar file to save the model and optimizer. Default is ../models/last.pth.tar')

args = parser.parse_args()
print(args)



def Validate(num_datapoints):
	# To validate the current model after each epoch.
	tokens_to_index = get_token_dict_from_file()

	err_count = 0
	count = 0
	for datapoint in validation_dataset():
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

	print("Fraction of Incorrect Validations: ", err_count / count)

	return err_count / count


def label_to_one_hot(y):
	# Given index 0 or 1, map into one_hot vector
	x = np.zeros(2)
	x[y] = 1
	return x




MODEL_DIR = os.path.join("..", "models")
# input("Is the saving directory okay, and are you loading the model if necessary?")

cuda_available = torch.cuda.is_available()

# Define Model. Decide whether to load (first case) or start new (else)
F = FormulaNet(args.num_steps, cuda_available)
opt = torch.optim.RMSprop(F.parameters(), lr = args.lr, alpha = args.weight_decay)


if args.load is True:
	file_path = os.path.join(MODEL_DIR, 'last.pth.tar')
	if args.model_path:
		file_path = args.model_path
	utils.load_checkpoint(F, file_path, cuda_available, opt)

else: # When not loading, make sure cuda is enabled.
	if cuda_available: 
		F.cuda()
		print("Cuda Enabled!")


F.train()




# Loss Function
loss = nn.BCEWithLogitsLoss() # Binary Cross-Entropy Loss with sigmoid attached in front.


""" 
==============================================
Begin Training
==============================================
"""
lr = args.lr
for epoch in range(args.start_epoch, args.epochs):
	# I will assume the graphs are shuffled somehow when calling train_dataset().

	batch_index = 0 # More like batch counter, counting the number of distinct (conjecutre, statement) pairs are stored.
	batch_number = 0 # Number of batches iterated.
	conjecture_state_batch = []
	label_batch = []

	for datapoint in train_dataset(): 
		# Collect datapoints for inter-graph batching.

		if (epoch == args.start_epoch) and (batch_number < args.start_batch - 1): 
			# If starting from a saved <start_epoch> and <start_batch>, pass through prior datapoints.
			batch_index += 1
			if batch_index < args.batch_size:
				continue
			else:
				batch_index = 0
				batch_number += 1
			continue

		conjecture_graph = datapoint.conjecture
		statement_graph = datapoint.statement
		label = label_to_one_hot(datapoint.label)

		conjecture_state_batch.append([conjecture_graph, statement_graph])
		label_batch.append(label)

		if batch_index < args.batch_size - 1: # Keep collecting (conjecture, statement) pairs.
			batch_index += 1
			continue

		else:
			# Start Training! 

			opt.zero_grad() # Delete stored gradients before doing forward-backward

			# Forward
			predict_val = F(conjecture_state_batch)

			if cuda_available:
				label_batch_tensor = torch.Tensor(label_batch).cuda()
			else:
				label_batch_tensor = torch.Tensor(label_batch)

			# Compute loss due to prediction. How to make label_scores just a scalar? argmax?
			curr_loss = loss(predict_val, label_batch_tensor)

			# Backpropogation.
			curr_loss.backward()
			opt.step()

			batch_number += 1

			# Reset batch.
			batch_index = 0
			conjecture_state_batch = []
			label_batch = []

			if (batch_number > 0) and (batch_number % 50 == 0):
				print("Trained %d Batches" %batch_number)
				print("Train Loss", curr_loss)
		
			if (batch_number > 0) and (batch_number % 1000 == 0):
				# Save model, optimizer, and epoch # every 1000 batches.
				state_dict = {'epoch:': epoch + 1, 
							  'state_dict': F.state_dict(), 
							  'optim_dict': opt.state_dict()}
				utils.save_checkpoint(state_dict, checkpoint = MODEL_DIR)
				print("Models and Optimizers Saved.")


	# --------------- End of Epoch --------------- #

	# Costruct new optimizers after each epoch.
	lr = lr / args.lr_decay
	opt = torch.optim.RMSprop(F.parameters(), lr = lr, alpha = args.weight_decay)

	print("----------------------------------------------------")
	print("----------------------------------------------------")
	print("Epoch # "+str(epoch + 1)+" done.")
	print("----------------------------------------------------")
	print("----------------------------------------------------")
	

	# Save Model After Each Epoch
	state_dict = {'epoch:': epoch + 1, 'state_dict': F.state_dict(), 'optim_dict': opt.state_dict()}
	utils.save_checkpoint(state_dict, checkpoint = MODEL_DIR)

	print("Models and Optimizers Saved.")



	# Validate Model after Each Epoch
	F.eval()
	val_err = Validate(0)

	F.train()

