System Requirements:
	Python3
	Compatible using Cuda or without.

Project Name: Premise Selection for Automated Theorem Proving


Members: 
	Seo Taek Kong
	Joseph Lubars*
	Harsh Gupta
	Siddhartha Satpathi 

	* was initially registered in the course and contributed significantly to the project, but is not enrolled in the course anymore.


Project Description:
	This is an implementation of FormulaNet, a graph-embedding method combined with a neural network architecture used to classify which statements are relevant in proving a conjecture. 



File Descriptions: (Notice utils.py is copied from github, which is used to store/load model)
	1. dataset.py
		Run this script to process the HolStep into graph-embedded objects used for FormulaNet. Training and Testing the model assumes that dataset.py is executed.

		This file handles the low-level I/O of the preprocessed data including reading the files in a random order, shuffling the data offline, and processing the raw data. 
		In a high level view, we just need the functions train_dataset(), validation_dataset(), and test_datset() which returns the (conjecture, statement, label) pairs using a generator object. This assumes that given <data_directory> explicitly chosen, the datasets are in <data_directory>/train, <data_directory>/validation, <data_directory>/test directories respectively.

	2. holstep.py
		Defines the preprocessed data objects <Graph> and <Node>. Implicit in these classes are visualization methods of the graphs and other debugging low_level functions. This is never explicitly necessary once the preprocessing data is available, other than the fact that G.nodes is a dictionary in the form {node_id: node_object}

	3. model.py
		This defines all the necessary Neural Network Architectures such as the linear mapping from one-hot nodes to 256-dimensional dense nodes, FI, ... FR, max-pool, classifier, and FormulaNet which combines all of these.
		Currently all the treelet related parts are commented out. Thus, default is FormulaNet-Basic. To use FormulaNet, the commented parts can just be uncommented. 

	4. train.py
		To train a new / saved model, this script may be called. When training a new model, --num_steps specifies the number of equation 1 updates you want the FormulaNet to run. Default is 0. Enter python3 train.py --help for more details.

	5. load.py
		This script may be run to test a saved model, Assuming the saved model is in ../models directory saved in file last.pth.tar. After running through the entire test dataset, it prints the test error percentage.

	6. !utils.py! -- copied from github.
		This was copied from a github repository used for Stanford CS 221 course. This is used to store / load a pytorch model / optimizer. The file was slightly modified so that the current epoch # is also saved.

	7. treelet.py
		Given a graph, the function returns all treelets where xv is a left-treelet, head-treelet, or right-treelet.


===========================================================
INSTRUCTIONS ON RUNNING THE CODE
===========================================================

How to Pre-Process Dataset (required before training model):
	python3 dataset.py <path_to_holstep_dataset>

How to Train a New Model:
	python3 train.py --num_steps <number of update steps>
	It is assumed that the processed data is in ../data

Training a Saved Model:
	python3 train.py --load True --epochs <Final epoch number> --start_epoch <Last saved epoch number from 0 to <Final epoch number>> --start_batch <Last saved batch number within the last --start_epoch> --model_path <Absolute path to model file (.pth.tar file)

How to Test given a saved model:
	python3 load.py
	After executing above, the user will be prompted to enter the number of steps (integer val, >= 0) the model uses.

	This assumes that the model is saved in ../models/last.pth.tar file which is the default save path for train.py
	Supports CPU and GPU regardless of where the model was trained on.

	We have also attached a saved model (1 number of steps, trained on 14,000 batches, batch-size 32.)
	If you would like to test this model, run load.py setting num_steps 1 when prompted