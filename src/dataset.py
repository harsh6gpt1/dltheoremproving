import os
import sys
import random
import pickle
from holstep import *
import numpy as np

import argparse

data_directory = os.path.join("..", "data")

test_directory = os.path.join(data_directory, "test")
train_directory = os.path.join(data_directory, "train")
validation_directory = os.path.join(data_directory, "validation")
token_file = os.path.join(data_directory, "tokens.pickle")
chunk_info_file = os.path.join(data_directory, "chunks.txt")
conjecture_filename = "conjectures"
raw_data_directory = "holstep"

num_training_chunks = 200
num_testing_chunks = 20
num_holdout_chunks = 20
holdout_fraction = 0.07
buffer_flush_interval = 100
reporting_interval = 10

class Datapoint:

    def __init__(self, conjecture, statement, label):
        self.conjecture = conjecture
        self.statement = statement
        self.label = label

    def to_file(self):
        return "{}\n{}\n{}\n".format(self.conjecture.to_json(), self.statement.to_json(), self.label)

    def __repr__(self):
        return self.to_file()


def datapoint_from_file(line1, line2, line3):
    conjecture = graph_from_json(line1.strip('\n'))
    statement = graph_from_json(line2.strip('\n'))
    label = int(line3.strip('\n'))
    return Datapoint(conjecture, statement, label)


def create_directories():
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(validation_directory):
        os.makedirs(validation_directory)


def flush_buffer(filename, datapoint_list, append=True):
    if append:
        with open(filename, 'a+') as datachunk:
            while len(datapoint_list) > 0:
                datachunk.write(datapoint_list.pop().to_file())
    else:
        with open(filename, 'w') as datachunk:
            for item in datapoint_list:
                datachunk.write(datapoint_list.pop().to_file())


def flush_buffers(buffer_list, directory):
    for index, datapoint_buffer in enumerate(buffer_list):
        flush_buffer(os.path.join(directory, str(index)), datapoint_buffer)


def process_raw_dataset(raw_dataset_directory):
    process_raw_training_dataset(raw_dataset_directory)
    process_raw_testing_dataset(raw_dataset_directory)


def process_raw_training_dataset(raw_dataset_directory):
    train_tokens = set([])
    raw_train_directory = os.path.join(raw_dataset_directory, "train")
    training_filenames = os.listdir(raw_train_directory)
    training_buffers = [list() for i in range(num_training_chunks)]
    training_sizes = {i:0 for i in range(num_training_chunks)}
    holdout_buffers = [list() for i in range(num_holdout_chunks)]
    holdout_sizes = {i:0 for i in range(num_holdout_chunks)}
    for index, filename in enumerate(training_filenames, 1):
        if index % 10 == 0:
            print("Processing training file {}...".format(index))
        if index % buffer_flush_interval == 0:
            flush_buffers(training_buffers, train_directory)
            flush_buffers(holdout_buffers, validation_directory)
        conjecture_graph, statements = parse_holstep_file(os.path.join(raw_train_directory, filename))
        for token in conjecture_graph.find_all_tokens():
            train_tokens.add(token)
        if random.random() > holdout_fraction:
            for statement in statements:
                for token in statement[0].find_all_tokens():
                    train_tokens.add(token)
                file_index = random.randrange(num_training_chunks)
                training_buffers[file_index].append(Datapoint(conjecture_graph, statement[0], statement[1]))
                training_sizes[file_index] += 1
        else:
            for statement in statements:
                file_index = random.randrange(num_holdout_chunks)
                holdout_buffers[file_index].append(Datapoint(conjecture_graph, statement[0], statement[1]))
                holdout_sizes[file_index] += 1
    flush_buffers(training_buffers, train_directory)
    flush_buffers(holdout_buffers, validation_directory)
    token_dict = {'UNKNOWN': 2}
    index = 3
    for token in train_tokens:
        if token == 'VAR':
            token_dict[token] = 0
        elif token == 'VARFUNC':
            token_dict[token] = 1
        else:
            token_dict[token] = index
            index += 1
    with open(token_file, 'wb') as token_pickle:
        pickle.dump(token_dict, token_pickle)


def process_raw_testing_dataset(raw_dataset_directory):
    raw_test_directory = os.path.join(raw_dataset_directory, "test")
    test_filenames = os.listdir(raw_test_directory)
    test_buffers = [list() for i in range(num_testing_chunks)]
    test_sizes = {i:0 for i in range(num_testing_chunks)}
    for index, filename in enumerate(test_filenames, 1):
        if index % reporting_interval == 0:
            print("Processing testing file {}...".format(index))
        if index % buffer_flush_interval == 0:
            flush_buffers(test_buffers, test_directory)
        conjecture_graph, statements = parse_holstep_file(os.path.join(raw_test_directory, filename))
        for statement in statements:
            file_index = random.randrange(num_testing_chunks)
            test_buffers[file_index].append(Datapoint(conjecture_graph, statement[0], statement[1]))
            test_sizes[file_index] += 1
    flush_buffers(test_buffers, test_directory)
        

def get_token_dict_from_file():
    with open(token_file, 'rb') as token_pickle:
        return pickle.load(token_pickle)


def shuffle_dataset(dataset_directory, num_files):
    current_file_num = 0
    while current_file_num < num_files:
        with open(os.path.join(dataset_directory, str(current_file_num)), 'r') as current_file:
            print("Shuffling file {}".format(current_file_num))
            datapoints = []
            while True:
                line1 = current_file.readline()
                line2 = current_file.readline()
                line3 = current_file.readline()
                if len(line1) <= 1:
                    break
                datapoint = datapoint_from_file(line1, line2, line3)
                datapoints.append(datapoint)
            random.shuffle(datapoints)
            flush_buffer(os.path.join(dataset_directory, str(current_file_num)), datapoints, append=False)
            current_file_num += 1


def get_dataset_from_directory(dataset_directory, num_files):
    datapoint_num = 0
    file_nums = np.arange(num_files)
    np.random.shuffle(file_nums) # Shuffles in-place

    # current_file_num = 0
    # while current_file_num < num_files:
    for current_file_num in file_nums:
        with open(os.path.join(dataset_directory, str(current_file_num)), 'r') as current_file:
            while True:
                line1 = current_file.readline()
                line2 = current_file.readline()
                line3 = current_file.readline()
                if len(line1) <= 1:
                    break
                datapoint = datapoint_from_file(line1, line2, line3)
                yield datapoint
                datapoint_num += 1
            current_file_num += 1


def train_dataset():
    train_generator = get_dataset_from_directory(train_directory, num_training_chunks)
    yield from train_generator


def validation_dataset():
    validation_generator = get_dataset_from_directory(validation_directory, num_holdout_chunks)
    yield from validation_generator


def test_dataset():
    test_generator = get_dataset_from_directory(test_directory, num_testing_chunks)
    yield from test_generator


def shuffle_all_datasets():
    shuffle_dataset(train_directory, num_training_chunks)
    shuffle_dataset(validation_directory, num_holdout_chunks)
    shuffle_dataset(test_directory, num_testing_chunks)


if __name__ == "__main__":
    create_directories()
    if len(sys.argv) > 1:
        process_raw_dataset(sys.argv[1])
    else:
        process_raw_dataset(raw_data_directory)
        shuffle_all_datasets()
