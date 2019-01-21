"""
Declare the neural network models LinearMap (1909 -> 256), FI / FO / FL ... / FR, max-pool, classifier, FormulaNet
"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from treelet import treelet_funct
from dataset import train_dataset, get_token_dict_from_file
import json


class LinearMap(nn.Module):
    """
    Map the one-hot vector to the dense vectors used as inputs to FI, ... FR neural networks.
    """
    def __init__(self, INPUT_DIM = 1909):
        super(LinearMap, self).__init__()
        HL1 = 256
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

    def forward(self, x):
        return F.relu(self.bn1(self.fc1(x)))



class FPClass(nn.Module):
    '''
    FP is the outer function in Section 3.3 used for Order-Preserving Embeddings 
    Input:  Linear combination of the node and neighboring update functions.
    Output: The next value for the node.
    '''
    def __init__(self):
        super(FPClass, self).__init__()
        INPUT_DIM = 256
        HL1 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return x

class FIClass(nn.Module):
    def __init__(self):
        super(FIClass, self).__init__()
        INPUT_DIM = 256 * 2
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)

    def forward(self, x_batch):
        """
        @ Args:
            x_batch (dense vectors, shape = [batch_size, length of parents(xv), xv for each xv)
                Collection of (x_u, x_v) 

                x_batch[:,:, xv_id] = The summands of FI for fixed xv

        @ Output:
            in_sum (array of dense_vectors): Size = Number of Nodes in G
        """
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch



class FHClass(nn.Module):
    """ Nueral Network Architecture for Treelet Functions FL, FH, FR
    """
    def __init__(self):
        super(FHClass, self).__init__()
        INPUT_DIM = 256 * 3
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)

    def forward(self, x_batch):
        """
        @ (Forward) Args:
            x_batch (2D Tensor)
                Batches of (xv, xu, xw), (xu, xv, xw), or (xw, xu, xv) depending on which purpose it is serving for.
        """
        # Order in xv, xu, xw
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch




class CondClassifier(nn.Module):
    """
    Architecutre used for the Conditional Classifier, i.e. network taking max-pooled (conjecture, statement) pair as input.
    """
    def __init__(self):
        super(CondClassifier, self).__init__()
        INPUT_DIM = 256 * 2
        HL1 = 256
        HL2 = 2
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

        self.fc2 = nn.Linear(HL1, HL2)

    def forward(self, x_conj, x_state):
        x = torch.cat([x_conj, x_state], dim = 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return x


class max_pool_dense_graph_var_inputs():
    def __init__(self):
        pass

    def __call__(self, G_dense):   
        """
        @ Args:
            G_dense (2D Tensor): 0-axis corresponds to indices of nodes in a single graph.

        @ Returns (1D Tensor, length 256):
            256-dim vector which is the graph max-pooled over all nodes.
        """
        maxed, _ = torch.max(G_dense, dim = 0)
        return maxed




class FormulaNet(nn.Module):
    def __init__(self, num_steps, cuda_available = False):
        super(FormulaNet, self).__init__()
        # Initialize models
        self.token_to_index = get_token_dict_from_file()
        self.num_tokens = len(self.token_to_index)

        self.dense_map = LinearMap(self.num_tokens) # maps one_hot -> 256 dimension vector
        self.FP = FPClass()
        self.FI = FIClass()
        self.FO = FIClass()
        # self.FL = FHClass()
        # self.FR = FHClass()
        # self.FH = FHClass()
        self.Classifier = CondClassifier()

        self.max_pool_dense_graph = max_pool_dense_graph_var_inputs()

        self.num_steps = num_steps
        self.cuda_available = cuda_available


    # Given graphs and all the functions, do one update in parallel for all nodes in the graph.
    def fullPass(self, dense_nodes, Gs):
        """
        @ Args:
            dense_nodes (2D Tensor): shape = (sum_i length(Gs[i].nodes), 256)
                Batch output of LinearMap

            Gs (Array of <Graph> objects): 
                This is necessary because the relationship between nodes in Gs are used to input into FI, FO, FL, FH, FR

        @ Vars:
            in_batch (list): Inter-intra graph batch of dense_nodes to be fed into FI, i.e. (xu, xv) forall xu, xv
                out_batch, head_batch, ... , right_batch are all inputs respectively to FO, .., FR, e.g. out_batch = (xv, xu) forall xv, xu
            in_indices (dict): indices of in_batch corresponding to each (xu, xv) pair, forall xu
                e.g. 
                    in_batch = [[0, 1], [0, 2], [1, 2]] => in_indices[0] = [1,2], in_indices[1] = [2]
            dv (1D Tensor): dv[xv] = in_degree(xv) + out_degree(xv)
            ev (1D Tensor): ev[xv] = number of treelets in which node xv is included.

        @ Return:
            new_nodes (2D Tensor, shape shape as dense_nodes>): One-step update of <dense_nodes> (output of equation 1 or 2)
        """
        # dv is determined by the number of summands for FI + summands of FO
        if self.cuda_available:
            dv = torch.zeros([dense_nodes.shape[0]]).cuda()
            # ev = torch.zeros([dense_nodes.shape[0]]).cuda()
        else:
            dv = torch.zeros(dense_nodes.shape[0])
            # ev = torch.zeros([dense_nodes.shape[0]])

        start_index = 0 # To keep track of which graph's xv_id and xu_id we are using.

        in_index = 0
        in_indices = {} # for each x in Gs, in_indices[x] gives the indices of in_batch related for summing
        in_batch = []

        out_index = 0
        out_batch = []
        out_indices = {}

        # left_index = 0
        # left_batch = []
        # left_indices = {}

        # head_index = 0
        # head_batch = []
        # head_indices = {}

        # right_index = 0
        # right_batch = []
        # right_indices = {}

        in_index_begin = 0 # NEW
        out_index_begin = 0

        for G in Gs: # Inter-graph Batching.
            end_index = start_index + len(G.nodes)

            # treelets = treelet_funct(G) # Treelets for this graph.
            for xv_id, xv_obj in G.nodes.items():
                xv_id_offset = xv_id + start_index
                xv_dense = dense_nodes[xv_id_offset]

                if len(xv_obj.parents) > 0:
                    xv_obj.parents = np.array(xv_obj.parents)
                    in_indices[xv_id_offset] = np.arange(len(xv_obj.parents)) + in_index_begin
                    dv[xv_id_offset] = len(xv_obj.parents)

                    xu_dense_collect = dense_nodes[xv_obj.parents + start_index]
                    xv_dense_collect = xv_dense.unsqueeze(0).repeat(len(xv_obj.parents), 1)
                    in_batch.extend(torch.cat([xu_dense_collect, xv_dense_collect], dim = 1))
                    in_index_begin = in_index_begin + len(xv_obj.parents)
                else:
                    in_indices[xv_id_offset] = np.array([])
                    dv[xv_id_offset] = 0


                if len(xv_obj.children) > 0:
                    xv_obj.children = np.array(xv_obj.children)
                    out_indices[xv_id_offset] = np.arange(len(xv_obj.children)) + out_index_begin
                    dv[xv_id_offset] += len(xv_obj.children)

                    xu_dense_collect = dense_nodes[xv_obj.children + start_index]
                    xv_dense_collect = xv_dense.unsqueeze(0).repeat(len(xv_obj.children), 1)
                    out_batch.extend(torch.cat([xv_dense_collect, xu_dense_collect], dim = 1))
                    out_index_begin = out_index_begin + len(xv_obj.children)
                else:
                    out_indices[xv_id_offset] = np.array([])
                    dv[xv_id_offset] += 0



                # # ----------------------- Iterate over treelets ----------------------- #
                # # Left Treelet: (xv, xu, xw)
                # left_indices[xv_id_offset] = []
                # for _, xu_id, xw_id in treelets[xv_id][0]:
                #     xu_id_offset = xu_id + start_index
                #     xw_id_offset = xw_id + start_index

                #     left_indices[xv_id_offset].append(left_index)
                #     left_index += 1

                #     xu_dense = dense_nodes[xu_id_offset]
                #     xw_dense = dense_nodes[xw_id_offset]
                #     left_batch.append(torch.cat([xv_dense, xu_dense, xw_dense], dim = 0))
                #     ev[xv_id_offset] += 1

                # # Head Treelet: (xu, xv, xw)
                # head_indices[xv_id_offset] = []
                # for xu_id, _, xw_id in treelets[xv_id][1]:
                #     xu_id_offset = xu_id + start_index
                #     xw_id_offset = xw_id + start_index

                #     head_indices[xv_id_offset].append(head_index)
                #     head_index += 1

                #     xu_dense = dense_nodes[xu_id_offset]
                #     xw_dense = dense_nodes[xw_id_offset]
                #     head_batch.append(torch.cat([xu_dense, xv_dense, xw_dense]))
                #     ev[xv_id_offset] += 1

                # # Right Treelet: (xu, xw, xv)
                # right_indices[xv_id_offset] = []
                # for xu_id, xw_id, _ in treelets[xv_id][2]:
                #     xu_id_offset = xu_id + start_index
                #     xw_id_offset = xw_id + start_index

                #     right_indices[xv_id_offset].append(right_index)
                #     right_index += 1

                #     xu_dense = dense_nodes[xu_id_offset]
                #     xw_dense = dense_nodes[xw_id_offset]
                #     right_batch.append(torch.cat([xu_dense, xw_dense, xv_dense]))
                #     ev[xv_id_offset] += 1

            start_index += len(G.nodes)


        # Pass in_batch into FI
        if len(in_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            if self.cuda_available:
                in_sum = torch.zeros(dense_nodes.shape).cuda()
            else:
                in_sum = torch.zeros(dense_nodes.shape)
        else:
            in_batch = torch.stack(in_batch, dim = 0)
            in_sum = self.FI(in_batch) 

        # Pass out_batch into FO
        if len(out_batch) <= 1: 
            if self.cuda_available:
                out_sum = torch.zeros(dense_nodes.shape).cuda()
            else:
                out_sum = torch.zeros(dense_nodes.shape)
        else:
            out_batch = torch.stack(out_batch, dim = 0)
            out_sum = self.FO(out_batch)

        # # Left Treelet
        # if len(left_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
        #     # print(3)
        #     if self.cuda_available:
        #         left_sum = torch.zeros(dense_nodes.shape).cuda()
        #     else:
        #         left_sum = torch.zeros(dense_nodes.shape)
        # else:
        #     left_batch = torch.stack(left_batch, dim = 0)
        #     left_sum = self.FL(left_batch)


        # # Head Treelet
        # if len(head_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
        #     # print(4)
        #     if self.cuda_available:
        #         head_sum = torch.zeros(dense_nodes.shape).cuda()
        #     else:
        #         head_sum = torch.zeros(dense_nodes.shape)
        # else:
        #     head_batch = torch.stack(head_batch, dim = 0)
        #     head_sum = self.FH(head_batch)


        # # Right Treelet
        # if len(right_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
        #     # print(5)
        #     if self.cuda_available:
        #         right_sum = torch.zeros(dense_nodes.shape).cuda()
        #     else:
        #         right_sum = torch.zeros(dense_nodes.shape)
        # else:
        #     right_batch = torch.stack(right_batch, dim = 0)
        #     right_sum = self.FR(right_batch)



        # Compute in_out_sum and treelet_sum to feed into FP
        if self.cuda_available:
            in_out_sum = torch.zeros(dense_nodes.shape).cuda() 
            # treelet_sum = torch.zeros(dense_nodes.shape).cuda()
        else:
            in_out_sum = torch.zeros(dense_nodes.shape)
            # treelet_sum = torch.zeros(dense_nodes.shape)


        # Gather inputs to pass into FP
        for xv_id_offset in range(len(dense_nodes)):
            if dv[xv_id_offset] > 0:
                new_in_sum = torch.sum(in_sum[in_indices[xv_id_offset], :], dim = 0)
                new_out_sum = torch.sum(out_sum[out_indices[xv_id_offset], :], dim = 0)
                in_out_sum[xv_id_offset] = (new_in_sum + new_out_sum) / dv[xv_id_offset]

            # if ev[xv_id_offset] > 0:
            #     new_left_sum = torch.sum(left_sum[left_indices[xv_id_offset], :], dim = 0)
            #     new_head_sum = torch.sum(head_sum[head_indices[xv_id_offset], :], dim = 0)
            #     new_right_sum = torch.sum(right_sum[right_indices[xv_id_offset], :], dim = 0)
            #     treelet_sum[xv_id_offset] = torch.zeros(256).cuda()

        # Add and then send to FP to update all the nodes!
        # new_nodes = self.FP(dense_nodes + in_out_sum + treelet_sum)
        new_nodes = self.FP(dense_nodes + in_out_sum)

        return new_nodes

    def graph_to_one_hot(self, G):
        """
        Given a graph object, return an array of one-hot vectors.
        """
        NUM_TOKENS = self.num_tokens
        if self.cuda_available:
            one_hot_graph = torch.zeros((len(G.nodes), NUM_TOKENS)).cuda()
        else:
            one_hot_graph = torch.zeros((len(G.nodes), NUM_TOKENS))

        for node_id, node_obj in G.nodes.items():
            token = node_obj.token
            if token not in self.token_to_index:
                token = "UNKNOWN"
            token_index = self.token_to_index[token]

            one_hot_graph[node_id, token_index] = 1
        return one_hot_graph

    def forward(self, conjecture_state_graphs):
        """ 
        A forward pass of FormulaNet (-Basic).
        
        1. Convert (conjecture, statement) pairs into one-hot representations.
        2. Apply dense_map to map the one-hot nodes (1908 dim) into dense_nodes (256 dim)
        3. Apply fullPass <num_steps> amount of iterations.
        4. max-pool the output of (3)
        5. Pass the max-pooled output of (conjecture, statement) pairs into a classifier.
        6. Return the label scores (NOT PROBABILITIES)

        @ Args:
            conjecture_state_graphs (2D array-like Graph objects, shape = (intra_graph_batch_size, 2)):
                conjecture_state_graphs[i] = conjecture_graph[i], statement_graph[i]

        @ Return:
            Label Scores for the conjecture-statement batch inputs
        """
        inter_graph_conj_state_node_batch = []
        conj_state_graphs = []

        start_index = 0
        conj_indices = {} # conj_index[i] = [first node id in conjecture i, last node id in conjecture i]
        state_indices = {} # similar
        for i in range(len(conjecture_state_graphs)): # Iterate over inter-graph-batch.
            conjecture_graph = conjecture_state_graphs[i][0]
            end_index = start_index + len(conjecture_graph.nodes)
            conj_indices[i] = [start_index, end_index]

            start_index = end_index
            statement_graph = conjecture_state_graphs[i][1]
            end_index = start_index + len(statement_graph.nodes)
            state_indices[i] = [start_index, end_index]

            start_index = end_index

            # Map graph object to an array of one hot vectors
            conj_one_hot = self.graph_to_one_hot(conjecture_graph)
            state_one_hot = self.graph_to_one_hot(statement_graph)

            conj_state_node_batch = torch.cat([conj_one_hot, state_one_hot], dim = 0)
            inter_graph_conj_state_node_batch.append(conj_state_node_batch)

            conj_state_graphs.append(conjecture_graph)
            conj_state_graphs.append(statement_graph)

        # print("Conj Indices: ",conj_indices)
        # print("State Indices: ", state_indices)

        inter_graph_conj_state_node_batch = torch.cat(inter_graph_conj_state_node_batch, dim = 0) # [:, 1909] Tensor (as if all nodes belonged to one huge graph)
        conj_state_dense_batch = self.dense_map(inter_graph_conj_state_node_batch)

        # Iterate equation 2.
        for t in range(self.num_steps):
            conj_state_dense_batch = self.fullPass(conj_state_dense_batch, conj_state_graphs)
            
        # Finished Updating. max-pool over all nodes in the graph

        # -------------------------- This will have to be modified ---------------------------- #
        # max_pool across each relevant graph. For example, the first max-pool should be over the first 36 nodes.
        conj_embeddings = []
        state_embeddings = []
        for i in range(len(conj_indices)):
            conj_embeddings.append(self.max_pool_dense_graph(conj_state_dense_batch[conj_indices[i][0] : conj_indices[i][1]]))
            state_embeddings.append(self.max_pool_dense_graph(conj_state_dense_batch[state_indices[i][0] : state_indices[i][1]]))

        conj_embeddings = torch.stack(conj_embeddings)
        state_embeddings = torch.stack(state_embeddings)

        # Classify
        # -------------------------- This will have to be modified ---------------------------- #
        # Classify across each conjecture-state embeddings. (for example, each should take only 2 indices each.)
        prediction = self.Classifier(conj_embeddings, state_embeddings)

        return prediction



# if __name__ == "__main__":
    # graph_to_index_offline()
