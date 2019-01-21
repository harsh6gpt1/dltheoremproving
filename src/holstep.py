import re
import networkx as nx
import os
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from enum import Enum
import json

test_1 = "|- (!t1. (!t2. (((~ (t1 /\ t2)) = ((~ t1) \/ (~ t2))) /\ ((~ (t1 \/ t2)) = ((~ t1) /\ (~ t2))))))"
test_2 = "c/\ c= c~ c/\ f0 f1 c\/ c~ f0 c~ f1 c= c~ c\/ f0 f1 c/\ c~ f0 c~ f1"

test_3 = "|- (!y. (!x. (((real_lt x) y) = (~ ((real_le y) x)))))"
test_4 = r"|- (((\x'. ((\v. ((i < k) /\ (x = v))) x')) ((EL i) ((truncate_simplex (k - (NUMERAL (BIT1 _0)))) vl))) = ((\v. ((i < k) /\ (x = v))) ((EL i) ((truncate_simplex (k - (NUMERAL (BIT1 _0)))) vl))))"
test_5 = r"(~ (i = k)), (x = ((EL i) vl)), ((LENGTH vl) = (k + (NUMERAL (BIT1 _0)))), (i < (LENGTH vl)), ((NUMERAL _0) < k) |- (((((((k - (NUMERAL (BIT1 _0))) + (NUMERAL (BIT1 _0))) <= (LENGTH vl)) /\ (i <= (k - (NUMERAL (BIT1 _0))))) /\ ((\x'. ((\v. ((i < k) /\ (x = v))) x')) ((EL i) vl))) ==> ((\x'. ((\v. ((i < k) /\ (x = v))) x')) ((EL i) ((truncate_simplex (k - (NUMERAL (BIT1 _0)))) vl)))) = ((\x'. ((\v. ((i < k) /\ (x = v))) x')) ((EL i) ((truncate_simplex (k - (NUMERAL (BIT1 _0)))) vl))))"

TOKEN_REGEX = re.compile(r"[(,)]|[^\s(,)]+")
QUANTIFIER_REGEX = re.compile(r"^([!?\\@]|\?!|lambda)([a-zA-Z0-9_%'<>]+)\.$")
VARNAME_REGEX = re.compile(r"^(GEN%PVAR%\d+|_\d+|[a-np-zA-Z])$")

OPERATORS = {
    "=", "/\\", "==>", "\\/", "o", ",", "+", "*", "EXP", "<=", "<", ">=", ">", "-",
    "DIV", "MOD", "treal_add", "treal_mul", "treal_le", "treal_eq", "/", "|-", "pow",
    "div", "rem", "==", "divides", "IN", "INSERT", "UNION", "INTER", "DIFF", "DELETE",
    "SUBSET", "PSUBSET", "HAS_SIZE", "CROSS", "<=_c", "<_c", "=_c", ">=_c", ">_c", "..",
    "$", "PCROSS", "!", "?", "\\", "@", "?!"
}  # Using the set of operators from FormulaNet, plus the quantifiers

class Graph:
    """
    Graph embedded object of a raw HolStep conjecture or statement, with each node represented as a token.
    """
    def __init__(self):
        self.nodes = {}
        self.root = None
        self.next_id = 0

    def add_node(self, token, quantified=False, child_list=None):
        new_node = Node(self.next_id, token, quantified)
        self.nodes[self.next_id] = new_node
        if child_list is not None:
            for child in child_list:
                self.add_edge(new_node, self.nodes[child])
        self.next_id += 1
        return new_node

    def delete_node(self, node):
        for child in node.children:
            child_node = self.nodes[child]
            child_node.parents.remove(node.uid)
        for parent in node.parents:
            parent_node = self.nodes[parent]
            parent_node.children.remove(node.uid)
        del self.nodes[node.uid]

    def get_node(self, uid):
        return self.nodes[uid]

    def merge_leaves(self):
        all_tokens = self.find_all_tokens()
        for token in all_tokens:
            if token in OPERATORS:
                continue
            #combined_leaf_node = self.add_node(token, False)
            quantified_nodes = []
            leaf_nodes = []
            remaining_nodes = []
            for node in all_tokens[token]:
                if node.quantified:
                    quantified_nodes.append(node)
                elif node.is_leaf():
                    leaf_nodes.append(node)
                    #self.merge_nodes(node, combined_leaf_node)
                else:
                    remaining_nodes.append(node)
            if len(quantified_nodes) > 0:
                for quantified_node in quantified_nodes:
                    combined_leaf_node = self.add_node(token, False)
                    quantifier = self.nodes[quantified_node.parents[0]]
                    leaves_to_delete = []
                    for node in leaf_nodes:
                        if self.is_descendant(node, quantifier):
                            leaves_to_delete.append(node)
                            self.merge_nodes(node, combined_leaf_node)
                    for node in leaves_to_delete:
                        leaf_nodes.remove(node)
                    self.add_edge(quantifier, combined_leaf_node, True)
                    combined_leaf_node.token = 'VAR'
                    self.delete_node(quantified_node)
                    if len(combined_leaf_node.parents) == 0:
                        self.delete_node(combined_leaf_node)
                    else:
                        leaf_nodes.append(combined_leaf_node)
                    for non_leaf_node in remaining_nodes:
                        if self.is_descendant(non_leaf_node, quantifier):
                            self.add_edge(quantifier, non_leaf_node)
                            non_leaf_node.token = 'VARFUNC'
        self.rename_misc_vars()

    def is_descendant(self, child, parent, depth=0):
        if len(child.parents) == 0:
            return False
        if child.uid == parent.uid:
            return True
        for test_parent in child.parents:
            if self.is_descendant(self.nodes[test_parent], parent, depth+1):
                return True
        return False

    def rename_misc_vars(self):
        all_tokens = self.find_all_tokens()
        for token in all_tokens:
            if re.match(VARNAME_REGEX, token):
                for node in all_tokens[token]:
                    if node.is_leaf():
                        node.token = 'VAR'
                    else:
                        node.token = 'VARFUNC'

    def reorder_nodes(self):
        num_nodes = len(self.nodes)
        node_map = {}
        reverse_map = {}
        cur_index = 1
        for uid in self.nodes:
            if uid != self.root:
                node_map[cur_index] = uid
                reverse_map[uid] = cur_index
                cur_index += 1
            else:
                # Root always gets node 0
                node_map[0] = uid
                reverse_map[uid] = 0
        for node in self.nodes.values():
            new_parents = [reverse_map[parent] for parent in node.parents]
            new_children = [reverse_map[child] for child in node.children]
            node.uid = reverse_map[node.uid]
            node.parents = new_parents
            node.children = new_children
        self.root = 0
        self.nodes = {i: self.nodes[node_map[i]] for i in node_map}
        self.next_id = len(self.nodes) + 1
    
    def rename_non_constants(self, constants):
        all_tokens = self.find_all_tokens()
        for token in all_tokens:
            if token in constants or token in OPERATORS:
                continue
            for node in all_tokens[token]:
                if node.is_leaf():
                    node.token = 'VAR'
                else:
                    node.token = 'VARFUNC'
                
    def add_edge(self, node1, node2, add_to_front=False):
        if node2.uid not in node1.children:
            if add_to_front:
                node1.children.insert(0, node2.uid)
            else:
                node1.children.append(node2.uid)
        if node1.uid not in node2.parents:
            if add_to_front:
                node2.parents.insert(0, node1.uid)
            else:
                node2.parents.append(node1.uid)

    def merge_nodes(self, node1, node2):
        # Merge node1 into node 2, deleting node 1 from the graph
        for child in node1.children:
            if child not in node2.children:
                node2.children.append(child.uid)
        for parent in node1.parents:
            if parent not in node2.parents:
                node2.parents.append(parent)
            parent_node = self.nodes[parent]
            parent_node.children = [x if x != node1.uid else node2.uid for x in parent_node.children]
        del self.nodes[node1.uid]

    def find_all_tokens(self):
        all_tokens = {}
        for node in self.nodes.values():
            if node.token in all_tokens:
                all_tokens[node.token].append(node)
            else:
                all_tokens[node.token] = [node]
        return all_tokens

    def to_networkx(self):
        G = nx.DiGraph()
        for node_id in self.nodes:
            G.add_node(node_id)
        for node in self.nodes.values():
            for child in node.children:
                G.add_edge(node.uid, child)
        return G

    def print_networkx(self, filename):
        G = self.to_networkx()
        pos = graphviz_layout(G, prog='dot')
        labels = {node_id: self.nodes[node_id].token for node_id in self.nodes}
        plt.figure()
        nx.draw(G, pos, with_labels=True, labels=labels, arrows=True)
        plt.savefig(filename)

    def to_json(self):
        self.reorder_nodes()
        obj_format = [{"token": node.token, "parents": node.parents, "children": node.children} for node in self.nodes.values()]
        return json.dumps(obj_format)

    def __repr__(self):
        return_lines = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            return_line = "{} , children: {}, parents: {}".format(
                    node, [self.nodes[x].uid for x in node.children], [self.nodes[x].uid for x in node.parents]
            )
            return_lines.append(return_line)
        return '\n'.join(return_lines)


class Node:

    def __init__(self, uid, token=None, quantified=False):
        self.uid = uid
        self.token = token
        self.children = []
        self.parents = []
        self.quantified = quantified

    def is_leaf(self):
        return len(self.children) == 0

    def is_operator(self):
        return self.token in OPERATORS

    def __repr__(self):
        return "{} ({})".format(self.uid, self.token)


def graph_from_json(json_string):
    obj_format = json.loads(json_string)
    graph = Graph()
    for index, item in enumerate(obj_format):
        graph.add_node(item["token"])
        graph.nodes[index].children = item["children"]
        graph.nodes[index].parents = item["parents"]
    graph.root = 0
    return graph


def split_into_tokens_regex(sentence):
    return re.findall(TOKEN_REGEX, sentence)


def find_constants(token_string):
    tokenization = token_string.split(" ")
    constants = set([])
    for token in tokenization:
        typechar = token[0]
        if typechar == 'c':
            constant = token[1:]
            constants.add(constant)
    return constants


def parse_holstep(holstep_string, token_string):
    tokenization = split_into_tokens_regex(holstep_string)
    constants = find_constants(token_string)

    stack = []
    graph = Graph()
    for token in tokenization:
        if token == ')':
            previous = []
            while len(stack) > 0:
                top_of_stack = stack.pop()
                if top_of_stack == '(':
                    break
                previous.insert(0, top_of_stack)
            stack.append(combine_nodes(previous, graph))
        elif token == '(':
            stack.append('(')
        elif re.match(QUANTIFIER_REGEX, token):
            match = re.match(QUANTIFIER_REGEX, token)
            operator = match.group(1)
            variable = match.group(2)
            if operator == 'lambda':
                operator = '\\'
            stack.append(graph.add_node(variable, True))
            stack.append(graph.add_node(operator, False))
        else:
            stack.append(graph.add_node(token, False))

    root_node = combine_nodes(stack, graph)
    graph.root = root_node.uid
    #graph.print_networkx("before.png")
    #print(in_string)
    #print(graph)
    graph.merge_leaves()
    graph.rename_non_constants(constants)
    #graph.print_networkx("after.png")
    
    return graph


def combine_nodes(node_list, graph):
    if len(node_list) == 2:
        graph.add_edge(node_list[0], node_list[1])
        return node_list[0]
    elif len(node_list) == 3:
        assert node_list[1].is_operator() # Must always be Identifier Operator Identifier
        graph.add_edge(node_list[1], node_list[0])
        graph.add_edge(node_list[1], node_list[2])
        return node_list[1]
    else:
        # If there are more than 3 elements in the expression, then
        # we have assumption1, assumption2, ..., assumptionk |- conclusion
        # Handle this situation here.
        assert node_list[-2].token == '|-'
        for node in node_list:
            if node.token == ',':
                graph.delete_node(node)
            elif node.token == '|-':
                continue
            else:
                graph.add_edge(node_list[-2], node)
        return node_list[-2]


def parse_holstep_file(filename):
    conjecture_name = None
    conjecture_graph = None
    dependencies = {}
    statements = []
    tokens = set([])
    with open(filename, 'r') as hol_file:
        while(True):
            line = hol_file.readline()
            if line == '':
                break
            line_start = line[0]
            if line_start == 'N':
                conjecture_name = line[2:]
                conjecture_graph = parse_holstep(hol_file.readline()[2:], hol_file.readline()[2:])
                for token in conjecture_graph.find_all_tokens():
                    tokens.add(token)
            elif line_start == 'D':
                dependency_name = line[2:]
                dependency_graph = parse_holstep(hol_file.readline()[2:], hol_file.readline()[2:])
                for token in conjecture_graph.find_all_tokens():
                    tokens.add(token)
                dependencies[dependency_name] = dependency_graph
            else:
                statement_graph = parse_holstep(line[2:], hol_file.readline()[2:])
                for token in statement_graph.find_all_tokens():
                    tokens.add(token)
                if line_start == '+':
                    statements.append((statement_graph, 1))
                else:
                    statements.append((statement_graph, 0))

    return conjecture_graph, statements


def parse_holstep_directory(path_to_directory):
    holstep_filenames = os.listdir(path_to_directory)
    all_tokens = set([])
    for index, filename in enumerate(holstep_filenames):
        if index % 100 == 0:
            print("{}/{}".format(index, len(holstep_filenames)))
        tokens = parse_holstep_file(os.path.join(path_to_directory, filename))
        all_tokens = all_tokens | tokens
    print(all_tokens)
    print(len(all_tokens))


if __name__ == "__main__":
    graph = parse_holstep(test_1, test_2)
    print(graph)
    saved = graph.to_json()
    print(saved)
    graph = graph_from_json(saved)
    print(graph)
    #print(parse_holstep_file("data2.txt"))
    #parse_holstep_directory("holstep/train")
