"""
Created on Tue Nov 13 16:59:22 2018
it takes as input a graph G and outputs all treelets of the graph
The output is a dictionary with key as the unique id of the node and list of three lists.
Suppose the node is v
list1 - all left treelets (v,x,y)
list2 - all head treelets (x,v,y)
list3 - all right treelets (x,y,v)
treelet[v] = [list1,list2,list3]  
@author: ssatpth2
"""

import itertools

def treelet_funct(G):
    treelet_dict = {}
    for key, value in G.nodes.items():
        ltreelet = []
        ptreelet = []
        rtreelet = []
        for l,r in itertools.combinations(value.children,2):
            ptreelet.append((l,value.uid,r))
        for p in value.parents:
            pnode = G.nodes[p]
            ind = pnode.children.index(value.uid)
            for i in range(len(pnode.children)):
                c = pnode.children[i]
                if i < ind:
                    rtreelet.append((c,p,value.uid))
                if i > ind:
                    ltreelet.append((value.uid,p,c))
        treelet_dict[value.uid] = [ltreelet,ptreelet,rtreelet]
    return treelet_dict
