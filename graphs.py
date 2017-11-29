# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import itertools
from networkx.algorithms.tree.branchings import Edmonds
from networkx.algorithms.tree import is_tree
from networkx.algorithms.cycles import find_cycle

#G = nx.DiGraph()
#G.add_node(1)
#G.add_node(2)
#G.add_node(3)
#G.add_node(4)
#G.add_edge(1,2,weight=12)
#G.add_edge(1,3,weight=4)
#G.add_edge(1,4,weight=4)
#G.add_edge(2,3,weight=5)
#G.add_edge(3,2,weight=6)
#G.add_edge(3,4,weight=8)
#G.add_edge(4,3,weight=7)
#G.add_edge(2,4,weight=7)
#G.add_edge(4,2,weight=5)

G = nx.DiGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_edge(1,2,weight=0)
G.add_edge(1,3,weight=15)
G.add_edge(1,4,weight=0)
G.add_edge(1,5,weight=0)
G.add_edge(2,3,weight=5)
G.add_edge(2,4,weight=5)
G.add_edge(2,5,weight=15)
G.add_edge(3,2,weight=20)
G.add_edge(3,4,weight=5)
G.add_edge(3,5,weight=30)
G.add_edge(4,2,weight=10)
G.add_edge(4,3,weight=20)
G.add_edge(4,5,weight=5)
G.add_edge(5,2,weight=5)
G.add_edge(5,3,weight=10)
G.add_edge(5,4,weight=15)



def matrix_to_graph(A, sent, labels, digraph=False):
    k = A.shape[0]
    nodes = range(k)
    if digraph:
        # This function turns a np matrix into a directed graph
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        # Trying to get labels on the arcs... No succes yet
        labels = {(i, j, A[i,j]): np.random.choice(labels) for i,j in itertools.product(nodes, nodes)}
    else:
        G = nx.from_numpy_matrix(A)
        
    for k, w in enumerate(sent):
        G.node[k]['word'] = w
        
    weighted_edges = {(i, j): A[i,j] for i,j in itertools.product(nodes, nodes)}
    return G, weighted_edges

A = np.random.random((11,11))
labels = ["nsubj", "dobj", "iobj", "det", "nmod", "amod", "cc", "conj"]
sent = "This is a test sentence".split()
diG, edges = matrix_to_graph(A, sent, labels, digraph=True)
diG.remove_node(0)
for i in range(1,len(A)):
    diG.remove_edge(i,1)

#function that takes a graph and cycle (list of edges) and returns
#a graph with len(cycle) - 1 
def contract_graph(G,C):
    G_c = G.copy()
    for u,v in C:
        G_c.remove_node(u)
    newnode = 999
    G_c.add_node(newnode)
    
    #get the min. outgoing edges out of the cycle
    nodes = list(G_c.nodes)
    nodes.remove(newnode)
    nodes_withoutroot = nodes
    nodes_withoutroot.remove(1)
    for node in nodes:    
        weightset = False
        for u,v in C:
            if weightset == False:
                min_outweight = G[u][node]['weight']
                min_outedge = (u,node)
                weightset = True
                #print(min_inweight,min_inedge)
            if G[u][node]['weight'] > min_outweight:
                min_outweight = G[u][node]['weight']
                min_outedge = (u,node)
        G_c.add_edge(newnode,min_outedge[1],weight=min_outweight,original_endpoint=min_outedge[0])
    
    #get min. incoming edges into the cycle
    nodes.append(1)
    for node in nodes:
        weightset = False
        for u,v in C:
            inweight = G[node][u]['weight']
            for a,b in C:
                if b!=u:
                    inweight += G[a][b]['weight']
            if weightset == False:
                min_inweight = inweight
                min_inedge = (node,u)
                weightset = True
            if inweight > min_inweight:
                min_inweight = inweight
                min_inedge = (node,u)
        G_c.add_edge(min_inedge[0],newnode, weight = min_inweight,original_endpoint=min_inedge[1])
        
    return G_c

#function takes a contracted graph and a contracted tree and returns the 
#extracted tree including the nodes of the original graph
def extract_graph(T_contract,G_contract,C):
    newnode = 999
    edges = T_contract.edges()
    T = T_contract.copy()
    for u,v in edges:
        if (v == newnode):
            addnode = G_contract[u][v]['original_endpoint']
            T.remove_edge(u,v)
            T.add_edge(u,addnode)
    for u,v in edges:
        if (u == newnode):
            addnode = G_contract[u][v]['original_endpoint']
            T.remove_edge(u,v)
            T.add_edge(addnode,v)
    for u,v in C:
        if not(v in T):
            T.add_edge(u,v)
    return T

#function that takes in a graph G and returns the "tree" that results from
#taking the max. incoming edge for each node
def returnTree(G):
    F = nx.DiGraph()
    for node in G.nodes:
        if node != 1:
            edges = G.in_edges(node, data='weight') #get incoming edges for each node
            isWeightSet = False
            res = ()
            for edge in edges:
                if (isWeightSet == False):
                    minweight = edge[2]
                    isWeightSet = True
                    res = (edge[0],edge[1])
                if edge[2] > minweight:
                    minweight = edge[2]
                    res = (edge[0],edge[1])
            F.add_edge(res[0],res[1],weight=minweight)
    return F

#function takes in graph G and returns the MST of that graph
#assumes node 1 is the root node, so we can loop over the rest
def findMST(G):    
    F = returnTree(G)
    #now we check if F is a tree (there are no cycles)
    #TO ADD: also check if every node except the root node has exactly one edge entering! 
    if is_tree(F):
        return F
    else:
        C = find_cycle(F)
        G_prime = contract_graph(G,C)
        T_prime = findMST(G_prime)
        #if is_tree(T_prime):
        T = extract_graph(T_prime,G_prime,C)
        return T

test = findMST(diG)
print(set(test.edges))

def minimum_tree(G):
    edmonds = Edmonds(G)
    tree = edmonds.find_optimum()
    return tree

test2 = minimum_tree(diG)
print(set(test2.edges))

print((set(test.edges) == set(test2.edges)))