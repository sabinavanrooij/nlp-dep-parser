# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from networkx.algorithms.tree.branchings import Edmonds
from networkx.algorithms.tree import is_tree
from networkx.algorithms.cycles import simple_cycles,find_cycle
from networkx.algorithms.minors import contracted_nodes,contracted_edge,identified_nodes

G = nx.DiGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_edge(1,2,weight=12)
G.add_edge(1,3,weight=4)
G.add_edge(1,4,weight=4)
G.add_edge(2,3,weight=5)
G.add_edge(3,2,weight=6)
G.add_edge(3,4,weight=8)
G.add_edge(4,3,weight=7)
G.add_edge(2,4,weight=7)
G.add_edge(4,2,weight=5)

#print(G.edges(data='weight'))
#nx.draw(G, pos=nx.spring_layout(G))
#print(G.in_edges(2, data='weight'))
#print(G.in_edges(3, data='weight'))
#print(G.in_edges(4, data='weight'))
#print(G.edges(data='weight'))

#function takes a graph and two nodes (numbers) that need to be contracted
#and returns a graph one less node than the original graph
#def contract_graph(G,u,v):
#    u_inedges = G.in_edges(u, data='weight') 
#    v_inedges = G.in_edges(v, data='weight')
#    u_outedges = G.out_edges(u, data='weight')
#    v_outedges = G.out_edges(v, data='weight') 
#    newnode = int(str(u) + str(v) + str(0)) #create a new node with extra 0 in case the sentence in longer than concatenation of the two numbers
#    G.add_node(newnode)
#    for a,b,d in u_inedges:
#        if G.has_edge(a,newnode):
#            if d > G[a][newnode]['weight']:
#                G.add_edge(a,newnode,weight=d)      
#        else:
#            G.add_edge(a,newnode,weight=d)      
#    for a,b,d in v_inedges:
#        if G.has_edge(a,newnode):
#            if d > G[a][newnode]['weight']:
#                G.add_edge(a,newnode,weight=d)      
#        else:
#            G.add_edge(a,newnode,weight=d) 
#    for a,b,d in u_outedges:
#        if G.has_edge(newnode,b):
#            if d > G[newnode][b]['weight']:
#                G.add_edge(newnode,b,weight=d)
#        else:
#            G.add_edge(newnode,b,weight=d)  
#    for a,b,d in v_outedges:
#        if G.has_edge(newnode,b):
#            if d > G[newnode][b]['weight']:
#                G.add_edge(newnode,b,weight=d)
#        else:
#            G.add_edge(newnode,b,weight=d) 
#    G.remove_node(u)
#    G.remove_node(v)
#    G.remove_edge(newnode,newnode)
#    return(G)


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
    for u,v in C:
        weightset = False
        for node in nodes:
            if weightset == False:
                min_outweight = G[u][node]['weight']
                min_outedge = (u,node)
                weightset = True
                #print(min_inweight,min_inedge)
            if G[u][node]['weight'] < min_outweight:
                min_outweight = G[u][node]['weight']
                min_outedge = (u,node)
        G_c.add_edge(newnode,min_outedge[1],weight=min_outweight,original_endpoint=min_outedge[1])
    
    #get min. incoming edges into the cycle
    nodes.append(1)
    for node in nodes:
        print("node",node)
        for u,v in C:
            min_inweight = G[node][u]['weight']
            print(min_inweight)
            for a,b in C:
                if b!=u:
                    min_inweight += G[a][b]['weight']
            print(min_inweight)
        
    
    
    print(G_c.edges(data='weight'))
    return G_c
    
test = contract_graph(G,[(3,4),(4,3)])
#print(test.nodes)

#takes a list of edges andreturns a list with edges that form a cycle 
#in a graph even if there are multiple cycles, it just returns one cycle
def returncycle(T):
    K = nx.DiGraph()
    for edge in T:
        K.add_edge(edge[0],edge[1])
    C = find_cycle(K)
    return C


#function that takes in a graph G and returns the "tree" that results from
#taking the max. incoming edge for each node
def returnTree(G):
    F = []
    V = G.number_of_nodes()
    for node in range(2,V+1):
        edges = G.in_edges(node, data='weight') #get incoming edges for each node
        len_edges = len(edges)
        print(len_edges)
        isWeightSet = False
        #maxweight = 0
        res = ()
        print(edges)
        for edge in edges:
            if (isWeightSet == False):
                maxweight = edge[2]
                isWeightSet = True
                res = (edge[0],edge[1])
            if edge[2] > maxweight:
                maxweight = edge[2]
                res = (edge[0],edge[1])
    F.append(res)
    return F

#print(returnTree(G))

#function takes in graph G and returns the MST of that graph
#assumes node 1 is the root node, so we can loop over
def findMST(G):
    F = [] # 
    T_prime = [] #the spanning tree
    score = [] #weights in edges
    V = G.number_of_nodes()

    #first we make a "tree" from the highest scoring incoming edges 
    for node in range(2,V+1):
        edges = G.in_edges(node, data='weight') #get incoming edges for each node
        len_edges = len(edges)
        print(len_edges)
        isWeightSet = False
        #maxweight = 0
        res = ()
        print(edges)
        if len_edges == 1:
            maxweight = edges[0][2]
            res = (edges[0][0],edges[0][1])
        else:
            for edge in edges:
                if (isWeightSet == False):
                    maxweight = edge[2]
                    isWeightSet = True
                    res = (edge[0],edge[1])
                if edge[2] > maxweight:
                    maxweight = edge[2]
                    res = (edge[0],edge[1])
        F.append(res)
        
        #adjust the weights by subtracting the maxweights
        #from all incoming edges
        for u,v,d in edges:
            G[u][v]['weight'] = d - maxweight
    
    #make F (list of edges) into digraph 
    K = nx.DiGraph()
    for edge in F:
        K.add_edge(edge[0],edge[1])
    
    #now we check if F is a tree (there are no cycles)
    #TO ADD: also check if every node except the root node has 
    #exactly one edge entering! 
    if is_tree(K):
        return K
    else:
        C = find_cycle(K)
        #G_prime = contracted_nodes(G,C[0][0],C[0][1])
        #G_prime = contract_graph(G,C[0][0],C[0][1])
        #print(G_prime)
        #T_prime = returnTree(G_prime)
        return C

#test = findMST(G)
#print(test.edges(data='weight'))