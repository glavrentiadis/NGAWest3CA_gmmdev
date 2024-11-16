#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:23:51 2024

@author: glavrent
"""

import numpy as np
import networkx as nx

def find_indept_clusters(grpids):
    #number of nodes
    n = grpids.shape[0]
    
    #create a graph
    print("Graph construction ...")
    G = nx.Graph()
   
    #add nodes to the graph
    print("\t setting up nodes ...")
    for i in range(n):
        G.add_node(i)  # Nodes are identified by their index in the arrays

    #add edges between nodes that share the same grp or stid
    print("\t setting up edges ...")
    # for j in range(n):
    #     i_n_cnct = [ l for l in range(j + 1, n) if np.any( np.isclose(grpids[j,:], grpids[l,:]) ) ]
    #     for l in i_n_cnct:
    #         G.add_edge(j, l)
    edge_idx = [[(j,l) for l in range(j + 1, n) if np.any( np.isclose(grpids[j,:], grpids[l,:]) ) ] for j in range(n)]
    edge_idx = sum(edge_idx, [])
    for j, l in edge_idx:
        G.add_edge(j, l)
    print("Completed graph construction.")

    print("Independet sub-graph identification ...")
   #group nodes with common edges
    clusters = []
    for G_s in nx.connected_components(G):
        clusters.append(list(G_s))
    print("Completed independent sub-graphs...")
    
    return clusters

