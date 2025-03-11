#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import random
import numpy as np
import networkx as nx
import pandas as pd
from data_generation_mpl import *
from tools_mpl import *
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import statistics
import time
from tqdm import tqdm


# #### Load a collection of real-world peptides

# In[3]:


all_seqs = []
with open("Peptide_mass.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # skip the first line
    next(csv_reader) # skip the first sequence that contains a space
    for row in csv_reader:
        seq = row[0]
        seq = seq.replace("L", "") 
        seq = seq.replace("Q", "") 
        all_seqs.append(seq)


# In[4]:


# List of amino acids
aas = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

all_seqs = []
n=0
# Generate a random string length between 8 and 15
while n<10000:
    string_length = random.randint(4, 8)
    random_string = ''.join(random.choice(aas) for _ in range(string_length))
    all_seqs.append(random_string)
    n+=1


# #### synthetic data generation
# * Type-1 noise: Noise from background peptides
# * Type-2 noise: Adducts (existing peptide + 22)
# * Type-3 noise: Random precursor-product pairs with neutral loss > 17
# * Type-1 rearrangements: Move the last amino acid in the sequence of a fragment to the beginning
# * Type-2 rearrangements: Subtract a random value between 20 and 80 from a fragment's mass
# 
# #### input setting
# - seqs_signal: peptides to be identified
# - seqs_noise1: background peptides
# - seqs_noise2: selected from seqs_signal for producing adducts
# - p_signal: fragmentation probabilities for seqs_signal
# - p_noise1: fragmentation probabilities for seqs_noise1
# - p_noise2: fragmentation probabilities for seqs_noise2
# - pct_noise3: fraction of type-3 noise
# - n_rearrange1: count of type-1 rearrangements
# - n_rearrange2: count of type-2 rearrangements

# ### Simulation

# In[5]:


def run_test(threshold_1):
    # generate the synthetic data
    coordinates, labels, mass2seq, destinations, rearrange1, rearrange2,random_noise, noise1, adducts,random_nodes_noise4, random_edges_noise4 = generate_data(seqs_signal, seqs_noise1, seqs_noise2, p_signal, p_noise1, p_noise2, pct_noise3, n_rearrange1, n_rearrange2, indexed, mass_lower, mass_upper, cnt_random_nodes_noise4, cnt_random_edges_noise4)
    C = len(destinations) # number of clusters

    # build the graph
    G = nx.from_edgelist(coordinates, create_using=nx.DiGraph())

    # pagerank scores
    scores = np.zeros((len(G), C))
    for j, dest in enumerate(destinations):
        r = nx.pagerank(G, alpha=0.85, personalization={dest: 1}, weight=None)
        for i, v in enumerate(G.nodes):
            scores[i, j] = r[v]
        
    # consider the subset of nodes that connect to any of the destinations 
    indexes = np.any(scores, 1)
    scores = scores[indexes]
    subg_nodes = [v for i, v in enumerate(G.nodes) if indexes[i]]
    subg = G.subgraph(subg_nodes)
    subg_adj = nx.to_numpy_array(subg, nodelist=subg_nodes).T

    # count nodes associted with each cluster
    cluster_size = [0] * C
    for i in range(indexed, C+indexed): # for label i
        for v in subg_nodes:
            if i in labels[v]: # if node v has label i
                cluster_size[i-indexed] += 1

    # count pairs with type-1 noise
    cnt_noisy1 = 0
    for (x, y) in noise1:
        if subg.has_edge(x, y):
            cnt_noisy1 += 1

    # count pairs of type-2 noise
    cnt_noisy2 = 0
    for (x, y) in adducts:
        if subg.has_edge(x, y):
            cnt_noisy2 += 1

    # count shared nodes
    cnt_shared_same = 0 # same mass, same structure
    cnt_shared_diff = 0 # same mass, different structure
    shared_M = 0
    for v in subg_nodes:
        if len(mass2seq[v]) > 1:
            if v in destinations:
                shared_M += 1
            if len({item[0] for item in mass2seq[v]}) == 1:
                cnt_shared_same += 1
            else:
                cnt_shared_diff += 1

    # count how many pairs in random_noise are included
    cnt_noisy3 = 0
    for (x, y) in random_noise:
        if subg.has_edge(x, y):
            cnt_noisy3 += 1

    # count how many pairs in random_edges_noise4 are included
    cnt_noisy4 = 0
    for (x, y) in random_edges_noise4:
        if subg.has_edge(x, y):
            cnt_noisy4 += 1
    
    # naive method
    est_labels_naive = predict_labels_thr2(scores, subg_nodes, indexed, 0)
    perf_naive, perf_pct_naive = comp_perf(est_labels_naive, labels)

    # methods based on pagerank
    est_labels_pr1 = predict_labels_max(scores, subg_nodes, indexed, [0]*C)
    perf_pr1, perf_pct_pr1 = comp_perf(est_labels_pr1, labels)

    scores_incluster = [[] for i in range(C)]
    for i, v in enumerate(subg_nodes):
        for j in est_labels_pr1[v]:
            scores_incluster[j-indexed].append(scores[i, j-indexed])
        
    scores_incluster = [[] for i in range(C)]
    for i, v in enumerate(subg_nodes):
        for j in est_labels_pr1[v]:
            scores_incluster[j-indexed].append(scores[i, j-indexed])

    #calculate percentile values for threshold
    percentiles = list(range(0,105,5))
    s = {p: [np.percentile(scores, p) for scores in scores_incluster] for p in percentiles}

    #predict labels using different threshold
    all_est_labels = dict()
    all_est_labels = {t1: {t2: predict_labels_maxthr(scores, subg_nodes, indexed, s[t1], s[t2]) 
                      for t2 in [0]+list(range(5, t1, 5))} 
                      for t1 in threshold_1}

    
    return (est_labels_naive,all_est_labels, labels,
    [cnt_noisy1,
    cnt_noisy2,
    cnt_noisy3 + cnt_noisy4],
    cnt_shared_same+cnt_shared_diff,
    [subg.number_of_nodes(),
    subg.number_of_edges()],
    [seqs_signal,seqs_noise1,seqs_noise2])


# In[300]:


# Parameters
threshold_1 = [70,75,80,85,90,95]

n_signal = 5
n_noise1 = 0
n_noise2 = n_signal+n_noise1

p_signal = [[0.7]*n_signal, [0.7]*n_signal] 
p_noise1 = [[0.7]*n_noise1, [0.7]*n_noise1]
p_noise2 = [[0.1]*n_noise2, [0.05]*n_noise2]
pct_noise3 = 0.01
n_rearrange1 = 4
n_rearrange2 = 4

mass_lower = 100
mass_upper = 1000
cnt_random_nodes_noise4 = 50
cnt_random_edges_noise4 = 100

indexed = 1

# Run simulation
C = n_signal

all_est_labels_collection = []
labels_collection = []
naive_labels_collection = []

total_iterations = 50

while nn < total_iterations:
    seqs_signal_noise1 = random.sample(all_seqs, n_signal + n_noise1)
    seqs_signal = seqs_signal_noise1[:n_signal]
    seqs_noise1 = seqs_signal_noise1[n_signal:]
    seqs_noise2 = random.sample(seqs_signal+seqs_noise1, n_noise2) 

    est_labels_naive, all_est_labels, labels, noises, isobaric, node_edge_count, sequences = run_test(threshold_1)

    nn+=1

# all_est_labels stores the signal assignment of each simulation using the partitioning algorithm with the corresponding parameters
# est_labels_naives stores the signal assignment of each simulation if no partitioning algorithm is used
# labels stores the ground truth

