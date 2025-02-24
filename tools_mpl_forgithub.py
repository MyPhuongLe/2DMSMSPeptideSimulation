#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import networkx as nx
from collections import *
from scipy.linalg import eigh
from sklearn.preprocessing import normalize

def laplacian_eig(graph, nodes, k):
    # graph is undirected
    # return top k eigen-pairs
    adj = nx.to_numpy_array(graph, nodelist=nodes, weight='weight') 
    deg_vec = np.sum(adj, 1) 
    deg = np.diag(deg_vec) 
    laplacian = deg - adj 
    laplacian_sym = np.diag(deg_vec ** -0.5) @ laplacian @ np.diag(deg_vec ** -0.5) 
    _, eigvec1 = eigh(laplacian, subset_by_index=[0, k-1])
    _, eigvec = eigh(laplacian_sym, subset_by_index=[0, k-1])
    eigvec2 = np.diag(deg_vec ** -0.5) @ eigvec
    eigvec3 = normalize(eigvec, norm='l2', axis=1)
    return eigvec1, eigvec2, eigvec3

def min_cond_set(v2, graph, nodes):
    v2_sort_index = np.argsort(v2)
    min_cond = float("inf")
    S_opt = []
    S = []
    for i in range(len(graph)-1):
        S.append(nodes[v2_sort_index[i]])
        cond = nx.conductance(graph, S, weight="weight")
        if cond < min_cond:
            min_cond, S_opt = cond, S[:]
    return S_opt

def set2label(S_opt, destinations, indexed, nodes):
    idx1 = [] 
    idx2 = [] 
    for i, dest in enumerate(destinations):
        if dest in S_opt:
            idx1.append(i+indexed)
        else:
            idx2.append(i+indexed)
    
    assert len(idx1) == 1
    assert len(idx2) == 1
    
    idx1 = idx1[0]
    idx2 = idx2[0]
    
    est_labels = defaultdict(set)
    for v in nodes:
        if v in S_opt:
            est_labels[v].add(idx1)
        else:
            est_labels[v].add(idx2)
    
    return est_labels

# return paths in the graph starting from node u of length n
def findPaths(graph, u, n):
    if n == 0:
        return [[u]]
    paths = [[u]+path for neighbor in graph.neighbors(u) for path in findPaths(graph, neighbor, n-1) if u not in path]
    return paths

def comp_adj2(graph, nodes):
    allpaths = []
    for u in nodes:
        allpaths.extend(findPaths(graph, u, 2))

    node_ids = {u: i for i, u in enumerate(nodes)}
        
    adj2 = np.zeros((len(nodes), len(nodes)))
    for a, b, c in allpaths:
        adj2[node_ids[a]][node_ids[b]] += 1
        adj2[node_ids[b]][node_ids[c]] += 1
        adj2[node_ids[a]][node_ids[c]] += 1

        adj2[node_ids[b]][node_ids[a]] += 1
        adj2[node_ids[c]][node_ids[b]] += 1
        adj2[node_ids[c]][node_ids[a]] += 1

    return adj2
        
def adj2net(adj, nodes):
    graph = nx.Graph()
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                u, v = nodes[i], nodes[j]
                graph.add_edge(u, v, weight=adj[i, j])
    return graph
    
def label_propagation(x, adj, nodes, destinations, eps, max_iter, reset=1):
    old_x = np.zeros(np.shape(x))
    new_x = normalize(x, axis=1, norm='l1')
    i = 0
    while np.linalg.norm(new_x - old_x) > eps and i < max_iter:
#         print(i, np.linalg.norm(new_x - old_x))
        old_x = new_x
        new_x = adj @ old_x
        for j, dest in enumerate(destinations):
            new_x[nodes.index(dest), j] = reset
        new_x = normalize(new_x, axis=1, norm='l1')
        i += 1
    return new_x

def predict_labels_nn(eigvec, nodes, destinations, indexed):
    est_labels = defaultdict(set)
    for i, v in enumerate(nodes):
        dist = []
        e1 = eigvec[i, :]
        for dest in destinations:
            e2 = eigvec[nodes.index(dest), :]
            dist.append(np.linalg.norm(e1 - e2))
#         indexes = np.where(dist == np.min(dist))[0]
        indexes = np.where(dist - np.min(dist) < 1e-8)[0]
        for idx in indexes:
            est_labels[v].add(idx+indexed)
    return est_labels
            
def predict_labels_max(scores, nodes, indexed, thr):
    est_labels = defaultdict(set)
    for i, v in enumerate(nodes):
        indexes = np.where(scores[i, :] == np.max(scores[i, :]))[0]
        for idx in indexes:
            if scores[i, idx] > thr[idx]:
                est_labels[v].add(idx+indexed)
    return est_labels

def predict_labels_thr(scores, nodes, indexed, thr):
    est_labels = defaultdict(set)
    for i, v in enumerate(nodes):
        indexes = np.where(scores[i, :] >= thr)[0]
        for idx in indexes:
            est_labels[v].add(idx+indexed)
    return est_labels

def predict_labels_thr2(scores, nodes, indexed, thr):
    est_labels = defaultdict(set)
    for i, v in enumerate(nodes):
        indexes = np.where(scores[i, :] > thr)[0]
        for idx in indexes:
            est_labels[v].add(idx+indexed)
    return est_labels

def predict_labels_maxthr(scores, nodes, indexed, thr1, thr2):
    est_labels = defaultdict(set)
    for i, v in enumerate(nodes):
        indexes = np.where(scores[i, :] == np.max(scores[i, :]))[0]
        for idx in indexes:
            if scores[i, idx] > thr2[idx]:
                est_labels[v].add(idx+indexed)
        indexes = np.where(scores[i, :] >= thr1)[0]
        for idx in indexes:
            est_labels[v].add(idx+indexed)
    return est_labels

def comp_perf(est_labels, labels):
    perf = []
    for v in est_labels:
        if est_labels[v] == labels[v]:
            perf.append("correct")
        elif est_labels[v].issubset(labels[v]):
            perf.append("half correct")
        elif est_labels[v].isdisjoint(labels[v]):
            perf.append("wrong")
        else:
            perf.append("half wrong")
    
    perf_pct = Counter(perf)
    perf_pct = {i: perf_pct[i]/len(perf) for i in perf_pct}
    
    for i in ["correct", "half correct", "half wrong", "wrong"]:
        if i not in perf_pct:
            perf_pct[i] = 0
    
    return perf, perf_pct

def retain_percentage(labels,est_labels,C):
    pct = []
    for i in range(C):
        cnt_labels = sum(1 for value_set in labels.values() if i+1 in value_set)
        cnt_est_labels = sum(1 for value_set in est_labels.values() if i+1 in value_set)
        pct.append(cnt_est_labels/cnt_labels)
    return pct

def percent_miss(labels, est_labels,C):
    count = 0
    miss = 0
    for n in labels:
        if any(item in labels[n] for item in list(range(1,C+1))):
            count+=1
            if n not in est_labels:
                miss+=1
    return(miss/count)

def percent_miss_each(labels,est_labels,C):
    miss_percent = []
    for i in range(1,C+1):
        missed = 0
        Csig = [key for key, value in labels.items() if i in value]
        est_Csig = [key for key, value in est_labels.items() if i in value]
        for n in Csig:
            if n not in est_Csig:
                missed+=1
        miss_percent.append(missed/len(Csig))
    return miss_percent

def fwrite_perf(f, perf_pct):
    f.write(str(round(perf_pct['correct'], 5)) + ',')
    f.write(str(round(perf_pct['half correct'], 5)) + ',')
    f.write(str(round(perf_pct['half wrong'], 5)) + ',')
    f.write(str(round(perf_pct['wrong'], 5)) + ',')

def f1_score_calculation(labels, est_labels,C):
    ground_truth = {i: [k for k, v in labels.items() if i in v] for i in range(1, C+1)}
    predicted = {i: [k for k, v in est_labels.items() if i in v] for i in range(1, C+1)}
    
    f1_dict = dict()
    for i in range(1,C+1):
        intersection = len(set(ground_truth[i]) & set(predicted[i]))
        precision = intersection/len(predicted[i]) if predicted[i] else 0
        recall = intersection / len(ground_truth[i]) if ground_truth[i] else 0
        f1_score = (2*precision*recall)/(precision+recall) if precision + recall else 0
        f1_dict[i] = [precision,recall,f1_score]
        
    return f1_dict

