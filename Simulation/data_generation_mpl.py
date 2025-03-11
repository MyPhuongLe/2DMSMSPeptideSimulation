#!/usr/bin/env python
# coding: utf-8

# In[22]:


# %load data_generation.py
import random
from collections import defaultdict 

amino_acid_mass = {"A": 71, 
                   "R": 156, 
                   "N": 114, 
                   "D": 115, 
                   "C": 103, 
                   "E": 129, 
                   "Q": 128, 
                   "G": 57, 
                   "H": 137, 
                   "I": 113, 
                   "L": 113, 
                   "K": 128, 
                   "M": 131, 
                   "F": 147, 
                   "P": 97, 
                   "S": 87, 
                   "T": 101, 
                   "W": 186, 
                   "Y": 163, 
                   "V": 99}

def comp_seq_mass(seq):
    total = 0
    for c in seq:
        total += amino_acid_mass[c]
    return total

def fragmentation(seq, p):
    if len(seq) < 2:
        return []
        
    sub_seqs = []
    n = len(seq)
    for l in range(1, n): # sub-seq of length l
        for i in range(n-l+1): # start from i
            if random.random() < p:
                sub_seqs.append(seq[i:i+l])
    return sub_seqs 

def create_rearrangements(stage1_signal, n_rearrange1, n_rearrange2):
    
    # collect all stage-1 signal fragments into a list
    stage1_signal_1d = sum(stage1_signal, []) 
    
    # filter fragments of length between 3 and 5 for potential rearrangements
    candidates1 = [seq for seq in stage1_signal_1d if 3 <= len(seq) <= 5]
    candidates1 = list(set(candidates1))
    
    # select fragments for type-1 rearrangements
    rearrange1 = {} 
    n_rearrange1_real = min(n_rearrange1, len(candidates1)) # just in case fragments are not enough
    selected1 = random.sample(candidates1, n_rearrange1_real)
    for seq in selected1:
        rearrange1[seq] = seq[-1] + seq[:-1]
        candidates1.remove(seq)
    
    # select fragments for type-2 rearrangements
    candidates_groups = {} # group remaining fragments by suffix
    candidates1 = sorted(candidates1, key=lambda x: len(x), reverse=True)
    for seq in candidates1:
        flag = True
        for exist in candidates_groups:
            if exist.endswith(seq):
                flag = False
                candidates_groups[exist].append(seq)
                break
        if flag:
            candidates_groups[seq] = [seq]

    candidates2 = []
    for key in candidates_groups:
        candidates2.append(random.choice(candidates_groups[key])) # select one from each group

    rearrange2 = {}
    n_rearrange2_real = min(n_rearrange2, len(candidates2)) # just in case fragments are not enough
    selected2 = random.sample(candidates2, n_rearrange2_real)
    for seq in selected2:
        rearrange2[seq] = random.choice(range(20, 81))
    
    return rearrange1, rearrange2

def generate_signal_noise1(seqs_signal, seqs_noise1, p_signal, p_noise1, stage1_signal, rearrange1, rearrange2):
    
#     pairs = set() # collection of (precursor, product, label)
    pairs_signal = set()
    pairs_noise1 = set()
    
    n_signal = len(seqs_signal)
    n_noise1 = len(seqs_noise1)
    
    # add signal
    for i in range(n_signal):
        for j in range(len(stage1_signal[i])):
            # stage-1
            if stage1_signal[i][j] in rearrange1:
                stage1_signal[i][j] = rearrange1[stage1_signal[i][j]]
            pairs_signal.add((seqs_signal[i], stage1_signal[i][j], i))
            # stage-2
            stage2 = fragmentation(stage1_signal[i][j], p_signal[1][i])
            for s in stage2:
                if s in rearrange1 and not (stage1_signal[i][j] in rearrange2 and stage1_signal[i][j].endswith(s)):
                    s = rearrange1[s]
                pairs_signal.add((stage1_signal[i][j], s, i))
        # stage-2 for the full sequence 
        stage2 = fragmentation(seqs_signal[i], p_signal[1][i])
        for s in stage2:
            if s in rearrange1:
                s = rearrange1[s]
            pairs_signal.add((seqs_signal[i], s, i))
    
    # add noise1
    for i in range(n_noise1):
        stage1 = fragmentation(seqs_noise1[i], p_noise1[0][i])
        for j in range(len(stage1)):
            # stage-1
            if stage1[j] in rearrange1:
                stage1[j] = rearrange1[stage1[j]]
            pairs_noise1.add((seqs_noise1[i], stage1[j], n_signal))
            # stage-2
            stage2 = fragmentation(stage1[j], p_noise1[1][i])
            for s in stage2:
                if s in rearrange1 and not (stage1[j] in rearrange2 and stage1[j].endswith(s)):
                    s = rearrange1[s]
                pairs_noise1.add((stage1[j], s, n_signal))
        # stage-2 for the full sequence
        stage2 = fragmentation(seqs_noise1[i], p_noise1[1][i])
        for s in stage2:
            if s in rearrange1:
                s = rearrange1[s]
            pairs_noise1.add((seqs_noise1[i], s, n_signal))
#     pairs = pairs_signal + pairs_noise1
    
    return pairs_signal, pairs_noise1

def generate_noise2(seqs_noise2, p_noise2):
    
    pairs_adducts = set() # collection of (precursor, product)
    
    n_noise2 = len(seqs_noise2)

    # add noise2
    for i in range(n_noise2):
        stage1 = fragmentation(seqs_noise2[i], p_noise2[0][i])
        for j in range(len(stage1)):
            pairs_adducts.add((seqs_noise2[i], stage1[j]))
            stage2 = fragmentation(stage1[j], p_noise2[1][i])
            for s in stage2:
                pairs_adducts.add((stage1[j], s))
        stage2 = fragmentation(seqs_noise2[i], p_noise2[1][i])
        for s in stage2:
            pairs_adducts.add((seqs_noise2[i], s))
    
    return pairs_adducts

# Type-1 noise: From other peptides
# Type-2 noise: Adducts
# Type-3 noise: Random noise
# Type-1 rearrangements: Move the last amino acid in the sequence of a fragment to the beginning
# Type-2 rearrangements: Subtract a random value between 20 and 80 from a fragment's mass

def generate_data(seqs_signal, seqs_noise1, seqs_noise2, p_signal, p_noise1, p_noise2, pct_noise3, 
                  n_rearrange1, n_rearrange2, indexed, mass_lower, mass_upper,
                  cnt_random_noisy_nodes1, cnt_random_noisy_edges1):
    
    coordinates = set()
    labels = defaultdict(set)
    mass2seq = defaultdict(set)

    n_signal = len(seqs_signal)
    
    # two-stage fragmentation
    
    stage1_signal = []
    for i in range(n_signal):
        stage1_signal.append(fragmentation(seqs_signal[i], p_signal[0][i]))
    
    rearrange1, rearrange2 = create_rearrangements(stage1_signal, n_rearrange1, n_rearrange2)
    
    pairs_signal,pairs_noise1 = generate_signal_noise1(seqs_signal, seqs_noise1, p_signal, p_noise1, stage1_signal, rearrange1, rearrange2)
    pairs_adducts = generate_noise2(seqs_noise2, p_noise2)
    
    # add signal
    for (precursor, product, label) in pairs_signal:
        x = comp_seq_mass(precursor)
        y = comp_seq_mass(product)
        if product in rearrange2:
            y -= rearrange2[product]
        elif precursor in rearrange2:
            x -= rearrange2[precursor]
            if len(product) > 1 and precursor.endswith(product):
                y -= rearrange2[precursor]
        coordinates.add((x, y))
        labels[x].add(label+indexed)
        labels[y].add(label+indexed)
        mass2seq[x].add((precursor, label+indexed))
        mass2seq[y].add((product, label+indexed))

    # add noise1
    noise1 = set()
    for (precursor, product, label) in pairs_noise1:
        x = comp_seq_mass(precursor)
        y = comp_seq_mass(product)
        if product in rearrange2:
            y -= rearrange2[product]
        elif precursor in rearrange2:
            x -= rearrange2[precursor]
            if len(product) > 1 and precursor.endswith(product):
                y -= rearrange2[precursor]
        coordinates.add((x, y))
        noise1.add((x, y))
        labels[x].add(label+indexed)
        labels[y].add(label+indexed)
        mass2seq[x].add((precursor, label+indexed))
        mass2seq[y].add((product, label+indexed))

    # add noise2
    adducts = set()
    for (precursor, product) in pairs_adducts:
        x = comp_seq_mass(precursor)
        y = comp_seq_mass(product)
        # the mass of a single amino acid stays normal
        if len(precursor) > 1:
            x += 22
        if len(product) > 1:
            y += 22
        coordinates.add((x, y))
        adducts.add((x, y))
        labels[x].add(n_signal+1+indexed)
        labels[y].add(n_signal+1+indexed)
        mass2seq[x].add((precursor, n_signal+1+indexed))
        mass2seq[y].add((product, n_signal+1+indexed))

    # add noise3
    random_noise = set()
    mass_list = list(labels.keys())
    cnt = len(coordinates) * pct_noise3 / (1-pct_noise3)
    while cnt > 0:
        xy = random.sample(mass_list, 2)
        if abs(xy[0]-xy[1]) > 17: # the neutral loss should be larger than 17
            xy = tuple(sorted(xy, reverse=True))
            if xy not in coordinates:
                coordinates.add(xy)
                random_noise.add(xy)
                cnt -= 1
    
    destinations = [comp_seq_mass(s) for s in seqs_signal]
    count_noise12 = len(pairs_noise1) + len(pairs_adducts)
    
#     return coordinates, labels, mass2seq, destinations, rearrange1, rearrange2, random_noise, noise1, adducts

    # add noise4: random edges connecting one existing node and one extra node
    random_noisy_nodes1 = set()
    while len(random_noisy_nodes1) < cnt_random_noisy_nodes1:
        random_noisy_node = random.choice(range(mass_lower, mass_upper+1))
        if random_noisy_node not in mass_list:
            random_noisy_nodes1.add(random_noisy_node)
    random_noisy_nodes1 = list(random_noisy_nodes1)
    
    random_noisy_edges1 = set()
    i = 0
    while len(random_noisy_edges1) < cnt_random_noisy_edges1:
        existing_node = random.choice(mass_list)
        extra_node = random_noisy_nodes1[i%cnt_random_noisy_nodes1]
        if abs(existing_node-extra_node) > 17:
            xy = tuple(sorted([existing_node, extra_node], reverse=True))
            random_noisy_edges1.add(xy)
            i += 1
    random_noisy_edges1 = list(random_noisy_edges1)
    
    for v in random_noisy_nodes1:
        labels[v].add(n_signal+2+indexed)
    
    for xy in random_noisy_edges1:
        coordinates.add(xy)
    
    destinations = [comp_seq_mass(s) for s in seqs_signal]
    
    return coordinates, labels, mass2seq, destinations, rearrange1, rearrange2, random_noise, noise1, adducts, random_noisy_nodes1, random_noisy_edges1


# In[ ]:





# In[ ]:




