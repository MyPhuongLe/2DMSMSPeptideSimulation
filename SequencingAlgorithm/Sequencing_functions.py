import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import pandas as pd

import scipy
from scipy import ndimage as ndi
from scipy import misc

import numpy as np
from numpy import *

from math import sqrt

from collections import Counter
from itertools import product, combinations

from pepfrag import MassType, ModSite, Peptide

from tqdm.notebook import trange, tqdm
from time import sleep

from itertools import combinations
import copy


nl_small = {18:"H2O",17:"NH3 loss",35:"NH3+H2O",36:"H2O+H2O",44:"C2H4O",45:"CO+NH3",46:"CO+H2O"}

#Calculate the neutral loss between two masses and get the list of structures in to correct format

def get_neutralloss(smallermass_d,highermass_d,nl_lib):
    nl_lib = nl_lib
    dfnl = nl_lib[nl_lib["Mass"]==round(highermass_d-smallermass_d)]
    nls = [[seq, type_] for seq, type_ in zip(dfnl["Sequence"].tolist(), dfnl["Type"].tolist())]
    return nls

#Check if sequence has more than 1 terminal modification

def check_sequence(sequence):
    if sequence.count("Ac") >= 2:
        return False
    elif sequence.count("NH2") >= 2:
        return False
    else:
        return True

#Sequence filter before M 

def contain_aa_potential_modified_firstpath(smallermass_d,highermass_d,sequence,M_d,frag_lib,nl_lib,nl_small,modifications):
    nl_small = nl_small
    modifications = modifications
    frag_lib = frag_lib
    nl_lib = nl_lib

    M = round(M_d)
    nls = get_neutralloss(smallermass_d,highermass_d,nl_lib)
    nl = round(highermass_d - smallermass_d)
    
    smallermass = round(smallermass_d)
    highermass = round(highermass_d)
    
    if nl in nl_small:
        print("Loss of "+nl_small[nl],smallermass,highermass)
        return sequence, "skip"

    #stop if no neutral loss is found, stop looping through mass series
    if not nls:
        print("Check! ",smallermass,highermass)
        return sequence, "break"
    
    #check if smaller mass has candidate:
    if smallermass not in sequence or sequence[smallermass] == []:
        df = frag_lib[frag_lib["Mass"]==smallermass] #get candidates of first mass
        sequence[smallermass] = list(zip(list(df["Sequence"]),list(df["Type"]))) #store candidates to dictionary

    #add candidate storage for next mass
    sequence[highermass] = []
    
    #Loop through each smaller mass candidate to get high mass pool
    for aas in sequence[smallermass]:
        #The modified higher mass is calculated for each smallermass-neutral loss pair
        #Consider modification from lower mass structure
        aas = list(aas)
        adjusted_mass = highermass
        
        mod_counts = {"p": 0, "Ac": 0, "NH2": 0}
        
        #Count modification in smaller mass
        for mod, mass_change in modifications.items():
            count = aas[0].count(mod)
            mod_counts[mod] = count
            aas[0] = aas[0].replace(mod, "") #strip modification off smaller mass sequence
            adjusted_mass += mass_change * count #update high mass
        
        #Loop through each neutral loss candidate to get high mass pool
        #Consider modification from neutral loss
        for nl_og in nls:
            nl = nl_og.copy()
            mod_counts_nl = copy.copy(mod_counts)
            
            adjusted_mass_nl = adjusted_mass
            
            #Count modification in smaller mass
            for mod, mass_change in modifications.items():
                count = nl[0].count(mod) + nl[1].count(mod)
                nl[0] = nl[0].replace(mod, "") #strip modification off NL sequence
                mod_counts_nl[mod] += count
                adjusted_mass_nl += mass_change * count #update highmass            
                        
            #Get list of sequences with adjusted high mass
            a = frag_lib[frag_lib["Mass"]==adjusted_mass_nl]
            df_adjusted_mass_nl = list(zip(list(a["Sequence"]),list(a["Type"])))
            
            
            #Filter out highmass structures with component amino acids
            for peptide in df_adjusted_mass_nl:
                sorted_aas = sorted(aas[0])
                if sorted_aas in [sorted(l) for l in list(combinations(list(peptide[0]),len(aas[0])))]:
                    if nl[0] == " " or sorted(nl[0]) in [sorted(l) for l in list(combinations(list(peptide[0]),len(nl[0])))]:
                        peptide=list(peptide)
                        
                        #reconstruct all modification
                        combine = {key: mod_counts_nl.get(key, 0) + mod_counts.get(key, 0) for key in set(mod_counts_nl) | set(mod_counts)}
                        peptide[0] = peptide[0]+"".join([mod*mod_counts_nl[mod] for mod in combine])
                        
                        if check_sequence(peptide[0]):
                            sequence[highermass].append(peptide)
                    else:
                        continue
                    
    return (sequence, "ok")

#Sequence filter with M identified

def contain_aa_potential_modified_nextpaths(smallermass_d,seq,M_d,frag_lib,nl_lib,nl_small,modifications):
    nl_small = nl_small
    modifications = modifications
    frag_lib = frag_lib
    nl_lib = nl_lib     
    
    M = round(M_d)
    sequence = seq
    smallermass = round(smallermass_d)
#     highermass = round(highermass_d)
    
    M_strucs = []
    nls = get_neutralloss(smallermass_d,M_d,nl_lib)
    nl = round(M_d-smallermass_d)

    #stop if no neutral loss is found
    if not nls:
        print("Check! ",smallermass,M)
        return sequence, "break"
        
    #check if smaller mass has candidate:
    if nl not in nl_small:
        if smallermass not in sequence or sequence[smallermass] == []:
            df = frag_lib[frag_lib["Mass"]==smallermass] #get candidates of first mass
            sequence[smallermass] = list(zip(list(df["Sequence"]),list(df["Type"]))) #store candidates to dictionary

    for aas in sequence[smallermass]:
        
        #Consider modification in lower mass
        aas = list(aas)
        adjusted_mass = M
        
        #List of all PTMs being considered
        mod_counts = {"p": 0, "Ac": 0, "NH2": 0}
        
        
        for mod, mass_change in modifications.items():
            count = aas[0].count(mod)
            mod_counts[mod] = count
            aas[0] = aas[0].replace(mod, "")
            adjusted_mass += mass_change * count
        
        #Consider modification in neutral loss
        for nl_og in nls:
            nl = nl_og.copy()
            mod_counts_nl = copy.copy(mod_counts)
            
            adjusted_mass_nl = adjusted_mass

            for mod, mass_change in modifications.items():
                count = nl[0].count(mod) + nl[1].count(mod)
                nl[0] = nl[0].replace(mod, "")
                mod_counts_nl[mod] += count
                adjusted_mass_nl += mass_change * count                    
            
            a = frag_lib[frag_lib["Mass"]==adjusted_mass_nl]
            df_adjusted_mass_nl = list(zip(list(a["Sequence"]),list(a["Type"])))
            
            for peptide in df_adjusted_mass_nl:
                sorted_aas = sorted(aas[0])
                if sorted_aas in [sorted(l) for l in list(combinations(list(peptide[0]),len(aas[0])))]:
                    if nl[0] == " " or sorted(nl[0]) in [sorted(l) for l in list(combinations(list(peptide[0]),len(nl[0])))]:
                        peptide=list(peptide)
                        combine = {key: mod_counts_nl.get(key, 0) + mod_counts.get(key, 0) for key in set(mod_counts_nl) | set(mod_counts)}
                        peptide[0] = peptide[0]+"".join([mod*mod_counts_nl[mod] for mod in combine])
                        if check_sequence(peptide[0]):
                            M_strucs.append(peptide)
                    else:
                        continue
                    
    return (M_strucs, "ok")


#Path filter of all path

#Input include the series of mass (path), the molecular mass or destination (M), 
#and an empty dictionary to store results (sequence)

def path_filter(path, M_d, sequence,frag_lib,nl_lib,nl_small,modifications):
    nl_small = nl_small
    modifications = modifications
    frag_lib = frag_lib
    nl_lib = nl_lib
    
    M = round(M_d) #molecular weight - destination
    
    first_nl = round(path[1] - path[0])
    
    if first_nl not in nl_small:
        if round(path[0]) not in sequence or sequence[round(path[0])] == []: #if no candidate has been identified for the first mass    
            df = frag_lib[frag_lib["Mass"]==round(path[0])] #get candidates of first mass
            sequence[round(path[0])] = list(zip(list(df["Sequence"]),list(df["Type"]))) #store candidates to dictionary
    
    if first_nl in nl_small:
        sequence[round(path[0])] = []
    
    if M not in sequence: #if no candidate has been identified for the molecular ion
        for i in range(1,len(path)): #loop through mass in path
            #compare candidate pools and get structures
            d,check = contain_aa_potential_modified_firstpath(path[i-1],path[i],sequence,M_d,frag_lib,nl_lib,nl_small,modifications)        
            
            if check == "skip": #stop loop if NL is a smoll molecule
                sequence[round(path[i-1])] = []
                continue

            if check == "break": #stop loop if NL not available
                return sequence
        
            else:
                sequence[round(path[i])] = list(set([tuple(l) for l in d[round(path[i])]])) 

    else: #if some candidate has been identified for the molecular ion
        for i in range(1,len(path)-1): #loop through mass in path up until before M
            d,check = contain_aa_potential_modified_firstpath(path[i-1],path[i],sequence,M_d,frag_lib,nl_lib,nl_small,modifications)        

            if check == "skip": #stop loop if NL not available
                sequence[round(path[i-1])] = []
                continue
            
            if check == "break": #stop loop if NL not available
                return sequence
            
            else:
                sequence[round(path[i])] = list(set([tuple(l) for l in d[round(path[i])]])) 
        
        sequence[M],check = contain_aa_potential_modified_nextpaths(path[i],sequence,M_d,frag_lib,nl_lib,nl_small,modifications)
    
    sequence[M] = [l for l in list(set([tuple(l) for l in sequence[M]])) if l[1] == "y"]    
    
    return sequence


#Find overlapping M 

def find_overlapping_components(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    overlapping_components = set1.intersection(set2)
    return list(overlapping_components)


def reverse_list(input_list):
    reversed_list = input_list[::-1]
    return reversed_list

def strip_func(frag, modifications):
    modifications = modifications
    for mod in modifications.keys():
        frag = frag.replace(mod, "")
    return frag

def contains_aa(big_str, small_str):
    # Count occurrences of each letter in both strings
    count_big = {}
    count_small= {}

    for aa in big_str:
        count_big[aa] = count_big.get(aa, 0) + 1

    for aa in small_str:
        count_small[aa] = count_small.get(aa, 0) + 1

    # Check if all letters in str2 are present in str1
    for aa, count in count_small.items():
        if aa not in count_big or count_big[aa] < count:
            return False

    return True

def alphabetize_string(string):
    # Convert string to a list of characters, sort it, and join back to form a string
    return ''.join(sorted(string))


def top_down_filter(sequence,modifications):
    modifications = modifications
    ms = reverse_list(list(sequence.keys()))
    for i in range(len(ms)-1):
        new_list = []
        for fb in sequence[ms[i]]:
            fb_seq = strip_func(fb[0],modifications)
            for fs in sequence[ms[i+1]]:
                fs_seq = strip_func(fs[0],modifications)
                if contains_aa(fb_seq,fs_seq):
                    if fs not in new_list:
                        new_list.append(fs)
        sequence[ms[i+1]] = new_list
    return sequence

def simplify_M(overlappingM):
    final = []
    for p in overlappingM:
        seq = p[0]
        seq = seq.replace("L","I")
        seq = seq.replace("Q","K")
        final.append((seq,p[1]))
    unique_sets = set(tuple(tpl) for tpl in final)
    unique_tuples = [tuple(s) for s in unique_sets]
    return unique_tuples

