# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 19:49:59 2020

@author: chs4001
"""


import pandas as pd
import numpy as np

from scipy.stats import ttest_ind, mannwhitneyu, shapiro, chi2_contingency, f_oneway, kruskal

import matplotlib.pyplot as plt


import os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass



def clust_SDoH_clust_outcome_change_bar(data, clust_col, SDoH_clust_col, save_dir):

    mkdir(save_dir)
    
    outcome_cols = [
                    # 'death', 
                    # 'is_icu', 'is_vent',
                    # 
                    'death_within_60d',
                    'icu_60d',
                    'vent_60d', 
                    'corticosteroids', 'enoxaparin', 'heparin',
                    'antibiotics', 'Hydroxychloroquine', 'Vasopressor',
                    ]
    
#    outcome_cols = {
#            'death_within_60d': 'Mortality, 60 day',
#            'icu_60d': 'ICU admission, 60 day',
#            'vent_60d': 'Mechanical ventilation, 60 day', 
#            
#            'heparin': 'Heparin medications',
#            'enoxaparin': 'Enoxaparin medications', 
#            
#            'antibiotics': 'Antibiotics', 
#            'corticosteroids': 'Corticosteroids',        
#            
#            'Vasopressor': 'Vasopressor medication',
#            'Hydroxychloroquine': 'Hydroxychloroquine',
#            }
    
    
    clust_labels = list(set(data[clust_col].values))
    clust_labels.sort()
    
    SDoH_clust_labels = list(set(data[SDoH_clust_col].dropna().values))
    SDoH_clust_labels.sort()  
    
    
    # fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    # fig = plt.figure(figsize=(12, 12))
    
    for i in range(len(outcome_cols)):
        col = outcome_cols[i]
        
        n_col = int(i % 3)
        n_row = int(i / 3)
        
        
        ref_ratios = []
        SDoH_clust_ratios = []
        SDoH_clust_raw = []
        for c in clust_labels:
            
            clust_data = data[data[clust_col] == c]
            
            N_c = len(clust_data)
            
            N_c_oc = len(clust_data[clust_data[col] == 1])
            
            ref_value = 100 * N_c_oc / N_c
            
            ref_ratios.append(ref_value)
            
            SDoH_clust_values = []
            
            clust_death_data = clust_data[clust_data[col] == 1]
            
#            print(c, '---')
#            print(N_c, N_c_oc)
#            print('ref', ref_value)
            
            for sc in SDoH_clust_labels:
                N_sc_c = len(clust_data[clust_data[SDoH_clust_col] == sc])
                
                N_sc_positive = len(clust_death_data[clust_death_data[SDoH_clust_col] == sc])
                N_sc_positive_value = 100 * N_sc_positive / len(clust_data[clust_data[SDoH_clust_col] == sc])
                
#                print(N_sc_c)
#                print(N_sc_positive)
#                print(N_sc_positive_value)
                SDoH_clust_values.append(N_sc_positive_value)
                        
            SDoH_clust_raw.append(SDoH_clust_values)
            
#            SDoH_clust_values = (np.array(SDoH_clust_values) - ref_value) / ref_value
            SDoH_clust_values = np.array(SDoH_clust_values) - ref_value
                        
            SDoH_clust_ratios.append(SDoH_clust_values)
        
        SDoH_clust_raw = np.array(SDoH_clust_raw)
        SDoH_clust_raw = np.round(SDoH_clust_raw, decimals=2)
        SDoH_clust_ratios = np.array(SDoH_clust_ratios)

        if col in [
                    'is_icu', 'is_vent', 
                    
                    'icu_60d',
                    'vent_60d',
                    ]:
            vmin=-10
            vmax=10
            
        elif col in['death', 'death_within_60d']:
            vmin=-7
            vmax=7
        else:
            vmin=-25
            vmax=20

        fig, ax = plt.subplots(figsize=(8, 3))
        labels = ['I', 'II', 'III', 'IV']
        barwidth = 0.2
        r1 = np.arange(4)
        r2 = [x + barwidth for x in r1]
        r3 = [x + barwidth for x in r2]
        plt.bar(r1, SDoH_clust_ratios[:, 0], width=barwidth, label='SDoH Level 1', color='#59405C', alpha=1)
        plt.bar(r2, SDoH_clust_ratios[:, 1], width=barwidth, label='SDoH Level 2', color='#87556F', alpha=0.7)  
        plt.bar(r3, SDoH_clust_ratios[:, 2], width=barwidth, label='SDoH Level 3', color='#A67A91', alpha=0.5)

        plt.axhline(y=0, color='k', lw=1)
        plt.grid(axis='y', color='gainsboro')

        plt.ylim(vmin, vmax)
        plt.xticks([r + barwidth for r in range(4)], labels)
        plt.xlabel('Subphenotypes')
        plt.ylabel('Difference of proportion')
        plt.title(col)
        plt.legend(loc='lower left', fontsize=10)

        plt.subplots_adjust(bottom=0.2)
        
        plt.savefig(save_dir + col + '_bar.pdf')
    








