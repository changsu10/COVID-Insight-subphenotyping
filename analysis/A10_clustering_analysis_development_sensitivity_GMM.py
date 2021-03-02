# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:21:22 2020

@author: chs4001


Perform clustering analysis on validation cohort.

Sensitivity analysis.

"""

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
import sklearn.mixture as mixture
import scipy

from scaler import min_max_scaling, z_score_scaling
#from _clust_charaterization import two_clust_compare


import umap

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

from joblib import dump, load

import pickle as pkl


TARGET_LABs = [
        'albumin',
        'alanine_aminotransferase', 
        'aspartate_aminotransferase', 
        'Bicarbonate', 
        'bilirubin', 
        'BUN',
        'creatine_kinase',
        'creatinine', 
        'CHLORIDE',
        'C-reactive_protein', 
        'D-dimer',
        'ESR', 
        'Ferritin', 
        'GLUCOSE', 
        'HEMOGLOBIN', 
        'interleukin-6', 
        'venous_lactate', 
        'LDH',  
        'lymphocyte_count', 
        'neutrophil_count', 
        'Neutrophils.band',
        'Oxygen_saturation',
        'procalcitonin', 
        'prothrombin_time',  
        'platelet_count', 
        'red_blood_cell_distribution_width', 
        'SODIUM', 
        'cardiac_troponin_I', 
        'cardiac_troponin_T',
        'white_blood_cell_count'
        ]


LOG_CLOS = [
        'alanine_aminotransferase', 
        'aspartate_aminotransferase', 
        'bilirubin', 
        'BUN',
        'creatine_kinase', 
        'creatinine', 
        'CHLORIDE',
        'C-reactive_protein',
        'D-dimer', 
        'Ferritin',
        'GLUCOSE',  
        'HEMOGLOBIN',
        'interleukin-6',
        'venous_lactate',
        'LDH',  
        'lymphocyte_count', 
        'Neutrophils.band',
        'procalcitonin',
        'prothrombin_time',
        'SODIUM', 
        'cardiac_troponin_I',
        'cardiac_troponin_T', 
        'white_blood_cell_count'
        ]

HIGH_MISSING_COLS = [
        'interleukin-6', 
        'D-dimer',
        'cardiac_troponin_I',
        'procalcitonin',
#        'cardiac_troponin_T',
        'ESR',
        'Neutrophils.band',
        'Oxygen_saturation',
        ]

TARGET_LABs = [var for var in TARGET_LABs if var not in HIGH_MISSING_COLS]
LOG_CLOS = [var for var in LOG_CLOS if var not in HIGH_MISSING_COLS]


#CLUSTERING_COLS = ['age_at_confirm'] + TARGET_LABs
CLUSTERING_COLS = TARGET_LABs


#SCALING_COLS = ['age_at_confirm'] + TARGET_LABs
SCALING_COLS = TARGET_LABs




MAIN_DIR = "W:\\WorkArea-chs4001"
FILE_NAME = "_demographics_deduplicated_labs_1006_window_-3_14_droped_full_missing.csv"
input_file = MAIN_DIR + '\\Data\\' + FILE_NAME

OUTPUT_FOLDER = '_development'

#split_method = 'split'
split_method = 'split_73'



# load data
data = pd.read_csv(input_file, header=0)


# # load confirm type: dx or lab
# SITES = [1, 3, 4, 5, 8]
# # SITES = [1]
# dx_only_confimred_patient = {}
# for site in SITES:
#     site_df_dx = pd.read_csv(MAIN_DIR + '\\Data\\cohort\\' + 'DxOnly_Site_%s.csv' % site)
#     dx_list = site_df_dx['ssid'].values.tolist()
#     site_df_lab = pd.read_csv(MAIN_DIR + '\\Data\\cohort\\' + 'LabOnly_Site_%s.csv' % site)
#     lab_list = site_df_lab['ssid'].values.tolist()
#     i = 0
#     for p in dx_list:
#         if p not in lab_list:
#             dx_only_confimred_patient[p] = True
#         else:
#             i+=1



# ------------ build cohort --------------- #
dev_site_data = data[data['site'].isin([1, 3, 4, 5])]

dev_data = dev_site_data[dev_site_data[split_method] == 'dev']
val_data = dev_site_data[dev_site_data[split_method] == 'val']

dev_study_data = dev_data[['ssid'] + CLUSTERING_COLS]

# # drop patients who were confirmed only by dx codes
# print(dev_study_data.shape)
# dev_study_data = dev_study_data[~dev_study_data['ssid'].isin(dx_only_confimred_patient)]
# print(dev_study_data.shape)


# for col in CLUSTERING_COLS:
#     print(col, np.nanmin(data[col].values), np.percentile(data[col].dropna().values, 0.1), len(data[data[col] == 0]))

# dev_study_data.to_csv(MAIN_DIR + '\\Data\\dev_scaled_1.csv', index=False)


# correlation
corr_mtx = dev_study_data[CLUSTERING_COLS].corr(method='spearman')
mask = np.triu(np.ones_like(corr_mtx, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
#cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_mtx, mask=mask, 
            #cmap=cmap, 
            cmap='RdBu_r',
            vmax=.7, center=0, vmin=-.7, 
            square=True, linewidths=1, cbar_kws={"shrink":.5})
plt.savefig(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\' + 'feature_corr.png')
plt.close()

corr_mtx.to_csv(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\' + 'feature_corr.csv', index=True)

#highly_corr_cols = ['aspartate_aminotransferase', 
##                    'LDH', 
#                    'BUN', 
#                    'cardiac_troponin_T', 
##                    'SODIUM', 
#                    'neutrophil_count']

# highly_corr_cols = [
#         'interleukin-6', 
#         'D-dimer',
#         'cardiac_troponin_I',
#         'procalcitonin',
# #        'cardiac_troponin_T',
#         'ESR',
#         'Neutrophils.band',
#         'Oxygen_saturation',
#         ]   
                 
highly_corr_cols = []

sens_CLUSTERING_COLS = [var for var in CLUSTERING_COLS if var not in highly_corr_cols]
sens_LOG_CLOS = [var for var in LOG_CLOS if var not in highly_corr_cols]

dev_study_data = dev_study_data[['ssid'] + sens_CLUSTERING_COLS]


#################### ------ development sensitivity ----- ############################
OUTPUT_FOLDER = '_development_sensitivity_GMM'
# ----------- Log transform ----------------- #
for col in sens_LOG_CLOS:
#    dev_study_data[col] = dev_study_data[col].replace({0: np.nanmedian(dev_study_data[col]) / 100})
    dev_study_data[col] = dev_study_data[col].replace({0: np.nan})
    dev_study_data[col] = np.log(dev_study_data[col].values)
    
dev_study_data.to_csv(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\dev_log_scaled.csv', index=False)

    

## ----------- median imputation ------------- #
#imputer = SimpleImputer(strategy='median')
#dev_study_data[sens_CLUSTERING_COLS] = imputer.fit_transform(dev_study_data[sens_CLUSTERING_COLS])
#
#dev_study_data.to_csv(MAIN_DIR + '\\Data\\dev_scaled_sens.csv', index=False)


# ----------- scaling ------------- #
for col in sens_CLUSTERING_COLS:
    dev_study_data[col] = z_score_scaling(dev_study_data[col].values)
dev_study_data.to_csv(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\dev_log_scaled_norm.csv', index=False)
##scaler = preprocessing.StandardScaler()
##scaler = preprocessing.MinMaxScaler()    
##dev_study_data[CLUSTERING_COLS] = scaler.fit_transform(dev_study_data[CLUSTERING_COLS])

#for col in sens_CLUSTERING_COLS:
#    value = dev_study_data[col]
#    print('------ %s -------' % col)
#    plt.hist(value, bins=100)
#    plt.show()
#    plt.close()




 # ----------- KNN imputation -------------- #
imputer = KNNImputer(n_neighbors=10, weights='distance')
#imputer = load(MAIN_DIR + '\\Results\\dev_knnimputer.joblib')
dev_study_data[sens_CLUSTERING_COLS] = imputer.fit_transform(dev_study_data[sens_CLUSTERING_COLS])
dev_study_data.to_csv(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\dev_scaled.csv', index=False)
dump(imputer, MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\dev_knnimputer.joblib')

# #for col in CLUSTERING_COLS:
# #    value = dev_study_data[col]
# #    print('------ %s imputed -------' % col)
# #    plt.hist(value, bins=100)
# #    plt.show()
# #    plt.close()
    


## ----------- removing outlier ------------- #
#abs_z_scores = np.abs(dev_study_data[sens_CLUSTERING_COLS].values)
#filtered_entiries = (abs_z_scores < 5).all(axis=1)
#dev_study_data = dev_study_data[filtered_entiries]






# -------------- umap -------------------- #
#umpa_reducer = umap.UMAP(n_neighbors=5, min_dist=0.35, random_state=0) # 0, 1
#umpa_reducer = umap.UMAP(n_neighbors=200, min_dist=0, random_state=1) # 0, 1
umpa_reducer = umap.UMAP(n_neighbors=10, min_dist=0.35, random_state=42) # 0, 1
X_umap = umpa_reducer.fit_transform(dev_study_data[sens_CLUSTERING_COLS])
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1, alpha=0.5)
plt.show()
plt.close()

## ------------ clustering ---------------- #
C = 4
gmm = mixture.GaussianMixture(n_components = C, covariance_type = 'spherical', random_state = 3) # spherical  tied  diag  full 1
X = dev_study_data[sens_CLUSTERING_COLS].values#[0:100, :]
gmm.fit(X)
labels = gmm.predict(X) + 1



lable_color = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}
               
lable_annotation = {1:'Subphenotype I',
                    2:'Subphenotype II',
                    3:'Subphenotype III',
                    4:'Subphenotype IV',
                    }             
               
               
# rename labels
clust_label_map = {
    1:4,
    2:3,
    3:2,
    4:1,
        }
new_labels = []
for i in labels:
    new_labels.append(clust_label_map[i])
labels = np.array(new_labels)

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.08, right=0.7, top=0.98, bottom=0.05)
for c in range(1, C+1):
    plt.scatter(X_umap[labels==c, 0], X_umap[labels==c, 1], s=1, alpha=.85, color=lable_color[c], label = lable_annotation[c])
    
plt.axis('off')
plt.legend(bbox_to_anchor = (1.4, 0.25), loc='upper right', fontsize=10, markerscale=4)
plt.savefig(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\umap_with_label.pdf', dpi=300)
plt.show()
plt.close()




# ------------ clustering statistics---------------- #
Cs = [2, 3, 4, 5, 6, 7, 8]
BIC_list = []
AIC_list = []
probs = []
for c in Cs:
    gmm_c = mixture.GaussianMixture(n_components = c, covariance_type = 'spherical', random_state = 3)
    X = dev_study_data[sens_CLUSTERING_COLS].values#[0:100, :]
    gmm_c.fit(X)

    BIC_list.append(gmm_c.bic(X))
    AIC_list.append(gmm_c.aic(X))
    
    pred = gmm_c.predict_proba(X)
  
    memb_probs = []
    for i in range(len(pred)):
        memb_probs.append(np.max(pred[i, :]) * 100)        
    c_prob_str = '%.2f [%.2f - %.2f]' % (np.median(memb_probs), np.percentile(memb_probs, 25), np.percentile(memb_probs, 75))   
    probs.append(c_prob_str)

stat_report_df = pd.DataFrame()
stat_report_df['Cluster number'] = Cs
stat_report_df['BIC'] = BIC_list
stat_report_df['AIC'] = AIC_list
stat_report_df['Membership probability, Median [IQR]'] = probs
stat_report_df.to_csv(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\clustering_statistics.csv', index=False)

plt.plot(Cs, BIC_list)
plt.plot(Cs, AIC_list)
plt.show()







#
# ---------- Characteristics ------------- #
# add label
#dev_site_data = data[data['site'].isin([1, 3, 4, 5])]
#dev_data = dev_site_data[dev_site_data[split_method] == 'dev']

dev_study_data['label'] = labels
dev_data = pd.merge(dev_study_data[['ssid', 'label']], data, how='left', on='ssid')


#dev_data = pd.merge()


dev_data['umap_0'] = X_umap[:, 0]
dev_data['umap_1'] = X_umap[:, 1]

dev_data[['ssid', 'label', 'umap_0', 'umap_1']].to_csv(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\_subphenotype_labels.csv', index=False)





# -------- cluster comparison ---------#
# load development phenotype labels
from sklearn.metrics import confusion_matrix
old_dev_lable_df = pd.read_csv(MAIN_DIR + '\\Results\\_development\\_subphenotype_labels.csv', header=0)
old_dev_lable_df = old_dev_lable_df.rename(columns={'label': 'label_dev'})
old_dev_lable_df = pd.merge(dev_data[['ssid', 'label']], old_dev_lable_df, how='left', on='ssid')

clust_labels = [0, 1, 2, 3]
clust_label_str = ['I', 'II', 'III', 'IV']

conf_mtx = confusion_matrix(old_dev_lable_df['label_dev'].values, old_dev_lable_df['label'].values, normalize='true')
#plt.figure(figsize=[5, 5])
plt.imshow(conf_mtx, cmap='Reds',
           vmin=0, vmax=1
           )
cbar = plt.colorbar()
cbar.set_label('Proportion', fontsize=12)
plt.xticks(clust_labels, clust_label_str)
plt.yticks(clust_labels, clust_label_str)
plt.ylabel('Development subphenotypes', fontsize=12)
plt.xlabel('Re-derived subphenotypes', fontsize=12)
plt.savefig(MAIN_DIR + '\\Results\\_development_sensitivity_GMM\\clust_comparison.pdf')
plt.show()
plt.close()


















