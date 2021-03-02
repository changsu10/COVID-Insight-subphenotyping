# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:21:22 2020

@author: chs4001


Perform clustering analysis.
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
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from scaler import min_max_scaling, z_score_scaling
#from _clust_charaterization import two_clust_compare



import umap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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








## ---------- Addressing outliers ------------ #
## remove 
#outlier_df = data[
#                                (data['white_blood_cell_count'] > 500) | 
#                                (data['lymphocyte_count'] > 100) |
#                                # (data['bilirubin'] > 10) | 
#                                (data['aspartate_aminotransferase'] > 500) |
#                                (data['alanine_aminotransferase'] > 500) |
#                                (data['creatine_kinase'] > 5000) |
#                                # (data['prothrombin_time'] > 100) |
#                                (data['interleukin-6'] > 1000) | 
#                                (data['Ferritin'] > 15000) | 
#                                (data['D-dimer'] > 20000) |
#                                (data['procalcitonin'] > 100) |
#                                (data['cardiac_troponin_I'] > 20) |
#                                (data['cardiac_troponin_T'] > 5) |
#                                (data['LDH'] > 2500) |
#                                # (data['BUN'] > 300) |
#                                (data['GLUCOSE'] > 1000)
#                                ]
#
#outlier_df.to_csv(MAIN_DIR + '\\Data\\removed_outliers.csv', index=False)
#
#print(len(outlier_df))
#print(len(data))
#
#data = data[~data['ssid'].isin(outlier_df['ssid'].values.tolist())] 
#
#print(len(data))
#
## set boundary
##CLPPERs = {
##           'venous_lactate': {'upper':10, 'lower':0.1},
##           'creatinine': {'upper':10, 'lower':0.05},
##           'white_blood_cell_count': {'upper':25, 'lower':0.5},
##           'lymphocyte_count': {'upper':10, 'lower':0.04},
##           'platelet_count': {'upper':800, 'lower':1},
##           'bilirubin': {'upper':5, 'lower':0.01},
##           'aspartate_aminotransferase': {'upper':300, 'lower':1},
##           'alanine_aminotransferase': {'upper':300, 'lower':1},
##           'creatine_kinase': {'upper':2000, 'lower':1},
##           'prothrombin_time': {'upper':100, 'lower':0.1},
##           'interleukin-6': {'upper':500, 'lower':0.5},
##           'Ferritin': {'upper':5000, 'lower':5},
##           'D-dimer': {'upper':15000, 'lower':10},
##           'procalcitonin': {'upper':20, 'lower':0.01},
##           'C-reactive_protein': {'upper':400, 'lower':0.5},
##           'cardiac_troponin_I': {'upper':10, 'lower':0.01},
##           'cardiac_troponin_T': {'upper':2, 'lower':0.01},
##           'Neutrophils.band': {'upper':40, 'lower':0.02},
##            }
##for col in CLUSTERING_COLS:
##    if col in CLPPERs:
##        data[col] = data[col].clip(upper=CLPPERs[col]['upper'], lower=CLPPERs[col]['lower'])




# ------------ build cohort --------------- #
dev_site_data = data[data['site'].isin([1, 3, 4, 5])]

dev_data = dev_site_data[dev_site_data[split_method] == 'dev']
val_data = dev_site_data[dev_site_data[split_method] == 'val']

dev_study_data = dev_data[['ssid'] + CLUSTERING_COLS]
val_study_data = val_data[['ssid'] + CLUSTERING_COLS]

val_ind_data = data[data['site'].isin([8])]



# ---  report missing --- #
missing_report_df = pd.DataFrame(columns=['Variable'])
missing_report_df['Variable'] = ['patient number', 'age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS
# development
N_dev = len(dev_data)
missing_nums = dev_data[['age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS].isnull().sum().tolist()
missing_report = []
missing_report.append(N_dev)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N_dev))
missing_report_df['Development'] = missing_report
# validation cohort 
N_val = len(val_data)
missing_nums = val_data[['age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS].isnull().sum().tolist()
missing_report = []
missing_report.append(N_val)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N_val))
missing_report_df['Validation cohort'] = missing_report
# validation site
N_ind_val = len(val_ind_data)
missing_nums = val_ind_data[['age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS].isnull().sum().tolist()
missing_report = []
missing_report.append(N_ind_val)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N_ind_val))
missing_report_df['Validation site'] = missing_report
# entire cohort
N = len(data)
missing_nums = data[['age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS].isnull().sum().tolist()
missing_report = []
missing_report.append(N)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N))
missing_report_df['Total'] = missing_report
missing_report_df.to_csv(MAIN_DIR + '\\Data\\_study_cohorts_missing_report.csv', index=False)


# --- report data distribution --- #
dist_report_df = pd.DataFrame(columns=['Variable'])
sex_cols = list(set(data['sex_raw'].values))
race_cols = list(set(data['race'].values))
cols = ['patient number', 'age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS + sex_cols + race_cols 
dist_report_df['Variable'] = cols
# development cohort
distr_data = []
N_dev = len(dev_data)
distr_data.append(N_dev)
for col in ['age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS:
    if len(dev_data[col].dropna().values.tolist()) == 0:
        distr_data.append('-')
        continue
    col_median = np.nanmedian(dev_data[col].values)
    col_IRQ_25 = np.percentile(dev_data[col].dropna(), 25)
    col_IRQ_75 = np.percentile(dev_data[col].dropna(), 75)
    distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75))
for col in sex_cols: # sex
    n_col = len(dev_data[dev_data['sex_raw'] == col])
    distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/N_dev))    
for col in race_cols: # race
    n_col = len(dev_data[dev_data['race'] == col])
    distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/N_dev))    
dist_report_df['Development cohort'] = distr_data
# validation cohort
distr_data = []
N_val = len(val_data)
distr_data.append(N_val)
for col in ['age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS:
    if len(val_data[col].dropna().values.tolist()) == 0:
        distr_data.append('-')
        continue
    col_median = np.nanmedian(val_data[col].values)
    col_IRQ_25 = np.percentile(val_data[col].dropna(), 25)
    col_IRQ_75 = np.percentile(val_data[col].dropna(), 75)
    distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75))
for col in sex_cols: # sex
    n_col = len(val_data[val_data['sex_raw'] == col])
    distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/N_val))    
for col in race_cols: # race
    n_col = len(val_data[val_data['race'] == col])
    distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/N_val))    
dist_report_df['Validation cohort'] = distr_data
# validation site
distr_data = []
N_ind_val = len(val_ind_data)
distr_data.append(N_ind_val)
for col in ['age_at_confirm'] + TARGET_LABs + HIGH_MISSING_COLS:
    if len(val_ind_data[col].dropna().values.tolist()) == 0:
        distr_data.append('-')
        continue
    col_median = np.nanmedian(val_ind_data[col].values)
    col_IRQ_25 = np.percentile(val_ind_data[col].dropna(), 25)
    col_IRQ_75 = np.percentile(val_ind_data[col].dropna(), 75)
    distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75))
for col in sex_cols: # sex
    n_col = len(val_ind_data[val_ind_data['sex_raw'] == col])
    distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/N_ind_val))    
for col in race_cols: # race
    n_col = len(val_ind_data[val_ind_data['race'] == col])
    distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/N_ind_val))    
dist_report_df['Validation site'] = distr_data

dist_report_df.to_csv(MAIN_DIR + '\\Data\\_study_cohorts_data_distribution.csv', index=False)




for col in CLUSTERING_COLS:
    print(col, np.nanmin(data[col].values), np.percentile(data[col].dropna().values, 0.1), len(data[data[col] == 0]))

dev_study_data.to_csv(MAIN_DIR + '\\Data\\dev_scaled_1.csv', index=False)









#################### ------ development ----- ############################
  
# ----------- Log transform ----------------- #
for col in LOG_CLOS:
#    dev_study_data[col] = dev_study_data[col].replace({0: np.nanmedian(dev_study_data[col]) / 100})
    dev_study_data[col] = dev_study_data[col].replace({0: np.nan})
    dev_study_data[col] = np.log(dev_study_data[col].values)
    
dev_study_data.to_csv(MAIN_DIR + '\\Data\\dev_scaled_2.csv', index=False)
for col in CLUSTERING_COLS:
    value = dev_study_data[col]
    print(col, np.nanmedian(value), '--', np.percentile(value.dropna(), 5), np.percentile(value.dropna(), 95), value.quantile(0.95))
    plt.hist(value, bins=100)
    plt.show()
    plt.close()
#    data[col] = data[col].clip(upper=data[col].quantile(0.95))
    
    

## ----------- median imputation ------------- #
#imputer = SimpleImputer(strategy='median')
#dev_study_data[CLUSTERING_COLS] = imputer.fit_transform(dev_study_data[CLUSTERING_COLS])


#dev_study_data.to_csv(MAIN_DIR + '\\Data\\dev_scaled.csv', index=False)


# ----------- scaling ------------- #
for col in CLUSTERING_COLS:
#    dev_study_data[col] = min_max_scaling(dev_study_data[col].values)
    dev_study_data[col] = z_score_scaling(dev_study_data[col].values)
dev_study_data.to_csv(MAIN_DIR + '\\Data\\dev_scaled_3.csv', index=False)
#scaler = preprocessing.StandardScaler()
#scaler = preprocessing.MinMaxScaler()    
#dev_study_data[CLUSTERING_COLS] = scaler.fit_transform(dev_study_data[CLUSTERING_COLS])

#for col in CLUSTERING_COLS:
#    value = dev_study_data[col]
#    print('------ %s -------' % col)
#    plt.hist(value, bins=100)
#    plt.show()
#    plt.close()


# ----------- KNN imputation -------------- #
imputer = KNNImputer(n_neighbors=10, weights='distance')
#imputer = load(MAIN_DIR + '\\Results\\dev_knnimputer.joblib')
dev_study_data[CLUSTERING_COLS] = imputer.fit_transform(dev_study_data[CLUSTERING_COLS])
dev_study_data.to_csv(MAIN_DIR + '\\Data\\dev_scaled.csv', index=False)
dump(imputer, MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\dev_knnimputer.joblib')

#for col in CLUSTERING_COLS:
#    value = dev_study_data[col]
#    print('------ %s imputed -------' % col)
#    plt.hist(value, bins=100)
#    plt.show()
#    plt.close()
    
dev_study_data.to_csv(MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\dev_scaled.csv', index=False)








# -------------- umap -------------------- #
#umpa_reducer = umap.UMAP(n_neighbors=5, min_dist=0.35, random_state=0) # 0, 1
umpa_reducer = umap.UMAP(n_neighbors=10, min_dist=0.35, random_state=42) # 0, 1
X_umap = umpa_reducer.fit_transform(dev_study_data[CLUSTERING_COLS])
#X_umap = umpa_reducer.fit_transform(X_pca)
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1, alpha=0.5)
plt.show()
plt.close()

## ------------ clustering ---------------- #
dist_mtx = euclidean_distances(dev_study_data[CLUSTERING_COLS].values)
#dist_mtx = euclidean_distances(X_pca)
linkage = hc.linkage(sp.distance.squareform(dist_mtx, checks=False), method='ward')
ns_plot = sns.clustermap(dist_mtx, row_linkage=linkage, col_linkage=linkage)
plt.savefig(MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\clustergram.png', dpi=300)
plt.close()

plt.figure(figsize=[8, 6])
hc.set_link_color_palette(['#CE4257', '#F9C77E', '#79A3D9', '#7B967A']) 
d_plot = hc.dendrogram(linkage, orientation='top', color_threshold=100, above_threshold_color='#808080')
plt.savefig(MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\dendrogram.pdf', dpi=300)
plt.close()



C = 4
labels = fcluster(linkage, C, criterion='maxclust')
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
        3:1,
        4:2,
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
plt.savefig(MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\umap_with_label.pdf', dpi=300)
plt.show()
plt.close()




# add normalized confirm time
SCREEN_DATE = datetime.datetime.strptime('2020-03-01', '%Y-%m-%d')
dev_data['confirm_date'] = pd.to_datetime(dev_data['confirm_date'], format='%Y-%m-%d')
normalized_confrim_date_list = []
for idx, row in dev_data.iterrows():
    confirm_date = row['confirm_date']
    normalized_confrim_date_list.append((confirm_date - SCREEN_DATE).days)
dev_data['norm_confirm_date'] = normalized_confrim_date_list


# progression overtime
data_time = []
max_time = 7 * 15
n_gap = int(max_time / 7)
bins = [i * 7 for i in range(0, n_gap+1)]

for c in range(1, C+1): # grouped bar plot version
    target_df = dev_data[dev_data['label'] == c]
    n_c = len(target_df)
  
    distribution = plt.hist(target_df['norm_confirm_date'].values, bins=bins)
    
    data_time.append(distribution[0])   
    
data_time = np.array(data_time)

data_time = data_time / data_time.sum(axis=0)

data_time = data_time[:, :-1]

total_dist = plt.hist(dev_data['norm_confirm_date'].values, bins=bins)[0]
total_dist = total_dist[:-1]
total_dist = total_dist / len(dev_data)

plt.close()      
#plt.plot(range(1, n_gap+1), data_time[0])
#plt.plot(range(1, n_gap+1), data_time[1])
#plt.plot(range(1, n_gap+1), data_time[2])
#plt.plot(range(1, n_gap+1), data_time[3])

fig = plt.figure(figsize=[10, 6])
fig.subplots_adjust(left=0.08, right=0.8, top=0.95, bottom=0.25, hspace=0.05)
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 0.5])
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])
ax1.bar(range(1, n_gap), data_time[0], color=lable_color[1], label=lable_annotation[1], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[1], bottom=data_time[0], color=lable_color[2], label=lable_annotation[2], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[2], bottom=data_time[0]+data_time[1], color=lable_color[3], label=lable_annotation[3], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[3], bottom=data_time[0]+data_time[1]+data_time[2], color=lable_color[4], label=lable_annotation[4], width=1, edgecolor='white', linewidth=1)
ax1.set_xticks([])
ax1.set_yticks([0, .25, .5, .75, 1])
ax1.set_yticklabels([0, 25, 50, 75, 100])
ax1.legend(bbox_to_anchor = (1.275, 0.36), loc='upper right', fontsize=11)
ax1.set_ylabel('Proportion within week (%)', fontsize=12)

ax2.bar(range(1, n_gap), total_dist, color='gray', width=1, edgecolor='white', linewidth=1)
ax2.set_ylim(0, .3)
ax2.set_xticks(range(1, n_gap))
ax2.set_xticklabels(['Week %d' % i for i in range(1, n_gap)], rotation=45)
ax2.set_yticks([0, .1, .2, .3])
ax2.set_yticklabels([0, 10, 20, 30])
ax2.set_ylabel('Proportion (%)', fontsize=12)
ax2.set_xlabel('Time since Mar 1, 2020', fontsize=12)

#plt.xticks(range(1, n_gap), ['Week %d' % i for i in range(1, n_gap)], rotation=45)
#plt.yticks([0, .25, .5, .75, 1], [0, 25, 50, 75, 100])
#plt.xlabel('Time since Mar 1, 2020', fontsize=12)
#plt.ylabel('Proportion of patients (%)', fontsize=12)
#plt.legend(bbox_to_anchor = (1.275, 0.35), loc='upper right', fontsize=11)
#plt.show()

plt.savefig(MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\confirm_time_bar_by_week.pdf')









# ----- SDoH clust <=> subphenotypes -------- #
# load SDoH clust label
SDoH_version = 'devlopment'  # devlopment, all
SDoH_label = pd.read_csv(MAIN_DIR + '\\Results\\_SDoH_clust\\' + SDoH_version +'\\_SDoH_label.csv', header=0)


#SDoH_label = pd.read_csv(MAIN_DIR + '\\Results\\_SDoH_label.csv', header=0)
dev_data_with_SDoH_clust = pd.merge(dev_data, SDoH_label, how='left', on='ssid')

import _clust_SDoH_clust_characterization
report_df = pd.DataFrame()
report_df = _clust_SDoH_clust_characterization.clust_SDoH_clust_cnt(data=dev_data_with_SDoH_clust,
                                                        clust_col='label', 
                                                        SDoH_clust_col='SDoH_clust'
                                                        )
for col in ['death_within_60d']: # + MED_COLS:
    temp_df = _clust_SDoH_clust_characterization.clust_SDoH_clust_outcome(data=dev_data_with_SDoH_clust,
                                                         clust_col='label', 
                                                         SDoH_clust_col='SDoH_clust', 
                                                         outcome_col=col)    
    report_df = pd.concat([report_df, temp_df])
report_df.to_csv(MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\characteristic_clust_SDoH_clust.csv', index=False)

#_clust_SDoH_clust_characterization.clust_SDoH_clust_outcome_change_bar(data=dev_data_with_SDoH_clust,
#                                                         clust_col='label', 
#                                                         SDoH_clust_col='SDoH_clust',
#                                                         save_dir = MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\SDoH_vs_subphenotype\\'
#                                                         )

_clust_SDoH_clust_characterization.clust_SDoH_clust_outcome_change_bar(data=dev_data_with_SDoH_clust,
                                                         clust_col='label', 
                                                         SDoH_clust_col='SDoH_clust',
                                                         save_dir = MAIN_DIR + '\\Results\\' + OUTPUT_FOLDER + '\\SDoH_vs_subphenotype\\'
                                                         )

    
    










