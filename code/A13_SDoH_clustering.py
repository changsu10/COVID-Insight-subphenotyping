# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:47:35 2020


Perform clustering analysis on SDoH data


@author: chs4001
"""

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import fcluster
from scaler import min_max_scaling, z_score_scaling


import umap

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

import os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)







MAIN_DIR = "W:\\WorkArea-chs4001"
FILE_NAME = "_demographics_deduplicated_labs_1006_window_-3_14_droped_full_missing.csv"
input_file = MAIN_DIR + '\\Data\\' + FILE_NAME

SDoH_COLS_NEW = ['median_household_income', 'p_no_highschool', 'p_essential_occup', 
             'p_nonwhite', 'p_unemployed', 'p_household_over_1_perroom']

#split_method = 'split'
split_method = 'split_73'







# load data
data = pd.read_csv(input_file, header=0)

# load SDoH data
all_patient_SDoH_df = pd.read_csv(MAIN_DIR + '\\Data\\' + FILE_NAME[:-4] + '_SDoH.csv', header=0)
print(all_patient_SDoH_df.shape)

all_patient_SDoH_df = pd.merge(data, all_patient_SDoH_df, how='left', on='ssid')
print(all_patient_SDoH_df.shape)

# drop nan
all_patient_SDoH_df = all_patient_SDoH_df.dropna(subset=SDoH_COLS_NEW, how='all')
print(all_patient_SDoH_df.shape)


# cohort
dev_site_data = all_patient_SDoH_df[all_patient_SDoH_df['site'].isin([1, 3, 4, 5])]

dev_data = dev_site_data[dev_site_data[split_method] == 'dev']
val_data = dev_site_data[dev_site_data[split_method] == 'val']

val_ind_data = all_patient_SDoH_df[all_patient_SDoH_df['site'].isin([8])]



# check missing
missing_report_df = pd.DataFrame(columns=['Variable'])
missing_report_df['Variable'] = ['patient number'] + SDoH_COLS_NEW
# development
N_dev = len(dev_data)
missing_nums = dev_data[SDoH_COLS_NEW].isnull().sum().tolist()
missing_report = []
missing_report.append(N_dev)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N_dev))
missing_report_df['Development'] = missing_report
# validation cohort 
N_val = len(val_data)
missing_nums = val_data[SDoH_COLS_NEW].isnull().sum().tolist()
missing_report = []
missing_report.append(N_val)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N_val))
missing_report_df['Validation cohort'] = missing_report
# validation site
N_ind_val = len(val_ind_data)
missing_nums = val_ind_data[SDoH_COLS_NEW].isnull().sum().tolist()
missing_report = []
missing_report.append(N_ind_val)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N_ind_val))
missing_report_df['Validation site'] = missing_report
# entire cohort
N = len(all_patient_SDoH_df)
missing_nums = all_patient_SDoH_df[SDoH_COLS_NEW].isnull().sum().tolist()
missing_report = []
missing_report.append(N)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N))
missing_report_df['Total'] = missing_report
missing_report_df.to_csv(MAIN_DIR + '\\Data\\_study_cohorts_missing_report_SDoH.csv', index=False)





# --- report data distribution --- #
dist_report_df = pd.DataFrame(columns=['Variable'])
cols = ['patient number'] + SDoH_COLS_NEW
dist_report_df['Variable'] = cols
# development cohort
distr_data = []
N_dev = len(dev_data)
distr_data.append(N_dev)
for col in SDoH_COLS_NEW:
    if len(dev_data[col].dropna().values.tolist()) == 0:
        distr_data.append('-')
        continue
    col_median = np.nanmedian(dev_data[col].values)
    col_IRQ_25 = np.percentile(dev_data[col].dropna(), 25)
    col_IRQ_75 = np.percentile(dev_data[col].dropna(), 75)
    distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75))   
dist_report_df['Development cohort'] = distr_data
# validation cohort
distr_data = []
N_val = len(val_data)
distr_data.append(N_val)
for col in SDoH_COLS_NEW:
    if len(val_data[col].dropna().values.tolist()) == 0:
        distr_data.append('-')
        continue
    col_median = np.nanmedian(val_data[col].values)
    col_IRQ_25 = np.percentile(val_data[col].dropna(), 25)
    col_IRQ_75 = np.percentile(val_data[col].dropna(), 75)
    distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75))   
dist_report_df['Validation cohort'] = distr_data
# validation site
distr_data = []
N_ind_val = len(val_ind_data)
distr_data.append(N_ind_val)
for col in SDoH_COLS_NEW:
    if len(val_ind_data[col].dropna().values.tolist()) == 0:
        distr_data.append('-')
        continue
    col_median = np.nanmedian(val_ind_data[col].values)
    col_IRQ_25 = np.percentile(val_ind_data[col].dropna(), 25)
    col_IRQ_75 = np.percentile(val_ind_data[col].dropna(), 75)
    distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75)) 
dist_report_df['Validation site'] = distr_data
# entire cohort
distr_data = []
N = len(all_patient_SDoH_df)
distr_data.append(N)
for col in SDoH_COLS_NEW:
    if len(all_patient_SDoH_df[col].dropna().values.tolist()) == 0:
        distr_data.append('-')
        continue
    col_median = np.nanmedian(all_patient_SDoH_df[col].values)
    col_IRQ_25 = np.percentile(all_patient_SDoH_df[col].dropna(), 25)
    col_IRQ_75 = np.percentile(all_patient_SDoH_df[col].dropna(), 75)
    distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75)) 
dist_report_df['All'] = distr_data

dist_report_df.to_csv(MAIN_DIR + '\\Data\\_study_cohorts_data_distribution_SDoH.csv', index=False)







study = 'devlopment' 
#study = 'validation' 
#study = 'validation_site' 

#study = 'all' 

save_dir = MAIN_DIR + '\\Results\\_SDoH_clust\\' + study + '\\'
mkdir(save_dir)

# ------------------- clustering --------------------- #
if study == 'devlopment':
    analysis_data = dev_data#[['ssid'] + SDoH_COLS_NEW]
elif study == 'validation':
    analysis_data = val_data#[['ssid'] + SDoH_COLS_NEW]
elif study == 'validation_site':
    analysis_data = val_ind_data#[['ssid'] + SDoH_COLS_NEW]
elif study == 'all':
    analysis_data = all_patient_SDoH_df#[['ssid'] + SDoH_COLS_NEW]
else:
    print('Please specify correct study cohort!!!!!')



#analysis_data = all_patient_SDoH_df[['ssid'] + SDoH_COLS_NEW]
# scaling
scaler = preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(analysis_data[SDoH_COLS_NEW])
# imputation
imputer = KNNImputer(n_neighbors=10, weights='distance')
data_scaled_imputed = imputer.fit_transform(data_scaled)
## umap
#umpa_reducer = umap.UMAP(n_neighbors=10, min_dist=0.35, random_state=42) # 0, 1
#X_umap = umpa_reducer.fit_transform(analysis_data[SDoH_COLS_NEW])
##X_umap = umpa_reducer.fit_transform(X_pca)
#plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1, alpha=0.5)
#plt.show()
#plt.close()


data_scaled_imputed_df = pd.DataFrame(data_scaled_imputed, columns=SDoH_COLS_NEW)
data_scaled_imputed_df['ssid']  = analysis_data['ssid'].values
data_scaled_imputed_df.to_csv(save_dir + 'dev_data_SDoH.csv', index=False)

## ------------ clustering ---------------- #
dist_mtx = euclidean_distances(data_scaled_imputed)
#dist_mtx = euclidean_distances(X_pca)
linkage = hc.linkage(sp.distance.squareform(dist_mtx, checks=False), method='ward')
ns_plot = sns.clustermap(dist_mtx, row_linkage=linkage, col_linkage=linkage)
plt.savefig(save_dir + '_SDoH_clustergram.png', dpi=300)
plt.close()

plt.figure(figsize=[8, 6])
hc.set_link_color_palette(['#330066', '#7F00FF', '#CC99FF', 'k'])
d_plot = hc.dendrogram(linkage, orientation='top', color_threshold=100, above_threshold_color='#808080')
plt.savefig(save_dir + '_SDoH_dendrogram.pdf')
plt.close()


C = 3
labels = fcluster(linkage, C, criterion='maxclust')

# rename labels
clust_label_map = {
        1:1,
        2:2,
        3:3,
        }
new_labels = []
for i in labels:
    new_labels.append(clust_label_map[i])
labels = np.array(new_labels)


analysis_data['SDoH_clust'] = labels
label_df = analysis_data[['ssid', 'SDoH_clust']]
label_df.to_csv(save_dir + '_SDoH_label.csv', index=False)



# ------------- cluster characteristics ----------------- #
# load outcome comorbidities
outcome_comorb_df = pd.read_csv(MAIN_DIR + '\\Data\\outcome_comorbidity_.csv', header=0).drop(columns=['sex', 'age_at_confirm', 'hispanic', 'race', 'site', 'facilityid'])
analysis_data = pd.merge(analysis_data, outcome_comorb_df, how='left', on='ssid')

# load icu ventilation outcome
icu_outcome_df = pd.read_csv(MAIN_DIR + '\\Data\\icu_vent.csv', header=0).drop(columns=['CONFIRM_DATE'])
analysis_data = pd.merge(analysis_data, icu_outcome_df, how='left', on='ssid')

# add normalized confirm time
SCREEN_DATE = datetime.datetime.strptime('2020-03-01', '%Y-%m-%d')
analysis_data['confirm_date'] = pd.to_datetime(analysis_data['confirm_date'], format='%Y-%m-%d')
normalized_confrim_date_list = []
for idx, row in analysis_data.iterrows():
    confirm_date = row['confirm_date']
    normalized_confrim_date_list.append((confirm_date - SCREEN_DATE).days)
analysis_data['norm_confirm_date'] = normalized_confrim_date_list





CONT_COLS = ['age_at_confirm', 'norm_confirm_date']
CAT_COLS = ['sex', 'death', 'death_within_28d', 'death_within_60d',
            'is_icu', 'icu_28d', 'icu_60d', 'is_vent', 'vent_28d', 'vent_60d']
import _clust_charaterization
import _clust_charaterization_SDoH
report_df_1 = _clust_charaterization.three_clust_compare_sorted(data_df=analysis_data, lable_col='SDoH_clust',
                      all_cols = CONT_COLS + CAT_COLS,
                      continous_cols = CONT_COLS, 
                      categorical_cols = CAT_COLS)

report_df_2 = _clust_charaterization_SDoH.three_clust_SDoH(data_df=analysis_data, lable_col='SDoH_clust',
                      var_cols = SDoH_COLS_NEW
                      )

report_df = pd.concat([report_df_1, report_df_2])
report_df.to_csv(save_dir + '_SDoH_clust_charcteristics.csv', index=False)












