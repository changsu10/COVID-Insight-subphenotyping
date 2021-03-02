# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:46:05 2020


1. Train classifier over development cohort

2. Predict subphenotype over Site 8

@author: chs4001
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

from scaler import min_max_scaling, z_score_scaling, z_score_scaling_external
#from _clust_charaterization import two_clust_compare


import umap


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

import datetime

from joblib import dump, load

import pickle as pkl


TARGET_LABs = [
#        'age_at_confirm',
        
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

#OUTPUT_FOLDER = '_development'

#split_method = 'split'
split_method = 'split_73'



# load data
data = pd.read_csv(input_file, header=0)


# ------------ build cohort --------------- #
dev_site_data = data[data['site'].isin([1, 3, 4, 5])]

dev_data = dev_site_data[dev_site_data[split_method] == 'dev']
val_data = dev_site_data[dev_site_data[split_method] == 'val']

dev_study_data = dev_data[['ssid'] + CLUSTERING_COLS]
val_study_data = val_data[['ssid'] + CLUSTERING_COLS]

val_ind_data = data[data['site'].isin([8])][['ssid'] + CLUSTERING_COLS]



# ------------ site 8 new version ------ #
SCREEN_END_DATE = datetime.datetime.strptime('2020-06-12', '%Y-%m-%d')

val_ind_data_new_raw = pd.read_csv(MAIN_DIR + '\\Data_1123\\_demographics_deduplicated_labs_1211_window_-3_14_droped_full_missing.csv',
                               header=0)

val_ind_data_new_raw['confirm_date'] = pd.to_datetime(val_ind_data_new_raw['confirm_date'], format='%Y-%m-%d')

val_ind_data_new_raw = val_ind_data_new_raw[val_ind_data_new_raw['confirm_date'] <= SCREEN_END_DATE]


val_ind_data_new = val_ind_data_new_raw[['ssid'] + CLUSTERING_COLS]






for col in CLUSTERING_COLS:
    print(col, np.nanmin(data[col].values), np.percentile(data[col].dropna().values, 0.1), len(data[data[col] == 0]))

for col in CLUSTERING_COLS:
    print(col, np.nanmin(val_ind_data_new_raw[col].values), np.percentile(val_ind_data_new_raw[col].dropna().values, 0.1), len(val_ind_data_new_raw[val_ind_data_new_raw[col] == 0]))
 
    
# ----------- Log transform ----------------- #
for col in LOG_CLOS:
    dev_study_data[col] = dev_study_data[col].replace({0: np.nan})
    dev_study_data[col] = np.log(dev_study_data[col].values)
    
    val_ind_data[col] = val_ind_data[col].replace({0: np.nan})
    val_ind_data[col] = np.log(val_ind_data[col].values)
    
    val_ind_data_new[col] = val_ind_data_new[col].replace({0: np.nan})
    val_ind_data_new[col] = np.log(val_ind_data_new[col].values)
    

# ----------- scaling ------------- #
for col in CLUSTERING_COLS:    
    val_ind_data[col] = z_score_scaling_external(x=val_ind_data[col], source_x=dev_study_data[col])
    val_ind_data_new[col] = z_score_scaling_external(x=val_ind_data_new[col], source_x=dev_study_data[col])
    
#    val_ind_data[col] = z_score_scaling(val_ind_data[col].values)
#    dev_study_data[col] = min_max_scaling(dev_study_data[col].values)
    dev_study_data[col] = z_score_scaling(dev_study_data[col].values)


#
#for col in CLUSTERING_COLS:
#    value = val_ind_data[col]
#    print('------ %s -------' % col)
#    plt.hist(value, bins=100)
#    plt.show()
#    plt.close()
#    
#    value = val_ind_data_new[col]
#    print('------ %s -------' % col)
#    plt.hist(value, bins=100)
#    plt.show()
#    plt.close()
    
    


# ----------- KNN imputation -------------- #
imputer = KNNImputer(n_neighbors=10, weights='distance')
#imputer = load(MAIN_DIR + '\\Results\\dev_knnimputer.joblib')
dev_study_data[CLUSTERING_COLS] = imputer.fit_transform(dev_study_data[CLUSTERING_COLS])
val_ind_data[CLUSTERING_COLS] = imputer.transform(val_ind_data[CLUSTERING_COLS])
val_ind_data_new[CLUSTERING_COLS] = imputer.transform(val_ind_data_new[CLUSTERING_COLS])



#for col in CLUSTERING_COLS:
#    value = val_ind_data[col]
#    print('------ %s imputed -------' % col)
#    plt.hist(value, bins=100)
#    plt.show()
#    plt.close()
#    
#    value = val_ind_data_new[col]
#    print('------ %s imputed -------' % col)
#    plt.hist(value, bins=100)
#    plt.show()
#    plt.close()
    
    
    

# -------------- umap -------------------- #
umpa_reducer = umap.UMAP(n_neighbors=10, min_dist=0.35, random_state=42) # 0, 1
#X_umap = umpa_reducer.fit_transform(val_ind_data[CLUSTERING_COLS])
X_umap = umpa_reducer.fit_transform(val_ind_data_new[CLUSTERING_COLS])

plt.scatter(X_umap[:, 0], X_umap[:, 1], s=1, alpha=0.5)
plt.show()
plt.close()



 

# ------------- Incorporate other information ----------- #
#comorb_df = pd.read_csv(MAIN_DIR + '\\Data\\outcome_comorbidity_.csv', header=0).drop(columns=['site', 'facilityid', 'hispanic',
#                                                                                                       'DEATH_DATE', 'death_day_since_confirm',
#                                                                                                       'death',
#                                                                                                       'death_within_28d', 'death_within_60d',
#                                                                                                       'dx_Cancer_CMS', 
#                                                                                                       'CONFIRM_DATE'])
## load bmi
#bmi_df = pd.read_csv(MAIN_DIR + '\\Data\\bmi__.csv', header=0)
#comorb_df = pd.merge(comorb_df, bmi_df[['ssid', 'bmi']], how='left', on='ssid')    
#    
#comorb_df['Obesity'] = 0
#comorb_df['White race'] = 0
#comorb_df['Black race'] = 0
#comorb_df['Asian race'] = 0
#comorb_df['Multiple race'] = 0
#comorb_df['Other/unknown race'] = 0
#
#comorb_df['Sex male'] = 0
#comorb_df['Sex female'] = 0
#
#
#for idx, row in comorb_df.iterrows():
#    race, sex = row['race'], row['sex']    
#    if race == '05':
#        comorb_df.loc[idx, 'White race'] = 1
#    elif race == '03':
#        comorb_df.loc[idx, 'Black race'] = 1
#    elif race == '02':
#        comorb_df.loc[idx, 'Asian race'] = 1
#    elif race == '06':
#        comorb_df.loc[idx, 'Multiple race'] = 1
#    else:
#        comorb_df.loc[idx, 'Other/unknown race'] = 1
#        
#    if sex == 'F':
#        comorb_df.loc[idx, 'Sex female'] = 1
#    if sex == 'M':
#        comorb_df.loc[idx, 'Sex male'] = 1
#        
#    bmi = row['bmi']
#    if bmi >= 30:
#        comorb_df.loc[idx, 'Obesity'] = 1
#   
#comorb_df.to_csv(MAIN_DIR + '\\Data\\_classification_other_info.csv', index=False)

# load exisiting comorbidity data
comorb_df = pd.read_csv(MAIN_DIR + '\\Data\\_classification_other_info.csv', header=0).drop(columns=['race', 'sex', 'Sex male'])
comorb_df.isnull().sum()
comorb_df = comorb_df.fillna(0)  # some patients have no comorbidity data, imputed by 0

# combine
dev_study_data_combined = pd.merge(dev_study_data, comorb_df, how='left', on='ssid')
val_ind_data_combined = pd.merge(val_ind_data, comorb_df, how='left', on='ssid')


# load label
dev_label_df = pd.read_csv(MAIN_DIR + '\\Results\\_development\\_subphenotype_labels.csv')[['ssid', 'label']]
dev_study_data_combined = pd.merge(dev_study_data_combined, dev_label_df, how='left', on='ssid')
#dev_study_data_combined.to_csv('test.csv', index=False)
dev_study_data_combined.to_csv(MAIN_DIR + '\\Data\\_classification_other_info_.csv', index=False)




# ------------- Incorporate other information for new site 8 ----------- #
#comorb_df_s8 = pd.read_csv(MAIN_DIR + '\\Data_1123\\outcome_comorbidity_.csv', header=0).drop(columns=['site', 'hispanic',
#                                                                                                       'DEATH_DATE', 'death_day_since_confirm',
#                                                                                                       'death',
#                                                                                                       'death_within_28d', 'death_within_60d',
#                                                                                                       'dx_Cancer_CMS', 
#                                                                                                       'CONFIRM_DATE'])
#
## load bmi
#bmi_df_s8 = pd.read_csv(MAIN_DIR + '\\Data_1123\\bmi__.csv', header=0)
#comorb_df_s8 = pd.merge(comorb_df_s8, bmi_df_s8[['ssid', 'bmi']], how='left', on='ssid')  
#    
#comorb_df_s8['Obesity'] = 0
#comorb_df_s8['White race'] = 0
#comorb_df_s8['Black race'] = 0
#comorb_df_s8['Asian race'] = 0
#comorb_df_s8['Multiple race'] = 0
#comorb_df_s8['Other/unknown race'] = 0
#
#comorb_df_s8['Sex male'] = 0
#comorb_df_s8['Sex female'] = 0
#
#for idx, row in comorb_df_s8.iterrows():
#    race, sex = row['race'], row['sex']    
#    if race == '05':
#        comorb_df_s8.loc[idx, 'White race'] = 1
#    elif race == '03':
#        comorb_df_s8.loc[idx, 'Black race'] = 1
#    elif race == '02':
#        comorb_df_s8.loc[idx, 'Asian race'] = 1
#    elif race == '06':
#        comorb_df_s8.loc[idx, 'Multiple race'] = 1
#    else:
#        comorb_df_s8.loc[idx, 'Other/unknown race'] = 1
#        
#    if sex == 'F':
#        comorb_df_s8.loc[idx, 'Sex female'] = 1
#    if sex == 'M':
#        comorb_df_s8.loc[idx, 'Sex male'] = 1
#        
#    bmi = row['bmi']
#    if bmi >= 30:
#        comorb_df_s8.loc[idx, 'Obesity'] = 1
#   
#comorb_df_s8.to_csv(MAIN_DIR + '\\Data_1123\\_classification_other_info_site_8.csv', index=False)

# load exisiting comorbidity data
comorb_df_s8 = pd.read_csv(MAIN_DIR + '\\Data_1123\\_classification_other_info_site_8.csv', header=0).drop(columns=['race', 'sex', 'Sex male'])
comorb_df_s8.isnull().sum()
comorb_df_s8 = comorb_df_s8.fillna(0)  # some patients have no comorbidity data, imputed by 0

# combine
val_ind_data_new_combined = pd.merge(val_ind_data_new, comorb_df_s8, how='left', on='ssid')
val_ind_data_new_combined.to_csv(MAIN_DIR + '\\Data_1123\\_classification_other_info_site_8_.csv', index=False)







# ------------- Model training over development cohort ----------- #
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp

import matplotlib.pyplot as plt

C =  4

data_cols = dev_study_data_combined.drop(columns=['ssid', 'label', 'bmi']).columns.tolist()
data_cols = TARGET_LABs

#X = dev_study_data_combined.drop(columns=['ssid', 'label']).values
X = dev_study_data_combined[data_cols].values
y = dev_study_data_combined['label'].values
y_bi = label_binarize(y, classes=[1, 2, 3, 4])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_tprs = dict()
cv_aucs = dict()
cv_mean_fpr = np.linspace(0, 1, 100)
j = 1
for train, test in cv.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    y_train_bi, y_test_bi = y_bi[train], y_bi[test]

    single_classifier = XGBClassifier(booster='gbtree', objective='binary:logistic', random_state=1)
    classifier = OneVsRestClassifier(single_classifier)
    
    classifier.fit(X_train, y_train_bi)
    
    y_score = classifier.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(C):
        fpr[i], tpr[i], _ = roc_curve(y_test_bi[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        cv_interp_tpr = interp(cv_mean_fpr, fpr[i], tpr[i])
        cv_interp_tpr[0] = 0.0
        if j == 1:
            cv_tprs[i] = [cv_interp_tpr]
            cv_aucs[i] = [roc_auc[i]]
        else:
            cv_tprs[i].append(cv_interp_tpr)
            cv_aucs[i].append(roc_auc[i])
        
        
    
    # micro average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bi.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    cv_interp_tpr = interp(cv_mean_fpr, fpr['micro'], tpr['micro'])
    cv_interp_tpr[0] = 0.0
    if j == 1:
        cv_tprs['micro'] = [cv_interp_tpr]
        cv_aucs['micro'] = [roc_auc['micro']]
    else:
        cv_tprs['micro'].append(cv_interp_tpr)
        cv_aucs['micro'].append(roc_auc['micro'])
    
    
    # macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(C)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(C):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= C
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    
    cv_interp_tpr = interp(cv_mean_fpr, fpr['macro'], tpr['macro'])
    cv_interp_tpr[0] = 0.0
    if j == 1:
        cv_tprs['macro'] = [cv_interp_tpr]
        cv_aucs['macro'] = [roc_auc['macro']]
    else:
        cv_tprs['macro'].append(cv_interp_tpr)
        cv_aucs['macro'].append(roc_auc['macro'])
    
        
#    plt.plot(fpr[0], tpr[0], label=1)
#    plt.plot(fpr[1], tpr[1], label=2)
#    plt.plot(fpr[2], tpr[2], label=3)
#    plt.plot(fpr[3], tpr[3], label=4)
#    plt.plot(fpr['micro'], tpr['micro'], label='micro')
#    plt.plot(fpr['macro'], tpr['macro'], label='macro')
#    plt.legend()
#    plt.show()
    
    print('Iteraction %d finished...' % j)
    
#    if j > 2:        
#        break
    
    j += 1
    
    

lable_color = {0:'#79A3D9', 1:'#7B967A', 2:'#F9C77E', 3:'#CE4257', 'micro':'#6C3075', 'macro':'gray'}              
lable_annotation = {0:'Subphenotype I vs rest',
                    1:'Subphenotype II vs rest',
                    2:'Subphenotype III vs rest',
                    3:'Subphenotype IV vs rest',
                    'micro': 'Micro-averaged ROC',
                    'macro': 'Macro-averaged ROC',
                    }     
fig = plt.figure(figsize=(6, 6))
for i in list(range(C)) + ['micro', 'macro']:
    mean_tpr = np.mean(cv_tprs[i], axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(cv_tprs[i], axis=0)
    mean_auc = np.mean(cv_aucs[i])
    std_auc = np.std(cv_aucs[i])
    tprs_upper = np.maximum(mean_tpr + std_tpr, 0)
    tprs_lower = np.minimum(mean_tpr - std_tpr, 1)
    plt.plot(cv_mean_fpr, mean_tpr, 
            label='%s (AUC = %0.2f $\pm$ %0.2f)' % (lable_annotation[i], mean_auc, std_auc),
            c=lable_color[i], lw=2)
    plt.fill_between(cv_mean_fpr, tprs_lower, tprs_upper, alpha=.2)
plt.plot([0, 1], [0, 1], linestyle=':', c='grey', lw=1)
#plt.legend(bbox_to_anchor = (2, 0), loc='lower right', fontsize=10)
plt.legend(fontsize=10)
plt.grid(c='#DCDCDC')
plt.xlabel('1 - Specificity', fontsize=12)
plt.ylabel('Sensitivity', fontsize=12)
plt.title('Recevier Operating Characteristic (ROC) curve', fontsize=13, y=1.01)
plt.savefig(MAIN_DIR + '\\Results\\_validation_ind_site_new_8\\ROC_train.pdf')
plt.show()
plt.close()

  







      
# train final classifier
single_classifier = XGBClassifier(booster='gbtree', objective='binary:logistic', random_state=1)
final_classifier = OneVsRestClassifier(single_classifier)
final_classifier.fit(X, y_bi)



# feature importance
import shap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
FEATURE_MAP = {
 'albumin': 'Albumin',
 'alanine_aminotransferase': 'ALT',
 'aspartate_aminotransferase': 'AST',
 'Bicarbonate': 'Bicarbonate',
 'bilirubin': 'Bilirubin',
 'BUN': 'BUN',
 'creatine_kinase': 'CK',
 'creatinine': 'Creatinine',
 'CHLORIDE': 'Chloride',
 'C-reactive_protein': 'CRP',
 'Ferritin': 'Ferritin',
 'GLUCOSE': 'Glucose',
 'HEMOGLOBIN': 'Hemoglobin',
 'venous_lactate': 'Lactate',
 'LDH': 'LDH',
 'lymphocyte_count': 'Lymphocyte',
 'neutrophil_count': 'Neutrophil',
 'prothrombin_time': 'PT',
 'platelet_count': 'Platelet',
 'red_blood_cell_distribution_width': 'RDW',
 'SODIUM': 'Sodium',
 'cardiac_troponin_T': 'Troponin T',
 'white_blood_cell_count': 'WBC',
# 'age_at_confirm': 'Age',
# 'dx_Hypertension': 'HTN',
# 'dx_Diabetes': 'Diabetes',
# 'dx_Coronary artery disease': 'CAD',
# 'dx_Heart failure': 'HF',
# 'dx_COPD': 'COPD',
# 'dx_Asthma': 'Asthma',
# 'dx_Cancer_AHRQ_CCS': 'Cancer',
# 'dx_Hyperlipidemia ': 'Hyperlipidemia',
# 'Obesity': 'Obesity',
# 'White race': 'White race',
# 'Black race': 'Black race',
# 'Asian race': 'Asian race',
# 'Multiple race': 'Multiple race',
# 'Other/unknown race': 'Other/unknown race',
# 'Sex female': 'Sex female',
        }

VAR_ORDER = [
                'C-reactive_protein', 
#                'ESR', 
#                'interleukin-6',                 
#                'procalcitonin', 
#                'Neutrophils.band',
                'LDH',  
                'lymphocyte_count', 
                'neutrophil_count', 
                'white_blood_cell_count',
                

                'albumin',
                'Ferritin',  
               
                
                 'alanine_aminotransferase', 
                 'aspartate_aminotransferase', 
                 'bilirubin',                                
                
                
                'creatine_kinase',
                'venous_lactate', 
#                'cardiac_troponin_I', 
                'cardiac_troponin_T',
                
                
                
                 'Bicarbonate', 
                 'BUN',
                 'creatinine',
                 'CHLORIDE',
                 'SODIUM', 
                
                
#                 'D-dimer',
                 'HEMOGLOBIN', 
                 'platelet_count',
                 'prothrombin_time',  # INR
                 'red_blood_cell_distribution_width', 
                 'GLUCOSE', 
                ]


features = [FEATURE_MAP[ft] for ft in data_cols]
for i in range(C): #[3]:
    print('model: ', i, '...')
    explainer = shap.TreeExplainer(final_classifier.estimators_[i])
    shap_values = explainer.shap_values(dev_study_data_combined[data_cols].rename(columns=FEATURE_MAP))
    
#    shap_values_df = pd.DataFrame(shap_values, columns=VAR_ORDER)
    
    order_index = []
    for var in VAR_ORDER:
        order_index.append(data_cols.index(var))
    
#    plt.figure(figsize=(4, 8))
    shap.summary_plot(shap_values[:, order_index], dev_study_data_combined[VAR_ORDER].rename(columns=FEATURE_MAP), 
#                      feature_names=features,
                      max_display=40, show=False,
                      plot_size=(6, 8),
                      sort=False,
                      )  # plot_type='violin'
    
    cmap = plt.get_cmap('RdYlBu_r')  # hot, RdBu_r, RdYlBu_r, copper, plasma
#    newcmap = ListedColormap, LinearSegmentedColormap
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, 'set_cmap'):
                fcc.set_cmap(cmap)
    
    plt.tight_layout()
    plt.xlim(-8, 8)
    plt.xlabel('SHAP value')
    plt.savefig(MAIN_DIR + '\\Results\\_validation_ind_site_new_8\\Shap_model_%s.pdf' % i)
    plt.show()
    plt.close()
    
#    fig = plt.figure()
#    fig.subplots_adjust(left=0.2)
#    shap.summary_plot(shap_values, dev_study_data_combined[data_cols], 
#                      feature_names=features, max_display=40, plot_type='bar',
#                      show=False,
#                      plot_size=(5, 12))   
##    plt.xlim(-6, 6)
#    plt.xlabel('Mean(|SHAP value|)')
#    plt.savefig(MAIN_DIR + '\\Results\\_validation_ind_site_new_8\\Shap_model_%s_bar.pdf' % i)
#    plt.close()
    








# predict label over validation site
y_val_pred = final_classifier.predict_proba(val_ind_data_new_combined[data_cols])
val_ind_subphenotype_label = np.argmax(y_val_pred, axis=1) + 1

val_ind_label_df = pd.DataFrame()
val_ind_label_df['ssid'] = val_ind_data_new_combined['ssid']
val_ind_label_df['label'] = val_ind_subphenotype_label
val_ind_label_df['umap_0'] = X_umap[:, 0]
val_ind_label_df['umap_1'] = X_umap[:, 1]
val_ind_label_df.to_csv(MAIN_DIR + '\\Results\\_validation_ind_site_new_8\\_subphenotype_labels.csv', index=False)




# plot label over umap
lable_color = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}              
lable_annotation = {1:'Subphenotype I',
                    2:'Subphenotype II',
                    3:'Subphenotype III',
                    4:'Subphenotype IV',
                    }             

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.08, right=0.7, top=0.98, bottom=0.05)
for c in range(1, C+1):
    plt.scatter(X_umap[val_ind_subphenotype_label==c, 0], X_umap[val_ind_subphenotype_label==c, 1], 
                s=1, alpha=.85, color=lable_color[c], label = lable_annotation[c])
    
plt.axis('off')
plt.legend(bbox_to_anchor = (1.4, 0.25), loc='upper right', fontsize=10, markerscale=4)
plt.savefig(MAIN_DIR + '\\Results\\_validation_ind_site_new_8\\umap_with_label.pdf', dpi=300)
plt.show()
plt.close()











# progression over time

data_time = []
max_time = 7 * 15
n_gap = int(max_time / 7)
bins = [i * 7 for i in range(0, n_gap+1)]

for c in range(1, C+1): # grouped bar plot version
    target_df = ind_site_data[ind_site_data['label'] == c]
    n_c = len(target_df)
  
    distribution = plt.hist(target_df['norm_confirm_date'].values, bins=bins)
    
    data_time.append(distribution[0])   
    
data_time = np.array(data_time)

data_time = data_time / data_time.sum(axis=0)

data_time = data_time[:, :-1]

total_dist = plt.hist(ind_site_data['norm_confirm_date'].values, bins=bins)[0]
total_dist = total_dist[:-1]
total_dist = total_dist / len(ind_site_data)

plt.close()      


fig = plt.figure(figsize=[10, 6])
fig.subplots_adjust(left=0.08, right=0.8, top=0.95, bottom=0.25, hspace=0.05)
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 0.5])
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])
ax1.bar(range(1, n_gap), data_time[0], color=lable_color[1], label=lable_annotation[1], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[1], bottom=data_time[0], color=lable_color[2], label=lable_annotation[2], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[2], bottom=data_time[0]+data_time[1], color=lable_color[3], label=lable_annotation[3], width=1, edgecolor='white', linewidth=1)
ax1.bar(range(1, n_gap), data_time[3], bottom=data_time[0]+data_time[1]+data_time[2], color=lable_color[4], label=lable_annotation[4], width=1, edgecolor='white', linewidth=1)
ax1.set_xlim(0, n_gap)
ax1.set_xticks([])
ax1.set_yticks([0, .25, .5, .75, 1])
ax1.set_yticklabels([0, 25, 50, 75, 100])
ax1.legend(bbox_to_anchor = (1.275, 0.36), loc='upper right', fontsize=11)
ax1.set_ylabel('Proportion within week (%)', fontsize=12)

ax2.bar(range(1, n_gap), total_dist, color='gray', width=1, edgecolor='white', linewidth=1)
ax2.set_ylim(0, .3)
ax2.set_xlim(0, n_gap)
ax2.set_xticks(range(1, n_gap))
ax2.set_xticklabels(['Week %d' % i for i in range(1, n_gap)], rotation=45)
ax2.set_yticks([0, .1, .2, .3])
ax2.set_yticklabels([0, 10, 20, 30])
ax2.set_ylabel('Proportion (%)', fontsize=12)
ax2.set_xlabel('Time since Mar 1, 2020', fontsize=12)


plt.savefig(MAIN_DIR + '\\Results\\_validation_ind_site_new_8\\confirm_time_bar_by_week.pdf')


    

   
   



