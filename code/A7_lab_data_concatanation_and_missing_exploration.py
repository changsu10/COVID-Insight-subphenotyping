# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 22:25:55 2020

@author: chs4001


1. Concatanate lab data across 5 sites, indexing by deduplicated demographics data

2. Drop patients missing all lab data

3. Explore missing rate within each site

4. Explore missing rate over whole data set

"""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



MAIN_DIR = "W:\\WorkArea-chs4001"

#LAB_DATA_VERSION = 'labs_1002_window_-3_14'
LAB_DATA_VERSION = 'labs_1006_window_-3_14'


SITES = [1, 3, 4, 5, 8]

#SITES = [1]


TARGET_LABs = [
        'venous_lactate', 'creatinine', 'white_blood_cell_count', 'lymphocyte_count', 
        'platelet_count', 'bilirubin', 'aspartate_aminotransferase', 'alanine_aminotransferase', 
        'creatine_kinase', 'prothrombin_time', 'interleukin-6', 'Ferritin', 'D-dimer', 'procalcitonin', 
        'albumin', 'red_blood_cell_distribution_width', 'neutrophil_count', 'C-reactive_protein', 
        'cardiac_troponin_I', 'cardiac_troponin_T',
        
        # added Oct 6
        'LDH', 'Bicarbonate', 'BUN','CHLORIDE', 'ESR', 'GLUCOSE', 
        'HEMOGLOBIN', 'SODIUM', 'Neutrophils.band',
        
        'Oxygen_saturation'
        
        ]




# load deduplicated demographics
demographic_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', header = 0)
print('Total (after deduplication): %s' % len(demographic_df))

demographic_df = demographic_df[demographic_df['sex'].isin(['F', 'M'])]
demographic_df = demographic_df[demographic_df['age_at_confirm'] >= 18]
print('After age/sex screening, remaining: %s' % len(demographic_df))

demographic_df['sex_raw'] = demographic_df['sex'].values
demographic_df['sex'] = demographic_df['sex'].replace({'F': 1, 'M': 0})

demographic_df['race'] = demographic_df['race'].replace({'02': 'Asian', 
                                                         '03': 'Black',
                                                         '05': 'White',
                                                         '06': 'Mutiple_race',
                                                         
                                                         '01': 'Other/unknown_race',
                                                         '04': 'Other/unknown_race',
                                                         'OT': 'Other/unknown_race',
                                                         
                                                         '07': 'Other/unknown_race',
                                                         'NI': 'Other/unknown_race',
                                                         'UN': 'Other/unknown_race',
                                                         })

demographic_df = demographic_df[['ssid', 'age_at_confirm', 'sex', 'sex_raw', 'race']]




# load lab data
lad_df_1 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\%s.csv' % (1, LAB_DATA_VERSION))
lad_df_3 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\%s.csv' % (3, LAB_DATA_VERSION))
lad_df_4 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\%s.csv' % (4, LAB_DATA_VERSION))
lad_df_5 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\%s.csv' % (5, LAB_DATA_VERSION))
lad_df_8 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\%s.csv' % (8, LAB_DATA_VERSION))

lad_df_1['site'] = 1
lad_df_3['site'] = 3
lad_df_4['site'] = 4
lad_df_5['site'] = 5
lad_df_8['site'] = 8

# lab data concatanation
lab_df = pd.concat([lad_df_1, lad_df_3, lad_df_4, lad_df_5, lad_df_8])
print('Lab data, originally patient number: %s' % len(lab_df))
lab_df = lab_df.drop(columns=['index', 'day_CIP', 'date_CIP', 'enc_id_CIP', 'day_CED', 'date_CED', 'enc_id_CED'])



# merge demographic data and lab data
data = pd.merge(demographic_df, lab_df, how='inner', on='ssid')

print('Merged, patients left: %s' % len(data))

data.to_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated_%s.csv' % LAB_DATA_VERSION, index=False)

data = data.dropna(subset=TARGET_LABs, how='all')

print('Droped patients missing all lab, left: %s' % len(data))



# --- assess lab data quality  --- #
missing_report_df = pd.DataFrame(columns=['Variable'])
cols = ['patient number', 'age_at_confirm'] + TARGET_LABs
missing_report_df['Variable'] = cols
# each site
for site in SITES:
    site_data = data[data['site'] == site] 
    n = len(site_data)
    
    site_data = site_data[['age_at_confirm'] + TARGET_LABs]
    
    missing_nums = site_data.isnull().sum().tolist()
    
    missing_report = []
    missing_report.append(n)
    for x in missing_nums:
        missing_report.append('%s (%.1f%%)' % (x, 100*x/n))
    
    missing_report_df['Site_%s' % site] = missing_report
    
# entire cohort
N = len(data)
missing_nums = data[['age_at_confirm'] + TARGET_LABs].isnull().sum().tolist()
missing_report = []
missing_report.append(N)
for x in missing_nums:
    missing_report.append('%s (%.1f%%)' % (x, 100*x/N))
missing_report_df['Total'] = missing_report
missing_report_df.to_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated_%s_missing_report.csv' % LAB_DATA_VERSION, index=False)



# -------- data distribution ---------- #
dist_report_df = pd.DataFrame(columns=['Variable'])
sex_cols = list(set(data['sex_raw'].values))
race_cols = list(set(data['race'].values))
cols = ['patient number', 'age_at_confirm'] + TARGET_LABs + sex_cols + race_cols
dist_report_df['Variable'] = cols
# each site
for site in SITES:
    site_data = data[data['site'] == site] 
    n = len(site_data)
    
    distr_data = []
    distr_data.append(n)
    for col in ['age_at_confirm'] + TARGET_LABs:
#        print(col)
        if len(site_data[col].dropna().values.tolist()) == 0:
            distr_data.append('-')
            continue
        col_median = np.nanmedian(site_data[col].values)
        col_IRQ_25 = np.percentile(site_data[col].dropna(), 25)
        col_IRQ_75 = np.percentile(site_data[col].dropna(), 75)
        distr_data.append('%.2f [%.2f - %.2f]' % (col_median, col_IRQ_25, col_IRQ_75))
    
    # sex
    for col in sex_cols:
        n_col = len(site_data[site_data['sex_raw'] == col])
        distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/n))
    
    # race
    for col in race_cols:
        n_col = len(site_data[site_data['race'] == col])
        distr_data.append('%d (%.1f%%)' % (n_col, 100*n_col/n))
    
    dist_report_df['Site_%s' % site] = distr_data

dist_report_df.to_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated_%s_droped_full_missing_distribution.csv' % LAB_DATA_VERSION, index=False)



# --- development - validation splitting --- #
# 6:4 split
X, y = data['ssid'].values, data['site'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
development_cohort = X_train.tolist()
validation_cohort = X_test.tolist()
dev_val_flags = []
for ssid in data['ssid'].values.tolist():
    if ssid in development_cohort:
        dev_val_flags.append('dev')
    else:
        dev_val_flags.append('val')
data['split'] = dev_val_flags

# 7:3 split
X, y = data['ssid'].values, data['site'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3) # 42 3
development_cohort = X_train.tolist()
validation_cohort = X_test.tolist()
dev_val_flags = []
for ssid in data['ssid'].values.tolist():
    if ssid in development_cohort:
        dev_val_flags.append('dev')
    else:
        dev_val_flags.append('val')
data['split_73'] = dev_val_flags

data.to_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated_%s_droped_full_missing.csv' % LAB_DATA_VERSION, index=False)



















    
    
    
    








