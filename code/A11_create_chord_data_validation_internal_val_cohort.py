# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:47:56 2020

@author: chs4001
"""

import pandas as pd
import numpy as np




MAIN_DIR = "W:\\WorkArea-chs4001"
COHORT_FOLDER = '_validation_test_cohort'

DATA_FILE_NAME = "_demographics_deduplicated_labs_1006_window_-3_14_droped_full_missing.csv"




lable_annotation = {
                    1:'Subphenotype I',
                    2:'Subphenotype II',
                    3:'Subphenotype III',
                    4:'Subphenotype IV',
                    } 

SDoH_version = 'validation_k_nearest'



# load subphenotype labels
phenotype_df = pd.read_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_subphenotype_labels.csv', header=0)

# load lab, outcome, cormorbidity, SDoH data, demographics
data_df = pd.read_csv(MAIN_DIR + '\\Data\\' + DATA_FILE_NAME, header=0)
outcome_comorb_df = pd.read_csv(MAIN_DIR + '\\Data\\outcome_comorbidity_.csv', header=0).drop(columns=['sex', 'age_at_confirm'])
demographic_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics.csv', header=0)[['ssid', 'race', 'site']]
bmi_df = pd.read_csv(MAIN_DIR + '\\Data\\bmi__.csv', header=0)
SDoH_label = pd.read_csv(MAIN_DIR + '\\Results\\_SDoH_clust\\' + SDoH_version +'\\_SDoH_label.csv', header=0)

data = pd.merge(phenotype_df, data_df, how='left', on='ssid')
data = pd.merge(data, outcome_comorb_df, how='left', on='ssid')
data = pd.merge(data, demographic_df, how='left', on='ssid')
data = pd.merge(data, bmi_df, how='left', on='ssid')
data = pd.merge(data, SDoH_label, how='left', on='ssid')


# development lab data
dev_site_data = data_df[data_df['site'].isin([1, 3, 4, 5])]
dev_data = dev_site_data[dev_site_data['split_73'] == 'dev']


C = 4

# age
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    chord_data.append([lable_annotation[c], 'Age 18-40', len(temp_data[temp_data['age_at_confirm'] < 41])])
    chord_data.append([lable_annotation[c], 'Age 41-60', len(temp_data[(temp_data['age_at_confirm'] >= 41) & (temp_data['age_at_confirm'] < 61)])])
    chord_data.append([lable_annotation[c], 'Age 61-80', len(temp_data[(temp_data['age_at_confirm'] >= 61) & (temp_data['age_at_confirm'] < 81)])])
    chord_data.append([lable_annotation[c], 'Age>80', len(temp_data[temp_data['age_at_confirm'] >= 81])])
age_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
age_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_age.csv', index=False)

# age proportion
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    Nc = len(temp_data)
    chord_data.append([lable_annotation[c], 'Age 18-40', len(temp_data[temp_data['age_at_confirm'] < 41]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Age 41-60', len(temp_data[(temp_data['age_at_confirm'] >= 41) & (temp_data['age_at_confirm'] < 61)])  * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Age 61-80', len(temp_data[(temp_data['age_at_confirm'] >= 61) & (temp_data['age_at_confirm'] < 81)]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Age>80', len(temp_data[temp_data['age_at_confirm'] >= 81]) * 100 / Nc])
age_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
age_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_age_proportion.csv', index=False)



# gender
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    chord_data.append([lable_annotation[c], 'Female', len(temp_data[temp_data['sex'] == 1])])
    chord_data.append([lable_annotation[c], 'Male', len(temp_data[temp_data['sex'] == 0])])
sex_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
sex_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_sex.csv', index=False)

# gender proportion
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    Nc = len(temp_data)
    chord_data.append([lable_annotation[c], 'Female', len(temp_data[temp_data['sex'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Male', len(temp_data[temp_data['sex'] == 0]) * 100 / Nc])
sex_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
sex_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_sex_proportion.csv', index=False)


# race
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    chord_data.append([lable_annotation[c], 'White', len(temp_data[temp_data['race'] == '05'])])
    chord_data.append([lable_annotation[c], 'Black', len(temp_data[temp_data['race'] == '03'])])
    chord_data.append([lable_annotation[c], 'Asian', len(temp_data[temp_data['race'] == '02'])])
    chord_data.append([lable_annotation[c], 'Other/unknown', len(temp_data) - len(temp_data[temp_data['race'] == '05']) - len(temp_data[temp_data['race'] == '03']) - len(temp_data[temp_data['race'] == '02'])])
race_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
race_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_race.csv', index=False)

# race proportion
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    Nc = len(temp_data)
    chord_data.append([lable_annotation[c], 'White', len(temp_data[temp_data['race'] == '05']) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Black', len(temp_data[temp_data['race'] == '03']) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Asian', len(temp_data[temp_data['race'] == '02']) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Other/unknown', 
                       (len(temp_data) - len(temp_data[temp_data['race'] == '05']) - len(temp_data[temp_data['race'] == '03']) - len(temp_data[temp_data['race'] == '02']))  * 100 / Nc ])
race_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
race_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_race_proportion.csv', index=False)

# comorbidity exact count
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    chord_data.append([lable_annotation[c], 'HTN', len(temp_data[temp_data['dx_Hypertension'] == 1])])
    chord_data.append([lable_annotation[c], 'Diabetes', len(temp_data[temp_data['dx_Diabetes'] == 1])])
    chord_data.append([lable_annotation[c], 'CAD', len(temp_data[temp_data['dx_Coronary artery disease'] == 1])])
    chord_data.append([lable_annotation[c], 'HF', len(temp_data[temp_data['dx_Heart failure'] == 1])])
    chord_data.append([lable_annotation[c], 'COPD', len(temp_data[temp_data['dx_COPD'] == 1])])
    chord_data.append([lable_annotation[c], 'ATA', len(temp_data[temp_data['dx_Asthma'] == 1])])
    chord_data.append([lable_annotation[c], 'Cancer', len(temp_data[temp_data['dx_Cancer_AHRQ_CCS'] == 1])])
    chord_data.append([lable_annotation[c], 'HLD', len(temp_data[temp_data['dx_Hyperlipidemia '] == 1])])
    chord_data.append([lable_annotation[c], 'Obesity', len(temp_data[temp_data['bmi'] >= 30])])
comorb_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
comorb_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_comorbidity.csv', index=False)

# comorbidity proportion
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    Nc = len(temp_data)
    chord_data.append([lable_annotation[c], 'HTN', len(temp_data[temp_data['dx_Hypertension'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Diabetes', len(temp_data[temp_data['dx_Diabetes'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'CAD', len(temp_data[temp_data['dx_Coronary artery disease'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'HF', len(temp_data[temp_data['dx_Heart failure'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'COPD', len(temp_data[temp_data['dx_COPD'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'ATA', len(temp_data[temp_data['dx_Asthma'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Cancer', len(temp_data[temp_data['dx_Cancer_AHRQ_CCS'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'HLD', len(temp_data[temp_data['dx_Hyperlipidemia '] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'Obesity', len(temp_data[temp_data['bmi'] >= 30]) * 100 / Nc])
comorb_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
comorb_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_comorbidity_proportion.csv', index=False)
    
  



  
# lab
LAB_VARS = {
        'Inflammation': [
                'C-reactive_protein', 
                'ESR', 
                'interleukin-6',                 
                'procalcitonin', 
                'Neutrophils.band',
                'LDH',  
                'lymphocyte_count', 
                'neutrophil_count', 
                'white_blood_cell_count', 
                'albumin',
                'Ferritin',
                ],
                               
        'Hepatic': [
                'albumin',
                'Ferritin',
                 'alanine_aminotransferase', 
                 'aspartate_aminotransferase', 
                 'bilirubin',                                
                 ],
        
        'Cardiovascular': [
                'creatine_kinase',
                'venous_lactate', 
                'cardiac_troponin_I', 
                'cardiac_troponin_T',
                ],
                
         'Renal': [
                 'Bicarbonate', 
                 'BUN',
                 'creatinine',
                 'CHLORIDE',
                 'SODIUM', 
                 ],  
                 
         'Hematologic': [
                 'D-dimer',
                 'HEMOGLOBIN', 
                 'platelet_count',
                 'prothrombin_time',  # INR
                 'red_blood_cell_distribution_width', 
                 'GLUCOSE', 
                 ],
        }
REVERSE_DIRECTION_COLS = {
        'albumin',
        'Bicarbonate',
        'HEMOGLOBIN',
        'platelet_count',
        'Oxygen_saturation'
        }
chord_data = []   
for c in range(1, C+1):
    temp_data = data[data['label'] == c] 
    Nc = len(temp_data)
    for var_class in LAB_VARS:
        var_list = LAB_VARS[var_class]
        value = 0
        for var in var_list:
            if var not in REVERSE_DIRECTION_COLS:
                n = len(temp_data[temp_data[var] > np.nanmedian(data[var].values)])
            else:
                n = len(temp_data[temp_data[var] < np.nanmedian(data[var].values)])
            value += 100 * n / Nc
        
        chord_data.append([c, var_class, value])
lab_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value']) # cumulative proportion of each variable
lab_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_lab_proportion.csv', index=False)    
            

# lab group median comparison
chord_data = []   
for c in range(1, C+1):
    temp_data = data[data['label'] == c]  
    for var_class in LAB_VARS:
        var_list = LAB_VARS[var_class]
        value = 0
        for var in var_list:
            if var not in REVERSE_DIRECTION_COLS:
                if np.nanmedian(temp_data[var].values) > np.nanmedian(dev_data[var].values):
                    value += 1
            else:
                if np.nanmedian(temp_data[var].values) < np.nanmedian(dev_data[var].values):
                    value += 1
        chord_data.append([lable_annotation[c], var_class, value])
lab_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
lab_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_lab_group_level.csv', index=False)                     
                    





# SDoH proportion
chord_data = []
for c in range(1, C+1):
    temp_data = data[data['label'] == c]
    Nc = len(temp_data)
    chord_data.append([lable_annotation[c], 'SDoH Level 1', len(temp_data[temp_data['SDoH_clust'] == 1]) * 100 / Nc])
    chord_data.append([lable_annotation[c], 'SDoH Level 2', len(temp_data[(temp_data['SDoH_clust'] == 2)])  * 100 / Nc])
    chord_data.append([lable_annotation[c], 'SDoH Level 3', len(temp_data[(temp_data['SDoH_clust'] == 3)])  * 100 / Nc])
#    chord_data.append([lable_annotation[c], 'Unknown/missing',
#                     (len(temp_data) - len(temp_data[(temp_data['SDoH_clust'] == 1)]) - len(temp_data[(temp_data['SDoH_clust'] == 2)]) - len(temp_data[(temp_data['SDoH_clust'] == 3)]))  * 100 / Nc])
SDoH_chord_df = pd.DataFrame(chord_data, columns=['from', 'to', 'value'])
SDoH_chord_df.to_csv(MAIN_DIR + '\\Results\\' + COHORT_FOLDER + '\\_chord_phenotype_SDoH_proportion.csv', index=False)  


