# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:11:12 2020

@author: chs4001

Extract outcomes and comorbidities.

Outcomes:
    
    1. mortality
    
Comorbidities
    hypertension, 
    diabetes, 
    CAD, 
    heart failure, 
    COPD, 
    Asthma, 
    cacncer_CMS, 
    cancer_AQHR_CCS, 
    Hyperlipidemia

"""


import pandas as pd
import numpy as np
import json
from datetime import datetime



MAIN_DIR = "W:\\WorkArea-chs4001"



# load cohort info
cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', header = 0)
cohort_df = cohort_df[['ssid', 'CONFIRM_DATE', 'age_at_confirm', 'hispanic', 'race', 'sex',
                       'site', 'facilityid']]
cohort_df['CONFIRM_DATE'] = pd.to_datetime(cohort_df['CONFIRM_DATE'], format='%Y-%m-%d')


# ---------------------- Outcome mortality ----------------------- #
# load death info
death_1 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site1\\CSV\\death.csv', header = 0)
death_3 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site3\\CSV\\death.csv', header = 0)
death_4 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site4\\CSV\\death.csv', header = 0)
death_5 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site5\\CSV\\death.csv', header = 0)
death_8 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site8\\CSV\\death.csv', header = 0)
death_df = pd.concat([death_1, death_3, death_4, death_5, death_8])
death_df = death_df[['ssid', 'DEATH_DATE']]
death_df['DEATH_DATE'] = pd.to_datetime(death_df['DEATH_DATE'], format='%d%b%Y')

cohort_df = pd.merge(cohort_df, death_df, how='left', on='ssid')

# update death info
cohort_df['death_day_since_confirm'] = np.nan
cohort_df['death'] = 0
cohort_df['death_within_28d'] =  0
cohort_df['death_within_60d'] =  0
cohort_df['DEATH_DATE'] = cohort_df['DEATH_DATE'].fillna('None')
for idx, row in cohort_df.iterrows():
    c_date, d_date = row['CONFIRM_DATE'], row['DEATH_DATE']
    if d_date != 'None':
        days = (d_date - c_date).days
        cohort_df.loc[idx, 'death_day_since_confirm'] = days
        cohort_df.loc[idx, 'death'] = 1
        
        if days <= 28:
            cohort_df.loc[idx, 'death_within_28d'] = 1
            
        if days <= 60:
            cohort_df.loc[idx, 'death_within_60d'] = 1

#cohort_df.to_csv(MAIN_DIR + '\\Data\\outcome_comorbidity.csv', index=False)













# ------------------------- Comorbidities -------------------------- #
SITES = [1, 3, 4, 5, 8]
#SITES = [4]

patient_list = {}
#for p in cohort_df['ssid'].values.tolist():
#    patient_list[p] = True

for idx, row in cohort_df.iterrows():
    p, confirm_date = row['ssid'], row['CONFIRM_DATE']
    patient_list[p] = confirm_date

# load comorbidity codes
with open(MAIN_DIR + '\\comorbidity_codes.json') as json_f:
    com_codes = json.load(json_f)

all_codes = {}
for com in com_codes:
    for ctype in ['icd_9', 'icd_10']:
        for c in com_codes[com][ctype]:
            all_codes[c] = True
            
# initial filtering  
com_type_list = list(com_codes.keys())
all_com_df = pd.DataFrame(columns=['ssid', 'site'] + com_type_list)
for site in SITES:
    t1 = datetime.now()
    
    dx_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\diagnosis.csv' % site, header = 0,
                        dtype={'dx':str, 'dx_type': 'str'})

    # filter by patient
    dx_df = dx_df[dx_df['ssid'].isin(patient_list)]
    

    
    com_df = pd.DataFrame(columns=['ssid', 'site'] + com_type_list)
    dx_site_cohort = list(set(dx_df['ssid'].values.tolist()))
    j = 0
    for p in dx_site_cohort: #[:10]:
#        if p not in [
#                    'd0c7a80d-e279-4d07-a0d6-f2f2244ba024',
#                     '0d0fa4ee-3cc9-48ef-a5c5-3a10f40aaf0e',
#                     '7567fc5b-d516-484e-9601-bf5bf76f1bcd'
#                     ]:
#            continue
#        print(p, '-----------')
        p_dx_df = dx_df[dx_df['ssid'] == p]
        p_dx_df['ADMIT_DATE'] = pd.to_datetime(p_dx_df['ADMIT_DATE'], format='%d%b%Y')
        
        p_confirm_date = patient_list[p]
        p_dx_df = p_dx_df[p_dx_df['ADMIT_DATE'] <= p_confirm_date]
        p_dx_df.to_csv('test.csv', index=False)
        
        p_dx_data = [p, site] + [0] * len(com_type_list)
        p_dx_9_codes = p_dx_df[p_dx_df['dx_type'] == '9']['dx'].values.tolist()
        p_dx_10_codes = p_dx_df[p_dx_df['dx_type'] == '10']['dx'].values.tolist()
        for i in range(len(com_type_list)):
            ctype = com_type_list[i]
            ctype_icd9_list = com_codes[ctype]['icd_9']
            ctype_icd10_list = com_codes[ctype]['icd_10']
            
            for c in p_dx_9_codes:
                if c in ctype_icd9_list:
#                    print(ctype, c, '9')
                    p_dx_data[i + 2] = 1
                    break
            
            for c in p_dx_10_codes:
                if c in ctype_icd10_list:
#                    print(ctype, c, '10')
                    p_dx_data[i + 2] = 1
                    break
        
        com_df.loc[j] = p_dx_data
        j += 1
  
    
    all_com_df = pd.concat([all_com_df, com_df])
    
    com_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\com_dx_data.csv' % site, index=False)
    
    t2 = datetime.now()
    
    print('Finished site: %s with %d seconds' % (site, (t2-t1).seconds))
    
    
    
all_com_df.to_csv(MAIN_DIR + '\\Data\\comorbidity_.csv', index=False)

all_com_df = all_com_df.drop(columns=['site'])

cohort_df = pd.merge(cohort_df, all_com_df, how='left', on='ssid')

cohort_df.to_csv(MAIN_DIR + '\\Data\\outcome_comorbidity_.csv', index=False)            
            
            
            
        
        
        


















