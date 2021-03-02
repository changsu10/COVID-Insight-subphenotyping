# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:01:21 2020


Extract Survival data


@author: chs4001
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import datetime


MAIN_DIR = "W:\\WorkArea-chs4001"
SITES = [1, 3, 4, 5, 8]
#SITES = [1]




## Step 1
#
## load cohort info
#cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', header = 0)
#cohort_df = cohort_df[['ssid', 'CONFIRM_DATE']]
#patient_list = {}
#for idx, row in cohort_df.iterrows():
#    p, confirm_date = row['ssid'], row['CONFIRM_DATE']
#    patient_list[p] = confirm_date
#
#
#
#surv_date_df = pd.DataFrame()
#for site in SITES:
#    site_enc_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\encounter.csv' % site, header=0)    
#    
#    site_enc_df = site_enc_df[site_enc_df['ssid'].isin(patient_list)][['ssid', 'ADMIT_DATE', 'DISCHARGE_DATE']]
#    
#    site_enc_df['ADMIT_DATE'] = pd.to_datetime(site_enc_df['ADMIT_DATE'], format='%d%b%Y')
#    
#    site_enc_df['DISCHARGE_DATE'] = pd.to_datetime(site_enc_df['DISCHARGE_DATE'], format='%d%b%Y')
#        
#    # sort by discharge date
#    site_enc_df = site_enc_df.sort_values(by=['ssid', 'DISCHARGE_DATE'], ascending=True)
#            
#    # keep the last event
#    site_enc_df = site_enc_df.drop_duplicates(subset=['ssid'], keep='last')
#    
#    last_date_raw = []
#    lase_date = []
#    for idx, row in site_enc_df.iterrows():
#        p, disc_date = row['ssid'], row['DISCHARGE_DATE']
#        confirm_date =  datetime.datetime.strptime(patient_list[p], '%Y-%m-%d')
#        days = (disc_date - confirm_date).days
#        
#        last_date_raw.append(days)
#        if days >= 0:
#            lase_date.append(days)
#        else:
#            lase_date.append(np.nan)
#     
#    site_enc_df['days'] = lase_date
##    site_enc_df.to_csv('test.csv', index=False)
#    
#    surv_date_df = pd.concat([surv_date_df, site_enc_df])
#    
#    print('Site %s finished...' % site)
#    
#surv_date_df.to_csv(MAIN_DIR + '\\Data\\survive_raw.csv', index=False)






# step 2

surv_date_df = pd.read_csv(MAIN_DIR + '\\Data\\survive_raw.csv', header=0)  


# load outcome info
outcome_comorb_df = pd.read_csv(MAIN_DIR + '\\Data\\outcome_comorbidity_.csv', header=0)[['ssid', 'death_day_since_confirm', 'death']]

# load icu ventilation outcome
icu_outcome_df = pd.read_csv(MAIN_DIR + '\\Data\\icu_vent.csv', header=0)[['ssid', 'icu_start_days_since_confirm', 
                                                                            'vent_start_days_since_confirm', 'is_icu', 'is_vent']]

# merge
surv_date_df = pd.merge(outcome_comorb_df, surv_date_df, how='left', on='ssid')    
    
surv_date_df = pd.merge(icu_outcome_df, surv_date_df, how='left', on='ssid')      
    
# update data
surv_date_df['dtime'] = np.nan
surv_date_df['itime'] = np.nan
surv_date_df['vtime'] = np.nan

for idx, row in surv_date_df.iterrows():
    dtime, itime, vtime, days = row['death_day_since_confirm'], row['icu_start_days_since_confirm'], row['vent_start_days_since_confirm'], row['days']
    
    if dtime >= 0:
        surv_date_df.loc[idx, 'dtime'] = dtime
    else:
        surv_date_df.loc[idx, 'dtime'] = days
        
    if itime >= 0:
        surv_date_df.loc[idx, 'itime'] = itime
    else:
        surv_date_df.loc[idx, 'itime'] = days
        
    if vtime >= 0:
        surv_date_df.loc[idx, 'vtime'] = vtime
    else:
        surv_date_df.loc[idx, 'vtime'] = days


surv_date_df.to_csv(MAIN_DIR + '\\Data\\survive_.csv', index=False)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        