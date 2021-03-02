# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:58:50 2020

@author: chs4001

Extract demographics information.

"""


import pandas as pd
import datetime
import numpy as np


MONTH_MAP = {
        'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06',
        'JUL':'07', 'AUG':'08', 'SEP':'09', 'OCT':'10', 'NOV':'11', 'DEC':'12'      
        }


MAIN_DIR = "W:\\WorkArea-chs4001"

SITES = [1, 3, 4, 5, 8]

#SITES = [1, 3]




combined_demographic_df = pd.DataFrame(columns=['ssid', 'CONFIRM_DATE', 'encounterid', 
                                                'BIRTH_DATE', 'sex', 'hispanic', 'race', 
                                                'site', 'age_at_confirm'])

for site in SITES:

    # load cohort
    cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\cohort\\Lab_DX_Site_%s.csv' % site)
    
    
    # load demographic data
    demographic_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\demographic.csv' % site,
                               header=0) 
    
    demographic_df = demographic_df[['ssid', 'BIRTH_DATE', 'sex', 'hispanic', 'race']]
        
    demographic_df = pd.merge(cohort_df, demographic_df, how='left', on='ssid')
    
    demographic_df['site'] = site
    
    demographic_df['age_at_confirm'] = np.nan
    
    for idx, row in demographic_df.iterrows():
        CONFIRM_DATE, BIRTH_DATE = row['CONFIRM_DATE'], row['BIRTH_DATE']
        BIRTH_DATE = '%s-%s-%s' % (BIRTH_DATE[5:], MONTH_MAP[BIRTH_DATE[2:5]], BIRTH_DATE[:2])
        
        CONFIRM_DATE_stamp = datetime.datetime.strptime(CONFIRM_DATE, '%Y-%m-%d')
        BIRTH_DATE_stamp = datetime.datetime.strptime(BIRTH_DATE, '%Y-%m-%d')
        
        age = int((CONFIRM_DATE_stamp - BIRTH_DATE_stamp).days) / 365
        
        demographic_df.loc[idx, 'age_at_confirm'] = age
    
    print(demographic_df.shape)
    
    combined_demographic_df = pd.concat([combined_demographic_df, demographic_df])


# add duplicate info
dedup_df = pd.read_csv(MAIN_DIR + '\\Data\\Dedup_SSID\\covid_ssid_dedup.csv', header=0)

combined_demographic_df = pd.merge(combined_demographic_df, dedup_df, how='left', on='ssid')

combined_demographic_df = combined_demographic_df.sort_values(by=['CONFIRM_DATE'])


combined_demographic_df.to_csv(MAIN_DIR + '\\Data\\_demographics.csv', index=False)   

# deduplicate
print(combined_demographic_df.shape)
combined_demographic_df = combined_demographic_df[(~combined_demographic_df.duplicated(subset=['unique_patid'], keep='first')) | 
                                                   (combined_demographic_df['unique_patid'].isnull())]
print(combined_demographic_df.shape)

combined_demographic_df.to_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', index=False)   


    


