# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:01:21 2020


Extract BMI data: load last BMI data of each patient


@author: chs4001
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


MAIN_DIR = "W:\\WorkArea-chs4001"
SITES = [1, 3, 4, 5, 8]
SITES = [5, 8]




# load cohort info
cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', header = 0)
cohort_df = cohort_df[['ssid', 'CONFIRM_DATE']]
patient_list = {}
for p in cohort_df['ssid'].values.tolist():
    patient_list[p] = True


# load BMI data
#bmi_df = pd.DataFrame(columns=['ssid', 'wt', 'ht', 's_wt', 's_ht', 'bmi'])
#for site in SITES:
#    print('Site ', site, '...')
#    vital_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\vital.csv' % site, header = 0)
#    print(vital_df.shape)
#    vital_df = vital_df[['ssid', 'MEASURE_DATE', 'ht', 'wt', 'original_bmi']]
#    vital_df = vital_df[vital_df['ssid'].isin(patient_list)]
#    print(vital_df.shape)
#    
#    site_patients = list(set(vital_df['ssid'].values.tolist()))
#    
#    site_bmi_data = []
#    
#    for p in site_patients:
##        if p not in ['7d13e196-d125-4d56-bfda-fc8856352eb2',
##                     '4a8d3fb6-5665-49e8-9616-9d8e448a460b',
##                     '9398dca8-3f57-48e2-acec-aa08de1f1fd8'
##                     ]:
##            continue
#        
#        p_vatal_df = vital_df[vital_df['ssid'] == p]
#        
#        p_vatal_df['MEASURE_DATE'] = pd.to_datetime(p_vatal_df['MEASURE_DATE'], format='%d%b%Y')
#        
#        p_vatal_df = p_vatal_df.sort_values(by=['MEASURE_DATE'], ascending=False)
#        
#        # weight
#        p_wt_df = p_vatal_df['wt'].dropna().values.tolist()
#        if len(p_wt_df) == 0:
#            wt = np.nan
#        else:
#            wt = p_wt_df[0]
#         
#        # height
#        p_ht_df = p_vatal_df['ht'].dropna().values.tolist()
#        if len(p_ht_df) == 0:
#            ht = np.nan
#        else:
#            ht = p_ht_df[0]
#            
#        # height
#        p_bmi_df = p_vatal_df['original_bmi'].dropna().values.tolist()
#        if len(p_bmi_df) == 0:
#            bmi = np.nan
#        else:
#            bmi = p_bmi_df[0]
#            
#        
#        if site == 8:
#            s_wt = (wt / 16) * .453592
#        else:
#            s_wt = wt * .453592
#            
#        s_ht = ht * .0254
#        
#        s_bmi = s_wt / (s_ht * s_ht)
#        
#        if np.isnan(bmi) == True:
#            bmi = s_bmi
#            
#        
#        site_bmi_data.append([p, wt, ht, s_wt, s_ht, bmi])
#    
#    site_bmi_df = pd.DataFrame(site_bmi_data, columns=['ssid', 'wt', 'ht', 's_wt', 's_ht', 'bmi'])
#    
#    bmi_df = pd.concat([bmi_df, site_bmi_df])
#    
#    site_bmi_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\bmi_.csv' % site, index=False) 
#    
#    print('Finished...')
    
#bmi_df.to_csv(MAIN_DIR + '\\Data\\bmi__.csv', index=False)


bmi_df_site1 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site1\\bmi_.csv', header=0)
bmi_df_site3 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site3\\bmi_.csv', header=0)
bmi_df_site4 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site4\\bmi_.csv', header=0)
bmi_df_site5 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site5\\bmi_.csv', header=0)
bmi_df_site8 = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site8\\bmi_.csv', header=0)

bmi_df = pd.concat([bmi_df_site1, bmi_df_site3, bmi_df_site4, bmi_df_site5, bmi_df_site8])
bmi_df.to_csv(MAIN_DIR + '\\Data\\bmi__.csv', index=False)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        