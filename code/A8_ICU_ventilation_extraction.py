# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:31:37 2020


Extract ICU and Ventilation event

@author: chs4001
"""

import pandas as pd
import numpy as np
import datetime


MAIN_DIR = "W:\\WorkArea-chs4001"

SCREEN_DATE = datetime.datetime.strptime('2020-03-01', '%Y-%m-%d')

ICU_CH_CODES = ['99291', '99292']
VENTILATION_CH_CODES = {
        '5A1935Z': 'Ventilation_<_24_Consecutive_Hours',
        '5A1945Z': 'Ventilation_24_96_Consecutive_Hours',
        '5A1955Z': 'Ventilation_>_96_Consecutive_Hours'
        }





SITES = [1, 3, 4, 5, 8]
#SITES = [1]



# load cohort info
cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', header = 0)
cohort_df = cohort_df[['ssid', 'CONFIRM_DATE', 'site']]
cohort_df['CONFIRM_DATE'] = pd.to_datetime(cohort_df['CONFIRM_DATE'], format='%Y-%m-%d')
patient_list = {}
for p in cohort_df['ssid'].values.tolist():
    patient_list[p] = True
  

## ------ initial raw data filtering ---------- #
#for site in SITES:
#    print('Site ', site, '...')
#    #  ----- filter procedure data
#    print('--- filter procedures...')
#    df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\procedures.csv' % site, header = 0,
#                     dtype = {'px': str, 'px_type': str})
#    print(df.shape)
#    
#    df = df[df['ssid'].isin(patient_list)]
#    print(df.shape)
#    
#    df['PX_DATE'] = pd.to_datetime(df['PX_DATE'], format='%d%b%Y')
#    df = df[df['PX_DATE'] >= SCREEN_DATE]
#    print(df.shape)
#    
#    # filtering by code
#    target_codes = ICU_CH_CODES + list(VENTILATION_CH_CODES.keys())
#    print(target_codes)
#    df = df[df['px'].isin(target_codes)]
#    print(df.shape)
#    
#    df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_procedures.csv' % site, index=False)
#    
#    
#    #  ----- filter obs_gen data
#    print('-- filter obs_gen...')
#    obs_gen_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\obs_gen.csv' % site, header = 0)
#    print(obs_gen_df.shape)
#    
#    obs_gen_df = obs_gen_df[obs_gen_df['ssid'].isin(patient_list)]
#    print(obs_gen_df.shape)
#    
#    obs_gen_df = obs_gen_df[obs_gen_df['obsgen_type'] == 'PC_COVID']
#    obs_gen_df = obs_gen_df[obs_gen_df['obsgen_code'].isin([2000, 3000])]
#    obs_gen_df = obs_gen_df[obs_gen_df['obsgen_result_text'] == 'Y']
#    obs_gen_df = obs_gen_df[obs_gen_df['obsgen_source'] == 'DR']
#    print(obs_gen_df.shape)
#    
#    obs_gen_df['OBSGEN_DATE'] = pd.to_datetime(obs_gen_df['OBSGEN_DATE'], format='%d%b%Y')
#    obs_gen_df = obs_gen_df[obs_gen_df['OBSGEN_DATE'] >= SCREEN_DATE]
#    print(obs_gen_df.shape)
#    
#    obs_gen_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_obs_gen.csv' % site, index=False)
    
  
    
    
    
    
## ------ generate data ---------- #
#for site in SITES:
#    print('Site ', site, '...')
#    proc_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_procedures.csv' % site, header=0)
#    obs_gen_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_obs_gen.csv' % site, header=0)
#    proc_df['PX_DATE'] = pd.to_datetime(proc_df['PX_DATE'], format='%Y-%m-%d')
#    obs_gen_df['OBSGEN_DATE'] = pd.to_datetime(obs_gen_df['OBSGEN_DATE'], format='%Y-%m-%d')
#    
#    site_cohort = cohort_df[cohort_df['site'] == site]
#    cohort_patient_list = {}
#    for idx, row in site_cohort.iterrows():
#        p, confirm_date = row['ssid'], row['CONFIRM_DATE']
#        cohort_patient_list[p] = confirm_date
#        
#        
#    icu_vent_data = []
#    
#    for p in cohort_patient_list:
##        if p not in ['5c8d80b5-6c8b-4f29-8e69-93639b7b6cbe',
##                     'd0fc34a5-dc70-47fb-b0bf-cb7127b9c84d',
##                     '9401de6a-7bee-476c-b394-10b7aa03e84c']:
##            continue
#        
#        confirm_date = cohort_patient_list[p]
#        
#        p_proc_df = proc_df[proc_df['ssid'] == p]
#        p_obs_gen_df = obs_gen_df[obs_gen_df['ssid'] == p]
#        
#        p_proc_df = p_proc_df[p_proc_df['PX_DATE'] >= confirm_date]
#        p_obs_gen_df = p_obs_gen_df[p_obs_gen_df['OBSGEN_DATE'] >= confirm_date]
#        
##        print(p, confirm_date, '----')
##        print(p_proc_df[['PX_DATE', 'px']])
##        print(p_obs_gen_df[['OBSGEN_DATE', 'obsgen_code']])
#        
#        # icu info 
#        p_proc_df_icu = p_proc_df[p_proc_df['px'].isin(ICU_CH_CODES)].sort_values(by=['PX_DATE'], ascending=True).reset_index()
#        p_obs_gen_df_icu = p_obs_gen_df[p_obs_gen_df['obsgen_code'] == 2000].sort_values(by=['OBSGEN_DATE'], ascending=True).reset_index()
#        
##        print(p_proc_df_icu[['PX_DATE', 'px']])
##        print(p_obs_gen_df_icu[['OBSGEN_DATE', 'obsgen_code']], '***', len(p_obs_gen_df_icu))
#        
#        icu_start = np.nan
#        icu_last = np.nan
#        
#        if len(p_proc_df_icu) != 0:
#            icu_start = p_proc_df_icu.loc[0, 'PX_DATE']
#            icu_last = p_proc_df_icu.loc[len(p_proc_df_icu) - 1, 'PX_DATE']
#         
#        
#        if len(p_obs_gen_df_icu) != 0:
#            icu_start_2 = p_obs_gen_df_icu.loc[0, 'OBSGEN_DATE']
#            icu_last_2 = p_obs_gen_df_icu.loc[len(p_obs_gen_df_icu) - 1, 'OBSGEN_DATE']
#            
##            if (icu_start_2 < icu_start) or (len(p_proc_df_icu) == 0):
##                icu_start =  icu_start_2
##            if (icu_last_2 > icu_last) or (len(p_proc_df_icu) == 0):
##                icu_last = icu_last_2
#                
#            if len(p_proc_df_icu) == 0:
#                icu_start =  icu_start_2
#                icu_last = icu_last_2
#            else:
#                if icu_start_2 < icu_start:
#                    icu_start =  icu_start_2
#                if icu_last_2 > icu_last:
#                    icu_last = icu_last_2
#            
##            print(icu_start_2, icu_last_2, '####')
##                
##        print(icu_start, icu_last)
##        print()
##        print()
##        
#        
#        
#        # ventilation info        
#        p_proc_df_vent = p_proc_df[p_proc_df['px'].isin(VENTILATION_CH_CODES)].sort_values(by=['PX_DATE'], ascending=True).reset_index()
#        p_obs_gen_df_vent = p_obs_gen_df[p_obs_gen_df['obsgen_code'] == 3000].sort_values(by=['OBSGEN_DATE'], ascending=True).reset_index()
#        
##        print(p_proc_df_vent[['PX_DATE', 'px']])
##        print(p_obs_gen_df_vent[['OBSGEN_DATE', 'obsgen_code']])
#        
#        vent_start = np.nan
#        vent_last = np.nan
#        vent_type = np.nan
#        
#        if len(p_proc_df_vent) != 0:
#            vent_start = p_proc_df_vent.loc[0, 'PX_DATE']
#            vent_last = p_proc_df_vent.loc[len(p_proc_df_vent) - 1, 'PX_DATE']
#            vent_type = VENTILATION_CH_CODES[p_proc_df_vent.loc[0, 'px']]
#         
#        
#        if len(p_obs_gen_df_vent) != 0:
#            vent_start_2 = p_obs_gen_df_vent.loc[0, 'OBSGEN_DATE']
#            vent_last_2 = p_obs_gen_df_vent.loc[len(p_obs_gen_df_vent) - 1, 'OBSGEN_DATE']
#
#                
#            if len(p_proc_df_vent) == 0:
#                vent_start =  vent_start_2
#                vent_last = vent_last_2
#            else:
#                if vent_start_2 < vent_start:
#                    vent_start =  vent_start_2
#                if vent_last_2 > vent_last:
#                    vent_last = vent_last_2
#                
#            
##            print(vent_start_2, vent_last_2, '####')
##                
##        print(vent_start, vent_last, vent_type)
##        print()
##        print()
#        
#        icu_vent_data.append([p, icu_start, icu_last, vent_start, vent_last, vent_type])
#    
#    site_icu_vent_data=pd.DataFrame(icu_vent_data, 
#                                    columns=['ssid', 'icu_start', 'icu_last', 
#                                             'vent_start', 'vent_last', 'vent_type'])
#
#    site_icu_vent_data.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\icu_vent.csv' % site, index=False)
    
    


icu_vent_df = pd.DataFrame(columns=['ssid', 'icu_start', 'icu_last', 
                                    'vent_start', 'vent_last', 'vent_type'])
for site in SITES:
    site_icu_vent_data = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\icu_vent.csv' % site, header=0)    
    icu_vent_df = pd.concat([icu_vent_df, site_icu_vent_data])

icu_vent_df = pd.merge(cohort_df.drop(columns=['site']), icu_vent_df, how='left', on='ssid')

icu_vent_df['CONFIRM_DATE'] = pd.to_datetime(icu_vent_df['CONFIRM_DATE'], format='%Y-%m-%d')
icu_vent_df['icu_start'] = pd.to_datetime(icu_vent_df['icu_start'], format='%Y-%m-%d')
icu_vent_df['icu_last'] = pd.to_datetime(icu_vent_df['icu_last'], format='%Y-%m-%d')
icu_vent_df['vent_start'] = pd.to_datetime(icu_vent_df['vent_start'], format='%Y-%m-%d')
icu_vent_df['vent_last'] = pd.to_datetime(icu_vent_df['vent_last'], format='%Y-%m-%d')

    
temp_df = icu_vent_df.fillna('None')
is_icu = []
icu_28d = []
icu_60d = []
is_vent = []
vent_28d = []
vent_60d = []
icu_start_days = []
icu_last_days = []
vent_start_days = []
vent_last_days = []
for idx, row in temp_df.iterrows():
    icu_start, vent_start = row['icu_start'], row['vent_start']
    
    confirm_date, icu_start, icu_last, vent_start, vent_last = row['CONFIRM_DATE'], row['icu_start'], row['icu_last'],\
                                                                    row['vent_start'], row['vent_last']
    
    if icu_start != 'None':
        is_icu.append(1)
#        print(confirm_date, icu_start, icu_last, vent_start, vent_last, (icu_start - confirm_date).days)
        icu_start_days.append((icu_start - confirm_date).days)
        icu_last_days.append((icu_last - confirm_date).days)
        
        if (icu_start - confirm_date).days <= 28:
            icu_28d.append(1)
        else: 
            icu_28d.append(0)
            
        if (icu_start - confirm_date).days <= 60:
            icu_60d.append(1)
        else: 
            icu_60d.append(0)
        
    else:
        is_icu.append(0)
        icu_start_days.append(np.nan)
        icu_last_days.append(np.nan)
        
        icu_28d.append(0)
        icu_60d.append(0)
    
    if vent_start != 'None':
        is_vent.append(1)
        vent_start_days.append((vent_start - confirm_date).days)
        vent_last_days.append((vent_last - confirm_date).days)
        
        if (vent_start - confirm_date).days <= 28:
            vent_28d.append(1)            
        else:
            vent_28d.append(0)
        
        if (vent_start - confirm_date).days <= 60:
            vent_60d.append(1)
        else:
            vent_60d.append(0)
            
        
    else:
        is_vent.append(0)
        vent_start_days.append(np.nan)
        vent_last_days.append(np.nan)
        
        vent_28d.append(0)
        vent_60d.append(0)
        
icu_vent_df['is_icu'] = is_icu
icu_vent_df['icu_28d'] = icu_28d
icu_vent_df['icu_60d'] = icu_60d

icu_vent_df['is_vent'] = is_vent
icu_vent_df['vent_28d'] = vent_28d
icu_vent_df['vent_60d'] = vent_60d

icu_vent_df['icu_start_days_since_confirm'] = icu_start_days
icu_vent_df['icu_last_days_since_confirm'] = icu_last_days
icu_vent_df['vent_start_days_since_confirm'] = vent_start_days
icu_vent_df['vent_last_days_since_confirm'] = vent_last_days

icu_vent_df.to_csv(MAIN_DIR + '\\Data\\icu_vent.csv', index=False)

    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
