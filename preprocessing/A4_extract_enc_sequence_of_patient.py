# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:15:00 2020

@author: chs4001


Extract encounter sequence of each patient.

    1. between Mar 1, 2020 and Jun 12, 2020
    
"""



import pandas as pd
import datetime
import json
import numpy as np



SCREEN_DATE = datetime.datetime.strptime('2020-03-01', '%Y-%m-%d')
SCREEN_END_DATE = datetime.datetime.strptime('2020-06-12', '%Y-%m-%d')

MONTH_MAP = {
        'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06',
        'JUL':'07', 'AUG':'08', 'SEP':'09', 'OCT':'10', 'NOV':'11', 'DEC':'12'      
        }


MAIN_DIR = "W:\\WorkArea-chs4001"

SITES = [1, 3, 4, 5, 8]

#SITES = [1]


for site in SITES:
    # load cohort
    cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\cohort\\Lab_DX_Site_%s.csv' % site)
    cohort = {}
    for idx, row in cohort_df.iterrows():
        p, confirm_date = row['ssid'], row['CONFIRM_DATE']
        cohort[p] = {
                'CONFIRM_DATE': confirm_date,
                'ENC_seq': []
              }
        
    
        
    # load encounter data
    enc_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\encounter.csv' % site,
                               header=0, parse_dates=True)   
    # filtering by ssid
    cohort_enc_df = enc_df[enc_df['ssid'].isin(cohort)]
    
    
    # update 
    for idx, row in cohort_enc_df.iterrows():
        if site == 1:
            ssid, encounterid, admit_date, enc_type = row['ssid'], row['ENCOUNTERID'], row['ADMIT_DATE'], row['ENC_TYPE']
        else:
            ssid, encounterid, admit_date, enc_type = row['ssid'], row['encounterid'], row['ADMIT_DATE'], row['enc_type']
            
        
        comfirm_date = cohort[ssid]['CONFIRM_DATE']
        confirm_date_stamp = datetime.datetime.strptime(comfirm_date, '%Y-%m-%d')
        
        admit_date_str = '%s-%s-%s' % (admit_date[5:], MONTH_MAP[admit_date[2:5]], admit_date[:2])
        
        admit_date_stamp = datetime.datetime.strptime(admit_date_str, '%Y-%m-%d')
        
#        print(ssid, encounterid, admit_date, enc_type, admit_date_stamp, confirm_date_stamp)
        
        if (admit_date_stamp < SCREEN_DATE) or (admit_date_stamp > SCREEN_END_DATE):
            # drop COVID diagnosis before Mar 1
            continue
        
        days_since_confirm = int((admit_date_stamp - confirm_date_stamp).days)
        
        cohort[ssid]['ENC_seq'].append([encounterid, enc_type, days_since_confirm, admit_date_str])
        
    
    # sort the encounter sequence of each patient by time
    for p in cohort:
        enc_seq = cohort[p]['ENC_seq']
        time_seq = []
        for entry in enc_seq:
            time_seq.append(entry[2])
            
        enc_seq = np.array(enc_seq)
        time_seq = np.array(time_seq)
        
        order = np.argsort(time_seq)
        
        enc_seq = enc_seq[order].tolist()
        
        cohort[p]['ENC_seq'] = enc_seq


        
    with open(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\enc_sequences.json' % site, 'w') as outf:
        json.dump(cohort, outf, indent=4)
        
    print('Site %s finished with %s patients...' % (site, len(cohort)))
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    