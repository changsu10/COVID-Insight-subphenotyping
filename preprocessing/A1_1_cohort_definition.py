# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:48:28 2020

@author: chs4001


Cohort definition.


Rules:
    Either any COVID-19 diagnosis code
    OR any COVID-19 lab with "Positive" 
"""


import pandas as pd
import datetime


COVID_DX_ICD10 = [
        'B34.2',
        'B97.21',
        'B97.29',
        'J12.81',
        'U07.1'
        ]

COVID_LAB = [
        '94500-6',
        '94309-2',
        '94507-1',
        '94508-9',
        '94306-8'       
        ]



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
    # --- cohort definition based on DX codes
    diagnosis_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\diagnosis.csv' % site,
                               header=0, parse_dates=True)
        
    diagnosis_df = diagnosis_df[diagnosis_df['dx'].isin(COVID_DX_ICD10)]
    
    
    patient_covid_dx = pd.DataFrame(columns=['ssid', 'CONFIRM_DATE'])
    
    for idx, row in diagnosis_df.iterrows():
        ssid, encounterid, admit_date = row['ssid'], row['encounterid'], row['ADMIT_DATE']
        
        admit_date_str = '%s-%s-%s' % (admit_date[5:], MONTH_MAP[admit_date[2:5]], admit_date[:2])
        
        admit_date_stamp = datetime.datetime.strptime(admit_date_str, '%Y-%m-%d')
        
#        print(ssid, admit_date, admit_date_str, admit_date_stamp)
        
        if (admit_date_stamp < SCREEN_DATE) or (admit_date_stamp > SCREEN_END_DATE):
            # drop COVID diagnosis before Mar 1
            continue
        
        patient_covid_dx = patient_covid_dx.append(
                pd.DataFrame({'ssid':ssid, 'encounterid':encounterid, 'CONFIRM_DATE': admit_date_stamp}, index=['0']), 
                ignore_index=True)
        
    patient_covid_dx = patient_covid_dx.sort_values(by=['ssid', 'CONFIRM_DATE'], ascending=[True, True])
    
    patient_covid_dx = patient_covid_dx.drop_duplicates(subset=['ssid'], keep='first')
    
    patient_covid_dx.to_csv(MAIN_DIR + '\\Data\\cohort\\DxOnly_Site_%s.csv' % site, index=False)  

    print('Site %s DX definition finished, total %s samples' % (site, len(patient_covid_dx)))   
    
    
    
    # --- cohort definition based on Labs
#    # Uncomment this block to extract COVID Labs 
#    lab_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\lab_result_cm.csv' % site,
#                               header=0, parse_dates=True)    
#    lab_df = lab_df[lab_df['lab_loinc'].isin(COVID_LAB)]    
#    lab_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\covid_labs.csv' % site, index=False)
#    print('Site %s COVID Lab extraction finished.' % site)
    
    lab_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\covid_labs.csv' % site,
                               header=0, parse_dates=True)  
    
    lab_df = lab_df[lab_df['result_qual'] == 'POSITIVE']
    
    patient_covid_lab_positive = pd.DataFrame(columns=['ssid', 'CONFIRM_DATE'])
    
    for idx, row in lab_df.iterrows():
        ssid, encounterid, specimen_date, result_qual = row['ssid'], row['encounterid'], row['SPECIMEN_DATE'], row['result_qual']
        
        specimen_date_str = '%s-%s-%s' % (specimen_date[5:], MONTH_MAP[specimen_date[2:5]], specimen_date[:2])
                
        specimen_date_stamp = datetime.datetime.strptime(specimen_date_str, '%Y-%m-%d')
        
        if (specimen_date_stamp < SCREEN_DATE) or (specimen_date_stamp > SCREEN_END_DATE):
            # drop COVID diagnosis before Mar 1
            print(ssid, encounterid, specimen_date, specimen_date_str, '-', result_qual)
            continue
                
        patient_covid_lab_positive = patient_covid_lab_positive.append(
                pd.DataFrame({'ssid':ssid, 'encounterid':encounterid, 'CONFIRM_DATE': specimen_date_stamp}, index=['0']), 
                ignore_index=True)
    
    patient_covid_lab_positive = patient_covid_lab_positive.sort_values(by=['ssid', 'CONFIRM_DATE'], ascending=[True, True])    
    patient_covid_lab_positive = patient_covid_lab_positive.drop_duplicates(subset=['ssid'], keep='first')
    patient_covid_lab_positive.to_csv(MAIN_DIR + '\\Data\\cohort\\LabOnly_Site_%s.csv' % site, index=False)  

    print('Site %s LAB definition finished, total %s samples' % (site, len(patient_covid_lab_positive)))
        
   
    
    
    # --- cohort definition by combining DX and LAB
    patient_covid_combined = pd.concat([patient_covid_dx, patient_covid_lab_positive])
    patient_covid_combined = patient_covid_combined.sort_values(by=['ssid', 'CONFIRM_DATE'], ascending=[True, True])
    patient_covid_combined = patient_covid_combined.drop_duplicates(subset=['ssid'], keep='first')
    patient_covid_combined.to_csv(MAIN_DIR + '\\Data\\cohort\\Lab_DX_Site_%s.csv' % site, index=False)
    print('Site %s combined definition finished, total %s samples' % (site, len(patient_covid_combined)))  
    
    
    
    

