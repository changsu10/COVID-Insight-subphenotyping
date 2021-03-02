# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:14:19 2020

@author: chs4001

Extract lab data.

    1. Determine lab collection window
    
        1) If has IP: Pick the first value between Confirm_date and IP_date

        2) If has no IP, but has ED: Pick the first value between Confirm_date and 14-days after confirmaiton

        3) If has no value within collection window: go back to 3 days before confirmation and take value close to Confirm_date
    
    2. Perform unit conversion for the picked value
    
    
"""


import pandas as pd
import numpy as np

from time import time
import datetime

import os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass


TARGET_LABs = {
        'venous_lactate': ['2519-7', '19240-1'],  # mmol/L
        'creatinine': ['2160-0', '21232-4', '38483-4'],  # mg/dL
        'white_blood_cell_count': ['33256-9', '12227-5', '6690-2', '6743-9', '806-0', '813-6', '26464-8', '26465-5'],  # x10^3 cells per uL
        'lymphocyte_count': ['731-0', '732-8', '26474-7', '30364-4'],  # x10^3 cells per uL
        'platelet_count': ['777-3', '778-1', '26515-7'],  # x10^3 cells per uL
        'bilirubin': ['33898-8', '1971-1', '14152-3', '1968-7', '15152-2', '58941-6',
                      '1974-5', '1975-2', '1978-6'],  # mg/dL
        'aspartate_aminotransferase': ['1920-8', '30239-8'],  # U/L
        'alanine_aminotransferase': ['1742-6', '1743-4'],  # U/L
        'creatine_kinase': ['2157-6'],  # U/L  
        'prothrombin_time': ['5964-2', '5902-2', '5901-4', '33356-7', '5959-2', '53748-0'],  # s
        'interleukin-6': ['26881-3'],  # pg/mL
#        'high-sensitive_CRP': ['71426-1', '30522-7'],  # mg/L
        'Ferritin': ['2276-4'],  # ng/mL
        'D-dimer': ['48065-7'],  # ug/mL
#        'high-sensitive_cardiac_troponin_T': ['67151-1'],  # ng/mL
        'procalcitonin': ['33959-8'],  # ng/mL
        'albumin': ['1751-7'],  # g/dL
        'red_blood_cell_distribution_width': ['30384-2', '30385-9', '788-0'],
        'neutrophil_count': ['26499-4', '751-8', '753-4'],  # x10^3 cells per uL
        
#        'proteinuria': ['14959-1', '14957-5', '2890-2', '50561-0'],
        'C-reactive_protein': ['71426-1', '1988-5', '30522-7'],
        'cardiac_troponin_I': ['42757-5', '10839-9'], 
        'cardiac_troponin_T': ['6598-7', '67151-1'],
        
         # additional labs, added in Oct 6  
        'LDH': ['14804-9', '14805-6', '2529-6', '2532-0', '60017-1'], 
        'Bicarbonate': ['16551-4', '19223-7', '2026-3', '2027-1', '2028-9', '20565-8'], 
        'BUN': ['3094-0', '6299-2'], 
        'CHLORIDE': ['2069-3', '2075-0', '41649-5', '41650-3', '51590-8'], 
        'ESR': ['30341-2', '4537-7'], 
        'GLUCOSE': ['2339-0', '2345-7', '32016-8', '41651-1', '41652-9', '41653-7'], 
        'HEMOGLOBIN': ['20509-6', '30313-1', '30350-3', '30351-1', '30352-9', '718-7'], 
        'SODIUM': ['2947-0', '2951-2', '32717-1', '39791-9', '39792-7'], 
        'Neutrophils.band': ['763-3', '26507-4', '764-1', '26508-2', '35332-6'], 
        
        'Oxygen_saturation': ['51733-4', '51732-6', '51731-8', '2708-6', '28642-7', '28643-5']
        }


SCREEN_DATE = datetime.datetime.strptime('2020-03-01', '%Y-%m-%d')
SCREEN_END_DATE = datetime.datetime.strptime('2020-06-12', '%Y-%m-%d')

MONTH_MAP = {
        'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06',
        'JUL':'07', 'AUG':'08', 'SEP':'09', 'OCT':'10', 'NOV':'11', 'DEC':'12'      
        }


MAIN_DIR = "W:\\WorkArea-chs4001"

SITES = [1, 3, 4, 5, 8]
#SITES = [4, 5, 8]


#VERSION = '0922'
#VERSION = '1002'
VERSION = '1006'


OBSERVATION_WIN_LOWER = 3
#OBSERVATION_WIN_UPPER = 1


for site in SITES:
    print('----- Processing Site %s...' % site)
    
    time1 = time()
    
    # load cohort
    cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\COVID_IP_ED.csv' % site)
        
    
    
    # load lab data
    lab_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_labs_%s_annotated_converted.csv' % (site, VERSION), header=0)
    lab_df = lab_df[['ssid', 'SPECIMEN_DATE', 'lab_name', 'lab_loinc', 
                     'result_num', 'result_unit', 'converted_result_num', 'converted_result_unit']]    
    lab_df['SPECIMEN_DATE'] = pd.to_datetime(lab_df['SPECIMEN_DATE'], format='%Y-%m-%d')
    
    
    lab_df = lab_df.dropna(subset=['converted_result_num'])
    
    
    
    
    # --- begin to extract data --- #
#    cohort_df = cohort_df.head(100)
    print('Originally %s patients...' % len(cohort_df))
    
    # excluded patients with no IP or ED encounter
    cohort_df = cohort_df[(cohort_df['has_IP'] != 0) | (cohort_df['has_ED'] != 0)].reset_index()
    print('Has IP or ED: %s' % len(cohort_df))
    print('Has IP: %s' % len(cohort_df[cohort_df['has_IP'] != 0]))
    
           
    lab_value_lists = {}  # key: lab_nam, value: a list of values along all patients
    lab_collection_day = {}  # key: lab_nam, value: a list of days (to confirm) when the specific value was collected
    for lab in TARGET_LABs:
        lab_value_lists[lab] = []
        lab_collection_day[lab] = []
      
    
    patient_list = []  # list of patient
   
    for idx, row in cohort_df.iterrows():
        ssid, confirm_date, date_CIP,	date_CED, has_IP, has_ED = row['ssid'], row['confirm_date'], row['date_CIP'], row['date_CED'],\
                                                                row['has_IP'], row['has_ED']
                                                                
#        print(ssid, confirm_date, date_CIP,	date_CED, has_IP, has_ED)
        confirm_date = datetime.datetime.strptime(confirm_date, '%Y-%m-%d')
        
        obw_lower = confirm_date - datetime.timedelta(days=OBSERVATION_WIN_LOWER)
        
        
        p_lab_df = lab_df[lab_df['ssid'] == ssid]  # all lab data of patient p
        
        # lab data from x-day before comfirm to confirm_date
        secondary_p_lab_df = p_lab_df[(p_lab_df['SPECIMEN_DATE'] >= obw_lower) & (p_lab_df['SPECIMEN_DATE'] < confirm_date)].sort_values(by='SPECIMEN_DATE', ascending=False)
#        print('---', ssid, confirm_date, obw_lower)
#        print(secondary_p_lab_df)
        
        if has_IP == 1: # has IP
            obw_upper = datetime.datetime.strptime(date_CIP, '%Y-%m-%d')
        else:  # has ED, no IP
            obw_upper = confirm_date + datetime.timedelta(days=14)
            
        primary_p_lab_df = p_lab_df[(p_lab_df['SPECIMEN_DATE'] >= confirm_date) & (p_lab_df['SPECIMEN_DATE'] <= obw_upper)].sort_values(by='SPECIMEN_DATE', ascending=True)
        
#        print('###', confirm_date, date_CIP)
#        print(primary_p_lab_df)
            
        for lab in lab_value_lists:
            # primary value
            temp_df = primary_p_lab_df[primary_p_lab_df['lab_name'] == lab].reset_index()
            if len(temp_df) != 0:
#                print('## primary')
                row = temp_df.iloc[0]
                
            # secondary value (has no data after confirm, then go back to x day before confirm)
            else:
                temp_df = secondary_p_lab_df[secondary_p_lab_df['lab_name'] == lab].reset_index()
                if len(temp_df) == 0:
#                    print('no data')
                    lab_value_lists[lab].append(np.nan)
                    lab_collection_day[lab].append(np.nan)
                    continue
                else:
#                    print('** secondary')
                    row = temp_df.iloc[0]
            
            loinc_code, lab_value, lab_unit, collection_date= row['lab_loinc'], row['converted_result_num'], row['result_unit'], row['SPECIMEN_DATE']
            
            lab_value_lists[lab].append(lab_value)
            
            lab_collection_day[lab].append((collection_date - confirm_date).days)
            
#            print(collection_date, confirm_date, (collection_date - confirm_date).days)
            

                
            
    for lab in lab_value_lists:
        cohort_df[lab] = lab_value_lists[lab]
        
    # save data    
    cohort_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\labs_%s_window_-%s_14.csv' % (site, VERSION, OBSERVATION_WIN_LOWER), index=False)

        
    for lab in lab_collection_day:
        cohort_df[lab] = lab_collection_day[lab]
        
    # save data    
    cohort_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\labs_%s_window_-%s_14_collection_date.csv' % (site, VERSION, OBSERVATION_WIN_LOWER), index=False)

        
    
        
    time2 = time()        
    print('Finished Site %s with %s mins....' % (site, int((time2 - time1) / 60))) 
    
        





















