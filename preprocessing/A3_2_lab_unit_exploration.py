# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:10:11 2020

@author: chs4001

Explore units of laboratory values.

"""


import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
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
        'high-sensitive_CRP': ['71426-1', '30522-7'],  # mg/L
        'Ferritin': ['2276-4'],  # ng/mL
        'D-dimer': ['48065-7'],  # ug/mL
        'high-sensitive_cardiac_troponin_T': ['67151-1'],  # ng/mL
        'procalcitonin': ['33959-8'],  # ng/mL
        'albumin': ['1751-7'],  # g/dL
        'red_blood_cell_distribution_width': ['30384-2', '30385-9', '788-0'],
        'neutrophil_count': ['26499-4', '751-8', '753-4'],  # x10^3 cells per uL
        
        'proteinuria': ['14959-1', '14957-5', '2890-2', '50561-0'],
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
SITES = [1, 3, 4, 5, 8]


#VERSION = '0922'
#VERSION = '1002'
VERSION = '1006'



for site in SITES:

#    lab_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_labs_%s.csv' % (site, VERSION), header=0)
    lab_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_labs_%s_annotated_converted.csv' % (site, VERSION), header=0)
    
    lab_df = lab_df[['ssid', 'specimen_source', 'lab_loinc', 'SPECIMEN_DATE', 'SPECIMEN_TIME',
                     'RESULT_DATE', 'RESULT_TIME', 'result_qual', 'result_snomed',
                     'result_num', 'result_modifier', 'result_unit',
                     'converted_result_num', 'converted_result_unit'
                     ]]
    
    
    
    if VERSION == '0922':
        time_format = '%d%b%Y'
    else:
        time_format = '%Y-%m-%d'
    lab_df['SPECIMEN_DATE'] = pd.to_datetime(lab_df['SPECIMEN_DATE'], format=time_format)
    
    lab_df = lab_df[lab_df['SPECIMEN_DATE'] >= SCREEN_DATE]


    
    for lab in TARGET_LABs:
        
#        if lab not in ['albumin']:
#            continue
        
        for lab_code in TARGET_LABs[lab]:
            
            temp_df = lab_df[lab_df['lab_loinc'] == lab_code]
            
#            print(temp_df[temp_df['result_num'] > 20])
            
            if len(temp_df) == 0:
                continue
            
            save_dir = MAIN_DIR + '\\Results\\Lab_unit_exploration_annotated_converted\\%s\\' % lab 
#            save_dir = MAIN_DIR + '\\Results\\Lab_unit_exploration\\%s\\' % lab 
            mkdir(save_dir)
            
            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            ax0.hist(temp_df['converted_result_num'].dropna().values, bins=100)
#            ax0.hist(temp_df['result_num'].dropna().values, bins=100)
            ax0.set_xlabel('Lab value')
            ax0.set_ylabel('Count')
            ax0.set_title('%s, code: %s' % (lab, lab_code))
            
            temp_df = temp_df.fillna(value={'converted_result_unit': 'None'})
#            temp_df = temp_df.fillna(value={'result_unit': 'None'})

            temp_df['converted_result_unit'].value_counts().plot('bar', ax=ax1)
#            temp_df['result_unit'].value_counts().plot('bar', ax=ax1)
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Lab unit')
            
            plt.savefig(save_dir + '%s_code_%s_Site_%s.png' % (lab, lab_code, site), dpi=300)
            plt.close()
            
            print('Site %s, Finished %s, code: %s...' % (site, lab, lab_code))
            
            

        
            
    

















