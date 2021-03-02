# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:12:28 2020

@author: chs4001

1. Add lab name

2. Unit conversion

"""

import pandas as pd
import numpy as np

from AA_lab_unit_conversion import lab_unit_conversion


import datetime


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

TARGET_LAB_CODES = {}
for lab in TARGET_LABs:
    lab_codes = TARGET_LABs[lab]
    for c in lab_codes:
        TARGET_LAB_CODES[c] = lab

#print(TARGET_LAB_CODES)


MAIN_DIR = "W:\\WorkArea-chs4001"

SITES = [1, 3, 4, 5, 8]
#SITES = [8]


#VERSION = '1002'
VERSION = '1006'






for site in SITES:
    print("Site %s" % site)
    # load lab data
    lab_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_labs_%s.csv' % (site, VERSION), header=0)
    
#    for idx, row in lab_df.iterrows():
#        lab_loinc = row['lab_loinc']
#        lab_df.loc[idx, 'lab_name'] = TARGET_LAB_CODES[lab_loinc]
    
    lab_loinc_list = lab_df['lab_loinc'].values.tolist()
    lab_name_list = []
    for lab_loinc in lab_loinc_list:
        lab_name_list.append(TARGET_LAB_CODES[lab_loinc])
        
    lab_df['lab_name'] = lab_name_list
    
#    lab_df = lab_df[lab_df['lab_name'] == 'Bicarbonate'] 
#    print(lab_df)
    
    
    converted_values = []
    converted_unit = []
    for idx, row in lab_df.iterrows():
        lab_name, loinc_code, lab_value, lab_unit = row['lab_name'], row['lab_loinc'], row['result_num'], row['result_unit']
        lab_value, lab_unit = lab_unit_conversion(site=site, lab_name=lab_name, loinc_code=loinc_code, lab_value=lab_value, lab_unit=lab_unit)
        
        converted_values.append(lab_value)
        converted_unit.append(lab_unit)
        
    lab_df['converted_result_num'] = converted_values
    lab_df['converted_result_unit'] = converted_unit
        
    
    
    
        
    lab_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_labs_%s_annotated_converted.csv' % (site, VERSION))
























