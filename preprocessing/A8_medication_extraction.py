# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:49:54 2020


Extract medication

@author: chs4001
"""

import pandas as pd
import numpy as np
import datetime


MAIN_DIR = "W:\\WorkArea-chs4001"

SCREEN_DATE = datetime.datetime.strptime('2020-03-01', '%Y-%m-%d')


SITES = [1, 3, 4, 5, 8]
#SITES = [5, 8]




def has_overlap(list_a, list_b):
    # return True if list_a and list_b have overlap
    for i in list_a:
        if i in list_b:
            return True
    return False    
    



# load medication vacabulary
med_vac_df = pd.read_csv(MAIN_DIR + "\\Medications.csv", header=0, encoding='ISO-8859-1',
                         dtype={'RxNorm': str})
med_varcabulary = {}
all_codes = []
for idx, row in med_vac_df.iterrows():
    med, rx = row['Medication'], row['RxNorm']
    if med not in med_varcabulary:
        med_varcabulary[med] = [rx]
    else:
        med_varcabulary[med].append(rx)
    all_codes.append(rx)




# load cohort info
cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', header = 0)
cohort_df = cohort_df[['ssid', 'CONFIRM_DATE', 'site']]
cohort_df['CONFIRM_DATE'] = pd.to_datetime(cohort_df['CONFIRM_DATE'], format='%Y-%m-%d')
patient_list = {}
for p in cohort_df['ssid'].values.tolist():
    patient_list[p] = True
    
print(len(patient_list))


  
## ------ initial raw data filtering ---------- #
#for site in SITES:
#    print('Site ', site, '...')
#
#    prescrib_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\prescribing.csv' % site, header = 0,
#                     dtype = {'rxnorm_cui': str})
#    print(prescrib_df.shape)
#    
#    prescrib_df = prescrib_df[prescrib_df['ssid'].isin(patient_list)]
#    print(prescrib_df.shape)
#    
#    prescrib_df['RX_ORDER_DATE'] = pd.to_datetime(prescrib_df['RX_ORDER_DATE'], format='%d%b%Y')
#    prescrib_df = prescrib_df[prescrib_df['RX_ORDER_DATE'] >= SCREEN_DATE]
#    print(prescrib_df.shape)
#    
#    # filtering by code
#    prescrib_df = prescrib_df[prescrib_df['rxnorm_cui'].isin(all_codes)]
#    print(prescrib_df.shape)
#    
#    prescrib_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_medications.csv' % site, index=False)





non_vaso_meds = [
        'sarilumab',
        'tocilizumab',
        'predniSONE',
        'methylprednisolone',
        'dexamethasone',
        'hydrocortisone',
        'enoxaparin',
        'heparin',
        'Ceftriaxone',
        'azithromycin',
        'piperacillin_tazobactam',
        'meropenem',
        'vancomycin',
        'doxycycline',
        'Hydroxychloroquine',
        'Remdesivir'
        ]
## ------- extract med data --------#
#for site in SITES:
#    print('Site ', site, '...')
#    
#    site_rx_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\target_medications.csv' % site, header=0,
#                             dtype={'rxnorm_cui': str})
#    site_rx_df['RX_ORDER_DATE'] = pd.to_datetime(site_rx_df['RX_ORDER_DATE'], format='%Y-%m-%d')
#    
#    print(site_rx_df.shape)
#
#    # load site cohort time info: confirm date, COVID-IP date
#    site_cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\COVID_IP_ED.csv' % site)
#    site_cohort_df['confirm_date'] = pd.to_datetime(site_cohort_df['confirm_date'], format='%Y-%m-%d')
#    site_cohort_df['date_CIP'] = pd.to_datetime(site_cohort_df['date_CIP'], format='%Y-%m-%d')
#    
#    rx_data = []
#    for idx, row in site_cohort_df.iterrows():
#        
#        
#        p, confirm_date, CIP_date = row['ssid'], row['confirm_date'], row['date_CIP']
#        
##        if p not in [
##                '058f0ba8-c751-42d6-a9ea-db99465a98dd',
##                     '05979255-b913-4ef9-a306-f210541d1eb8',
##                     '051b7db5-9696-42ec-9e4e-8b1f74f7e016'
##                     ]:
##            continue
#        
##        print(p, confirm_date, CIP_date, pd.isnull(CIP_date))
#        p_med_data = [p] + [0] * (len(non_vaso_meds) + 1)
#        
#        p_rx_df = site_rx_df[site_rx_df['ssid'] == p]
#        
#        p_rx_df = p_rx_df[p_rx_df['RX_ORDER_DATE'] >= confirm_date]
#        
#        p_meds = p_rx_df['rxnorm_cui'].values.tolist()
##        print(p_rx_df[['RX_ORDER_DATE', 'rxnorm_cui']])
#        
#        # non Vasopressor medication
#        for i in range(len(non_vaso_meds)):
#            rx = non_vaso_meds[i]
#            
#            if has_overlap(med_varcabulary[rx], p_meds) == True:
#                p_med_data[i + 1] = 1
#           
#        # Vasopressor medication        
#        p_rx_df = p_rx_df[p_rx_df['RX_ORDER_DATE'] >= CIP_date]
##        p_rx_df = p_rx_df[p_rx_df['RX_ORDER_DATE'] >= np.datetime64('NaT')] # test
##        print(p_rx_df[['RX_ORDER_DATE', 'rxnorm_cui']])
#        if has_overlap(med_varcabulary['Vasopressor'], p_rx_df['rxnorm_cui'].values.tolist()) == True:
#            p_med_data[-1] = 1
#        
#        rx_data.append(p_med_data)
#        
#
#        
#    rx_df = pd.DataFrame(rx_data, columns=['ssid'] + non_vaso_meds + ['Vasopressor'])
#    
#    rx_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\medications_.csv' % site, index=False)
    
    


# final combination
all_rx_df = pd.DataFrame(columns=['ssid'] + non_vaso_meds + ['Vasopressor'])
for site in SITES:
    rx_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\medications_.csv' % site, header=0)
    all_rx_df = pd.concat([all_rx_df, rx_df])
    
# update
corticosteroids = []
antibiotics = []

cort_meds = ['predniSONE',
        'methylprednisolone',
        'dexamethasone',
        'hydrocortisone']

anti_meds = ['Ceftriaxone',
        'azithromycin',
        'piperacillin_tazobactam',
        'meropenem',
        'vancomycin',
        'doxycycline']

for idx, row in all_rx_df.iterrows():
    if np.sum(row[cort_meds].values) > 0:
        corticosteroids.append(1)
    else:
        corticosteroids.append(0)
    
    if np.sum(row[anti_meds].values) > 0:
        antibiotics.append(1)
    else:
        antibiotics.append(0)
    
all_rx_df['corticosteroids'] = corticosteroids
all_rx_df['antibiotics'] = antibiotics
col_order = [
        'ssid',
        
        'sarilumab',
        'tocilizumab',
        
        'corticosteroids',
        'predniSONE',
        'methylprednisolone',
        'dexamethasone',
        'hydrocortisone',
        
        'enoxaparin',
        'heparin',
        
        'antibiotics',
        'Ceftriaxone',
        'azithromycin',
        'piperacillin_tazobactam',
        'meropenem',
        'vancomycin',
        'doxycycline',
        
        'Hydroxychloroquine',
        'Remdesivir',
        'Vasopressor'
        ]
all_rx_df = all_rx_df[col_order]
all_rx_df.to_csv(MAIN_DIR + '\\Data\\medications_.csv', index=False)

    
    
    
    







































