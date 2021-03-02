# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:39:32 2020


Extract socioeconomic data


@author: chs4001
"""


import pandas as pd
import numpy as np


MAIN_DIR = "W:\\WorkArea-chs4001"
SITES = [1, 3, 4, 5, 8]
#SITES = [5, 8]




# load cohort info
cohort_df = pd.read_csv(MAIN_DIR + '\\Data\\_demographics_deduplicated.csv', header = 0)
cohort_df = cohort_df[['ssid', 'CONFIRM_DATE']]
patient_list = {}
for p in cohort_df['ssid'].values.tolist():
    patient_list[p] = True

print(len(patient_list))

## --- load address data ---- #
#all_add_df = pd.DataFrame(columns=['ssid', 'ZIP_CODE'])
#for site in SITES:
#    add_df = pd.read_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\CSV\\lds_address_history.csv' % site, header = 0,
#                         dtype={'address_zip5': str, 'address_zip9': str})
#    
#    
#    add_df = add_df[add_df['ssid'].isin(patient_list)]
#    
#    site_patients = list(set(add_df['ssid'].values.tolist()))
#    
#    site_add_data = []
#    
#    for p in site_patients:        
##        if p not in [
##            '2ec43f0c-f581-4059-ad7d-48e64ce8bd8f',
##                '4968de1f-444c-4ade-b91a-3fb31e7e7f45',
##             '38d972dc-697e-47f0-ae37-34800e6a9ff3',
##             '29fb68f2-50bd-4f49-abe2-d9851577d46c'
##             ]:
##            continue
#        
#        p_add_df = add_df[add_df['ssid'] == p]
#        
#        p_add_df = p_add_df.dropna(subset=['address_zip5', 'address_zip9'], how='all').reset_index()
#        
#        if len(p_add_df) == 0:
#            continue
#        
#        p_add_df = p_add_df.fillna('None')
#        
#        p_add_df['ADDRESS_PERIOD_START'] = pd.to_datetime(p_add_df['ADDRESS_PERIOD_START'], format='%d%b%Y')
#        p_add_df = p_add_df.sort_values(by=['ADDRESS_PERIOD_START'], ascending=False)
#
#        zip5, zip9 = p_add_df.loc[0, ['address_zip5', 'address_zip9']]
#        if zip5 == 'None':
#            
#            if zip9 != 'None':
#                zip9 = '0' * (9 - len(zip9)) + zip9
#                zip5 = zip9[:5]          
#        
#        site_add_data.append([p, zip5])
#        
#    site_add_df = pd.DataFrame(site_add_data, columns=['ssid', 'ZIP_CODE'])
#    site_add_df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\address_.txt' % site, index=False) 
#    
#    all_add_df = pd.concat([all_add_df, site_add_df])
#        
#all_add_df.to_csv(MAIN_DIR + '\\Data\\address.txt', index=False) 


# load existing address data
all_add_df = pd.read_csv(MAIN_DIR + '\\Data\\address.txt', header=0, 
                         dtype={'ZIP_CODE': str})
print(all_add_df)









## ----- load SD0H data, indexed by zip code ------ #
### load zip_zcta map
##zip_zcta_df = pd.read_excel(MAIN_DIR + '\\Data\\SDoH\\zip_to_zcta_2019.xlsx', index_col=None, header=0,
##                            dtype={'ZIP_CODE': str, 'ZCTA':str})
##
### load zcta_SDoH data
##zcta_SDoH_df = pd.read_csv(MAIN_DIR + '\\Data\\SDoH\\SDoH Variables.csv', header=0,
##                            dtype={'ZCTA': str})
### update missing digits of zcta
##for idx, row in zcta_SDoH_df.iterrows():
##    zcta = row['ZCTA']
##    if len(zcta) == 3:
##        zcta = '00' + zcta
##    if len(zcta) == 4:
##        zcta = '0' + zcta
##    zcta_SDoH_df.loc[idx, 'ZCTA'] = zcta
##    
### merge zcta_SDoH and zip_zcta data
##zip_SDoH_df = pd.merge(zip_zcta_df, zcta_SDoH_df, how='left', on='ZCTA')
##zip_SDoH_df.to_csv(MAIN_DIR + '\\Data\\SDoH\\zip_SDoH.txt', index=False)
#
# load existing zip_SDoH data
zip_SDoH_df = pd.read_csv(MAIN_DIR + '\\Data\\SDoH\\zip_SDoH.txt', header=0, 
                          dtype={'ZIP_CODE': str, 'ZCTA':str})
print(zip_SDoH_df)

patient_SDoH_df = pd.merge(all_add_df, zip_SDoH_df, how='left', on='ZIP_CODE')

patient_SDoH_df.to_csv(MAIN_DIR + '\\Data\\patient_SDoH.csv', index=False)




















