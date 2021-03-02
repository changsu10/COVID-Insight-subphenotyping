# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:01:11 2020

@author: chs4001


Investigate the most probable encounter type of COVID confirmation
"""


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




COVID_WINDOW = 14  # day
PRE_WINDOW = 0  # 0 day or -1 day

MAIN_DIR = "W:\\WorkArea-chs4001"


SITES = [1, 3, 4, 5, 8]
#SITES = [1]


for site in SITES:
    with open(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\enc_sequences.json' % site) as jsonf:
        cohort_enc_seq = json.load(jsonf)
        
        N = len(cohort_enc_seq)
        
        data_list = []
        for p in cohort_enc_seq:  
            
            confirm_date = cohort_enc_seq[p]['CONFIRM_DATE']
                       
            day_IP = -999          
            IP_date = np.nan          
            IP_enc_id = np.nan
            has_IP = 0
            
            day_ED = -999           
            ED_date = np.nan          
            ED_enc_id = np.nan
            has_ED = 0
            
            
            
            IP_flag = False
            ED_flag = False
            
            for entry in cohort_enc_seq[p]['ENC_seq']:
                enc_time = int(entry[2])
                enc_type = entry[1]
                enc_id = entry[0]
                enc_date = entry[3]
                
                if enc_time < PRE_WINDOW:
                    continue
                
                if enc_time > COVID_WINDOW:
                    break
                
                if (enc_type == 'IP') or (enc_type == 'EI'):
                    if IP_flag != True:
                        day_IP = enc_time
                        IP_enc_id = enc_id
                        IP_flag = True
                        IP_date = enc_date
                        has_IP =1
                
                
                if enc_type == 'ED':
                    if IP_flag == True and enc_time > day_IP:
                        continue
                    
                    if ED_flag != True:
                        day_ED = enc_time
                        ED_enc_id = enc_id
                        ED_flag = True
                        ED_date = enc_date
                        has_ED = 1
                    
        
            p_list = [p, confirm_date, day_IP, IP_date, IP_enc_id, day_ED, ED_date, ED_enc_id, has_IP, has_ED]
            data_list.append(p_list)
            
        df = pd.DataFrame(data_list, columns=['ssid', 'confirm_date', 'day_CIP', 'date_CIP', 'enc_id_CIP',
                                              'day_CED', 'date_CED', 'enc_id_CED', 'has_IP', 'has_ED'])
     
        
        df.to_csv(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\COVID_IP_ED.csv' % site, index=False)
                
                
        
#        # statistic plots
#        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
#        
#        n_no_IP = len(df[df['day_CIP'] == -999])
#        ax0.bar(['has_COVID_IP', 'has_no_COVID_IP'], [len(df)-n_no_IP, n_no_IP])
#        ax0.set_title('Has IP addmision?')
#        ax0.set_ylabel('# of patients')
#        
#        ip_days = df[df['day_CIP'] != -999]['day_CIP'].values
#        ax1.hist(ip_days, bins=len(set(ip_days.tolist())))
#        ax1.set_title('Days of 1st IP since COVID confirm')
#        ax1.set_xlabel('Days')
#        ax1.set_ylabel('# of patients')
#                       
##        ax2.hist(df[df['day_CIP'] != -1]['ED_within_window'].values, bins=range(10), label='Has IP', alpha=0.5)
##        ax2.hist(df[df['day_CIP'] == -1]['ED_within_window'].values, bins=range(10), label='Has no IP', alpha=0.5)
#        width = .35
#        x = np.array([0, 1, 2, 3])
#        df_has_ip = df[df['day_CIP'] != -999]
#        ax2.bar(x - width/2,
#                [len(df_has_ip[df_has_ip['ED_within_window'] == 0]),
#                 len(df_has_ip[df_has_ip['ED_within_window'] == 1]),
#                 len(df_has_ip[df_has_ip['ED_within_window'] == 2]),
#                 len(df_has_ip[df_has_ip['ED_within_window'] >= 3])
#                 ], 
#                width, label='Has IP', alpha=0.5)
#    
#        df_no_ip = df[df['day_CIP'] == -999]
#        ax2.bar(x + width/2,
#                [len(df_no_ip[df_no_ip['ED_within_window'] == 0]),
#                 len(df_no_ip[df_no_ip['ED_within_window'] == 1]),
#                 len(df_no_ip[df_no_ip['ED_within_window'] == 2]),
#                 len(df_no_ip[df_no_ip['ED_within_window'] >= 3])
#                 ], 
#                width, label='Has no IP', alpha=0.5)
#        ax2.set_xticks(x)
#        ax2.set_xticklabels(['0', '1', '2', '>=3'])
#        
#        
#        
#        
#        ax2.legend()
#        ax2.set_xlabel('# of ED encounters')
#        ax2.set_ylabel('# of patients')
#        ax2.set_title('ED before 14 days / first IP')
#                       
#        plt.savefig(MAIN_DIR + '\\Data\\INSIGHT_COVID_Site%s\\COVID_IP_1.png' % site, dpi=300)
#        plt.close()
        
        
        
        
               
              

                
                
                
                    
            
            
