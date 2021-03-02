# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:45:44 2020

@author: chs4001
"""

import numpy as np

def lab_unit_conversion(site, lab_name, loinc_code, lab_value, lab_unit):
    if lab_name == 'albumin':
        if lab_value > 100 or lab_unit == 'mg/dL':
            lab_value = lab_value / 1000
            lab_unit = 'g/dL'
           
    # cardiac_troponin_T
    if lab_name == 'cardiac_troponin_T':
        if loinc_code == '67151-1':
#            print(site, lab_name, loinc_code, lab_value, lab_unit)
            lab_value = lab_value / 1000
            lab_unit = 'ng/mL'
#            print(lab_value, lab_unit)
        
    if lab_name == 'C-reactive_protein':
        if loinc_code == '1988-5':
            if site == 3 and (lab_unit == 'mg/dL' or lab_unit == 'NI'):                                
                lab_value = lab_value * 10
                lab_unit = 'mg/L'
                
            if site == 5:               
                lab_value = lab_value * 10
                lab_unit = 'mg/L'
                
            if site == 8 and lab_unit == 'mg/dL':      
                lab_value = lab_value * 10
                lab_unit = 'mg/L'
                
        if loinc_code == '30522-7':
            if site == 1 or site == 5:
                lab_value = np.nan
                
    if lab_name == 'D-dimer':
        if lab_value <= 100:
            lab_value = lab_value * 1000
            lab_unit = 'ng/mL'
            
    if lab_name == 'lymphocyte_count':
        if site == 1:
            if loinc_code == '731-0' or  loinc_code == '732-8':
                lab_value = np.nan
            
        if lab_value > 1000:
            lab_value = np.nan
            
    if lab_name == 'neutrophil_count':
        if lab_value > 1000:
            lab_value = np.nan
            
    
    if lab_name == 'proteinuria':  # too many units
        lab_value = np.nan
        
    if lab_name == 'red_blood_cell_distribution_width':
        if loinc_code == '30384-2':
            lab_value = np.nan
            
    if lab_name == 'white_blood_cell_count':
        if loinc_code == '26465-5' or loinc_code == '806-0':
            lab_value = np.nan
            
        if loinc_code == '6743-9':
            lab_value = lab_value / 1000
            lab_unit = '10*3/uL'
            
        if lab_unit == '10*6/uL':
            lab_value = lab_value * 1000
            lab_unit = '10*3/uL'
            
    if lab_name == 'Bicarbonate':
        if lab_unit == 'mm[Hg]':           
#            print(site, lab_name, loinc_code, lab_value, lab_unit)
            lab_value = np.nan
            lab_unit = np.nan
              
    
    return lab_value, lab_unit
        