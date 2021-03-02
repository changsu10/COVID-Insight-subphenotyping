library(circlize)


MAIN_PATH = 'W:\\WorkArea-chs4001\\Results\\_development\\'




# age
file_path = paste(MAIN_PATH, '_chord_phenotype_age_proportion.csv', sep='')
age_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I', 
           'Age 18-40', 'Age 41-60', 'Age 61-80', 'Age>80')


grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9', #00567F, 54D2D2, FF6666
             'Age 18-40'='#DCDCDC', 'Age 41-60'='#DCDCDC', 'Age 61-80'='#DCDCDC', 'Age>80'='#DCDCDC')
chordDiagram(age_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))



# gender
file_path = paste(MAIN_PATH, '_chord_phenotype_sex_proportion.csv', sep='')
sex_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I', 
           'Female', 'Male')

grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9', #00567F, 54D2D2, FF6666
             'Female'='#DCDCDC', 'Male'='#DCDCDC')
chordDiagram(sex_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))


# race
file_path = paste(MAIN_PATH, '_chord_phenotype_race_proportion.csv', sep='')
race_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I', 
           'White', 'Black', 'Asian', 'Other/unknown')
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9', #00567F, 54D2D2, FF6666
             'White'='#DCDCDC', 'Black'='#DCDCDC', 'Asian'='#DCDCDC', 'Other/unknown'='#DCDCDC')
chordDiagram(race_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))



# comorbidity
file_path = paste(MAIN_PATH, '_chord_phenotype_comorbidity.csv', sep='')
comorbidity_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'HTN', 'Diabetes','CAD', 'HF',
           'COPD', 'ATA', 'Cancer', 'HLD', 'Obesity')
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9', #00567F, 54D2D2, FF6666
             'HTN'='#DCDCDC', 'Diabetes'='#DCDCDC', 'CAD'='#DCDCDC', 'HF'='#DCDCDC',
             'COPD'='#DCDCDC', 'ATA'='#DCDCDC', 'Cancer'='#DCDCDC', 'HLD'='#DCDCDC',
             'Obesity'='#DCDCDC')
chordDiagram(comorbidity_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))




########### comorbidity proportion ##############
file_path = paste(MAIN_PATH, '_chord_phenotype_comorbidity_proportion.csv', sep='')
comorbidity_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'HTN', 'Diabetes','CAD', 'HF',
           'COPD', 'ATA', 'Cancer', 'HLD', 'Obesity')
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9', #00567F, 54D2D2, FF6666
             'HTN'='#DCDCDC', 'Diabetes'='#DCDCDC', 'CAD'='#DCDCDC', 'HF'='#DCDCDC',
             'COPD'='#DCDCDC', 'ATA'='#DCDCDC', 'Cancer'='#DCDCDC', 'HLD'='#DCDCDC',
             'Obesity'='#DCDCDC')
chordDiagram(comorbidity_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))

# I
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'HTN', 'Diabetes','CAD', 'HF',
           'COPD', 'ATA', 'Cancer', 'HLD', 'Obesity')
grid.col = c('Subphenotype IV'='#DCDCDC', 'Subphenotype III'='#DCDCDC', 
             'Subphenotype II'='#DCDCDC', 'Subphenotype I'='#79A3D9', #00567F, 54D2D2, FF6666
             'HTN'='#DCDCDC', 'Diabetes'='#DCDCDC', 'CAD'='#DCDCDC', 'HF'='#DCDCDC',
             'COPD'='#DCDCDC', 'ATA'='#DCDCDC', 'Cancer'='#DCDCDC', 'HLD'='#DCDCDC',
             'Obesity'='#DCDCDC')
chordDiagram(comorbidity_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))
# II
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'HTN', 'Diabetes','CAD', 'HF',
           'COPD', 'ATA', 'Cancer', 'HLD', 'Obesity')
grid.col = c('Subphenotype IV'='#DCDCDC', 'Subphenotype III'='#DCDCDC', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#DCDCDC', #00567F, 54D2D2, FF6666
             'HTN'='#DCDCDC', 'Diabetes'='#DCDCDC', 'CAD'='#DCDCDC', 'HF'='#DCDCDC',
             'COPD'='#DCDCDC', 'ATA'='#DCDCDC', 'Cancer'='#DCDCDC', 'HLD'='#DCDCDC',
             'Obesity'='#DCDCDC')
chordDiagram(comorbidity_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))
# III
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'HTN', 'Diabetes','CAD', 'HF',
           'COPD', 'ATA', 'Cancer', 'HLD', 'Obesity')
grid.col = c('Subphenotype IV'='#DCDCDC', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#DCDCDC', 'Subphenotype I'='#DCDCDC', #00567F, 54D2D2, FF6666
             'HTN'='#DCDCDC', 'Diabetes'='#DCDCDC', 'CAD'='#DCDCDC', 'HF'='#DCDCDC',
             'COPD'='#DCDCDC', 'ATA'='#DCDCDC', 'Cancer'='#DCDCDC', 'HLD'='#DCDCDC',
             'Obesity'='#DCDCDC')
chordDiagram(comorbidity_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))
# IV
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'HTN', 'Diabetes','CAD', 'HF',
           'COPD', 'ATA', 'Cancer', 'HLD', 'Obesity')
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#DCDCDC', 
             'Subphenotype II'='#DCDCDC', 'Subphenotype I'='#DCDCDC', #00567F, 54D2D2, FF6666
             'HTN'='#DCDCDC', 'Diabetes'='#DCDCDC', 'CAD'='#DCDCDC', 'HF'='#DCDCDC',
             'COPD'='#DCDCDC', 'ATA'='#DCDCDC', 'Cancer'='#DCDCDC', 'HLD'='#DCDCDC',
             'Obesity'='#DCDCDC')
chordDiagram(comorbidity_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))














# lab proportion
file_path = paste(MAIN_PATH, '_chord_phenotype_lab_proportion.csv', sep='')
lab_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'Inflammation', 'Hepatic','Cardiovascular', 'Renal', 'Hematologic')
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9',
             'Inflammation'='#DCDCDC', 'Hepatic'='#DCDCDC', 'Cardiovascular'='#DCDCDC', 
             'Renal'='#DCDCDC', 'Hematologic'='#DCDCDC')
chordDiagram(lab_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))


# lab group level
file_path = paste(MAIN_PATH, '_chord_phenotype_lab_group_level.csv', sep='')
lab_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'Inflammation', 'Hepatic','Cardiovascular', 'Renal', 'Hematologic')
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9',
             'Inflammation'='#DCDCDC', 'Hepatic'='#DCDCDC', 'Cardiovascular'='#DCDCDC', 
             'Renal'='#DCDCDC', 'Hematologic'='#DCDCDC')
p <- chordDiagram(lab_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))

# I
grid.col = c('Subphenotype IV'='#DCDCDC', 'Subphenotype III'='#DCDCDC', 
             'Subphenotype II'='#DCDCDC', 'Subphenotype I'='#79A3D9',
             'Inflammation'='#DCDCDC', 'Hepatic'='#DCDCDC', 'Cardiovascular'='#DCDCDC', 
             'Renal'='#DCDCDC', 'Hematologic'='#DCDCDC')
p <- chordDiagram(lab_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))
# II
grid.col = c('Subphenotype IV'='#DCDCDC', 'Subphenotype III'='#DCDCDC', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#DCDCDC',
             'Inflammation'='#DCDCDC', 'Hepatic'='#DCDCDC', 'Cardiovascular'='#DCDCDC', 
             'Renal'='#DCDCDC', 'Hematologic'='#DCDCDC')
p <- chordDiagram(lab_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))
# III
grid.col = c('Subphenotype IV'='#DCDCDC', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#DCDCDC', 'Subphenotype I'='#DCDCDC',
             'Inflammation'='#DCDCDC', 'Hepatic'='#DCDCDC', 'Cardiovascular'='#DCDCDC', 
             'Renal'='#DCDCDC', 'Hematologic'='#DCDCDC')
p <- chordDiagram(lab_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))
# IV
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#DCDCDC', 
             'Subphenotype II'='#DCDCDC', 'Subphenotype I'='#DCDCDC',
             'Inflammation'='#DCDCDC', 'Hepatic'='#DCDCDC', 'Cardiovascular'='#DCDCDC', 
             'Renal'='#DCDCDC', 'Hematologic'='#DCDCDC')
p <- chordDiagram(lab_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))




# SDoH proportion
file_path = paste(MAIN_PATH, '_chord_phenotype_SDoH_proportion.csv', sep='')
lab_df <- read.csv(file_path)
orders = c('Subphenotype IV', 'Subphenotype III', 'Subphenotype II', 'Subphenotype I',
           'SDoH Level 1', 'SDoH Level 2','SDoH Level 3', 'Unknown/missing')
grid.col = c('Subphenotype IV'='#CE4257', 'Subphenotype III'='#F9C77E', 
             'Subphenotype II'='#7B967A', 'Subphenotype I'='#79A3D9',
             'SDoH Level 1'='#DCDCDC', 'SDoH Level 2'='#DCDCDC', 'SDoH Level 3'='#DCDCDC', 
             'Unknown/missing'='#DCDCDC')
chordDiagram(lab_df, order = orders, grid.col=grid.col, scale=FALSE, annotationTrack = c('name', 'grid'))














