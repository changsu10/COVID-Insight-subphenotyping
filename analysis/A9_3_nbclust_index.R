library(NbClust)
library(data.table)

# ----- development ----- #
# load data
input_data <- read.csv('W:\\WorkArea-chs4001\\Results\\_development\\dev_scaled.csv')


data_mtx <- data.matrix(subset(input_data, select=-c(ssid)))


# run NbClust
res <- NbClust(data_mtx, diss = NULL, distance = 'euclidean', min.nc = 2, max.nc = 8, 
               method = 'ward.D2', index = 'all')

write.csv(as.data.frame(t(as.matrix(res$All.index))), 
          'W:\\WorkArea-chs4001\\Results\\_output_data\\1_development\\nbclust_index.csv')

write.csv(res$Best.partition, 
          'W:\\WorkArea-chs4001\\Results\\_output_data\\1_development\\nbclust_label.csv')




# ----- validation sensitivity ----- #
# load data
input_data <- read.csv('W:\\WorkArea-chs4001\\Results\\_development_sensitivity\\dev_scaled.csv')


data_mtx <- data.matrix(subset(input_data, select=-c(ssid)))


# run NbClust
res <- NbClust(data_mtx, diss = NULL, distance = 'euclidean', min.nc = 2, max.nc = 8, 
               method = 'ward.D2', index = 'all')

write.csv(as.data.frame(t(as.matrix(res$All.index))), 
          'W:\\WorkArea-chs4001\\Results\\_output_data\\2_development_sensitivity\\1_all_feature_drop_outlier\\nbclust_index.csv')




# ----- validation - 30% validation cohort ----- #
# load data
input_data <- read.csv('W:\\WorkArea-chs4001\\Results\\_validation_test_cohort\\val_scaled.csv')


data_mtx <- data.matrix(subset(input_data, select=-c(ssid)))


# run NbClust
res <- NbClust(data_mtx, diss = NULL, distance = 'euclidean', min.nc = 2, max.nc = 8, 
               method = 'ward.D2', index = 'all')

write.csv(as.data.frame(t(as.matrix(res$All.index))), 
          'W:\\WorkArea-chs4001\\Results\\_output_data\\3_validation_30perc_cohort\\nbclust_index.csv')









# ----- development SDoH ----- #
# load data
input_data <- read.csv('W:\\WorkArea-chs4001\\Results\\_SDoH_clust\\devlopment\\dev_data_SDoH.csv')


data_mtx <- data.matrix(input_data)


# run NbClust
res <- NbClust(data_mtx, diss = NULL, distance = 'euclidean', min.nc = 2, max.nc = 8, 
               method = 'ward.D2', index = 'all')

write.csv(as.data.frame(t(as.matrix(res$All.index))), 
          'W:\\WorkArea-chs4001\\Results\\_output_data\\1_development\\SDoH\\nbclust_index.csv')



