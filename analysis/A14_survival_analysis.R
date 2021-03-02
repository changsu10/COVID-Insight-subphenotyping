library(survival)
library(survminer)

# fit <- survfit(Surv(time, status) ~ sex, data=lung)
# print(fit)
# 
# ggsurvplot(
#   fit,
#   pval = TRUE,
#   #risk.table = TRUE,
#   #risk.table.col = 'strata',
#   #linetype = 'strata',
#   #surv.median.line = 'hv',
#   #ggtheme = theme_bw(),
#   #palette = c('#E7B800', '#2E9FDF')
# )


# load label data
subphenotype_df <- read.csv('W:\\WorkArea-chs4001\\Results\\_development\\_subphenotype_labels.csv',
                            header = TRUE)

# load patient survive data
survive_df <- read.csv('W:\\WorkArea-chs4001\\Data\\survive_.csv', header = TRUE)

# merge data
merged_survive_df <- merge(subphenotype_df, survive_df, by='ssid', all.x = TRUE)




# Mortality --- survival analysis
fit <- survfit(Surv(dtime, death) ~ label, data=merged_survive_df)
print(fit)
p <- ggsurvplot(
  fit,
  pval = TRUE,
  pval.size = 3.5,
  pval.coord = c(50, 0.25),
  
  conf.int = TRUE,
  
  xlab = 'Time (days)',
  
  size = 0.6,
  
  risk.table = TRUE,
  risk.table.pos = 'out',
  risk.table.col = 'black',
  risk.table.y.text.col = FALSE,
  risk.table.height = 0.3,
  risk.table.fontsize = 3,
  risk.table.title.fontsize = 3,

  
  legend.labs = c('I', 'II', 'III', 'IV'),
  legend = 'top', #c(0.15, 0.2),
  legend.fontsize = 8,
  legend.title = 'Subphenotype',
  
  linetype = c(1, 1, 1, 1),
  #surv.median.line = 'hv',
  
  censor = FALSE,
  #censor.shape = '|',
  #censor.size = 3,
  
  ggtheme = theme_bw(),
  
  xlim = c(0, 60),
  break.time.by = 10,
  
  
  
  font.tickslab = c(9, 'plain', 'black'),
  font.x = c(11, 'plain', 'black'),
  font.y = c(11, 'plain', 'black'),
  
  palette = c('#79A3D9', '#7B967A', '#F9C77E', '#CE4257')
)
p$table <- ggpubr::ggpar(p$table, 
                         font.title = list(size=10))
print(p)
ggsave('W:\\WorkArea-chs4001\\Results\\_development\\_mortality_survival.pdf',
       plot = print(p),
       width = 14, height = 14, units = 'cm')





# ICU --- survival analysis
fit <- survfit(Surv(itime, is_icu) ~ label, data=merged_survive_df)
print(fit)
p <- ggsurvplot(
  fit,
  
  fun = 'event',
  
  pval = TRUE,
  pval.size = 3.5,
  pval.coord = c(50, 0.05),
  
  conf.int = TRUE,
  
  xlab = 'Time (days)',
  
  size = 0.6,
  
  risk.table = TRUE,
  risk.table.pos = 'out',
  risk.table.col = 'black',
  risk.table.y.text.col = FALSE,
  risk.table.height = 0.3,
  risk.table.fontsize = 3,
  risk.table.title.fontsize = 3,
  
  
  legend.labs = c('I', 'II', 'III', 'IV'),
  legend = 'top', #c(0.15, 0.2),
  legend.fontsize = 8,
  legend.title = 'Subphenotype',
  
  linetype = c(1, 1, 1, 1),
  #surv.median.line = 'hv',
  
  censor = FALSE,
  #censor.shape = '|',
  #censor.size = 3,
  
  ggtheme = theme_bw(),
  
  xlim = c(0, 60),
  ylim = c(0, 0.5),
  break.time.by = 10,
  
  
  
  font.tickslab = c(9, 'plain', 'black'),
  font.x = c(11, 'plain', 'black'),
  font.y = c(11, 'plain', 'black'),
  
  palette = c('#79A3D9', '#7B967A', '#F9C77E', '#CE4257')
)
p$table <- ggpubr::ggpar(p$table, 
                         font.title = list(size=10))
print(p)
ggsave('W:\\WorkArea-chs4001\\Results\\_development\\_icu_survival.pdf',
       plot = print(p),
       width = 14, height = 14, units = 'cm')






# Ventilation --- survival analysis
fit <- survfit(Surv(vtime, is_vent) ~ label, data=merged_survive_df)
p <- ggsurvplot(
  fit,
  
  fun = 'event',
  
  pval = TRUE,
  pval.size = 3.5,
  pval.coord = c(50, 0.05),
  
  conf.int = TRUE,
  xlab = 'Time (days)',
  
  size = 0.6,
  
  risk.table = TRUE,
  risk.table.pos = 'out',
  risk.table.col = 'black',
  risk.table.y.text.col = FALSE,
  risk.table.height = 0.3,
  risk.table.fontsize = 3,
  risk.table.title.fontsize = 3,
  
  
  legend.labs = c('I', 'II', 'III', 'IV'),
  legend = 'top', #c(0.15, 0.2),
  legend.fontsize = 8,
  legend.title = 'Subphenotype',
  
  linetype = c(1, 1, 1, 1),
  #surv.median.line = 'hv',
  
  censor = FALSE,
  #censor.shape = '|',
  #censor.size = 3,
  
  ggtheme = theme_bw(),
  
  xlim = c(0, 60),
  ylim = c(0, 0.5),
  break.time.by = 10,
  
  
  
  font.tickslab = c(9, 'plain', 'black'),
  font.x = c(11, 'plain', 'black'),
  font.y = c(11, 'plain', 'black'),
  
  palette = c('#79A3D9', '#7B967A', '#F9C77E', '#CE4257')
)
p$table <- ggpubr::ggpar(p$table, 
                         font.title = list(size=10))
print(p)
ggsave('W:\\WorkArea-chs4001\\Results\\_development\\_vent_survival.pdf',
       plot = print(p),
       width = 14, height = 14, units = 'cm')








