
# load subphenotype label
df  <- read.csv("W:\\WorkArea-chs4001\\Results\\_development\\_subphenotype_labels.csv")

# load SDoH label
SDoH_clust <- read.csv("W:\\WorkArea-chs4001\\Results\\_SDoH_clust\\devlopment\\_SDoH_label.csv")

# load outcome data
outcome_df <- read.csv("W:\\WorkArea-chs4001\\Data\\outcome_comorbidity_.csv")

# merge
data_df <- merge(x=SDoH_clust, y=df, by='ssid', all.x = TRUE)
data_df <- merge(x=data_df, y=outcome_df, by='ssid', all.x = TRUE)


# logistic regression analysis
data_df$SDoH_clust <- factor(data_df$SDoH_clust, levels = c(1, 2, 3))

data_df_1 = subset(data_df, label==1)
data_df_2 = subset(data_df, label==2)
data_df_3 = subset(data_df, label==3)
data_df_4 = subset(data_df, label==4)

logit_1 = glm(death_within_60d ~ SDoH_clust + age_at_confirm + sex, family = "binomial", data=data_df_1)
summary(logit_1)

logit_2 = glm(death_within_60d ~ SDoH_clust + age_at_confirm + sex, family = "binomial", data=data_df_2)
summary(logit_2)

logit_3 = glm(death_within_60d ~ SDoH_clust + age_at_confirm + sex, family = "binomial", data=data_df_3)
summary(logit_3)

logit_4 = glm(death_within_60d ~ SDoH_clust + age_at_confirm + sex, family = "binomial", data=data_df_4)
summary(logit_4)