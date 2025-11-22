library(tidyverse)
library(brms) 
library(cmdstanr) 
options(brms.backend = "cmdstanr") 

df = read_csv("yearly_gross_with_cluster.csv") 

# cluster as factor 
df$pred_cluster = factor(df$pred_cluster)
# cik as factor 
df$cik = factor(df$cik) 

cluster_lm = lm(
  yearly_gross ~ pred_cluster, data = df
)
summary(cluster_lm)
# only cluster 5 and 7 are significant relative to baseline

cluster5_and_7 = df %>% filter(pred_cluster %in% c(5,7)) 
# makes sense because cluster 5 is electronics, cluster 7 is software 

# looking at geographical locations 
table(df$loc) 

df = df %>% 
  mutate(
    region = case_when(
      loc %in% c('US-CT','US-ME','US-MA','US-RI','US-VT') ~ 'New England', 
      loc %in% c('US-NJ','US-NY','US-PA') ~ 'Mid-Atlantic', 
      loc %in% c('US-IL','US-IN','US-MI','US-OH','US-WI') ~ 'East North Central', 
      loc %in% c('US-IA','US-KS','US-MN','US-MO','US-NE','US-SD') ~ 'West North Central', 
      loc %in% c('US-DE','US-FL','US-GA','US-MD','US-NC','US-SC','US-VA','US-WV') ~ 'South Atlantic', 
      loc %in% c('US-AL','US-KY','US-MI','US-TN') ~ 'East South Central', 
      loc %in% c('US-AR','US-LA','US-OK','US-TX') ~ 'West South Central', 
      loc %in% c('US-AZ','US-CO','US-ID','US-MT','US-NV','US-UT') ~ 'Mountain', 
      loc %in% c('US-AK','US-CA','US-HI','US-OR','US-WA') ~ 'Pacific', 
      .default = 'International'
    )
  )

df$region = factor(df$region)
table(df$region)

region_lm = lm(
  yearly_gross ~ region, data = df
)
summary(region_lm) 

region_cluster_lm = lm(
  yearly_gross ~ region + pred_cluster, data = df 
) 
summary(region_cluster_lm) 

# build a bayesian time series model 
# we'll train on 2021-23 and predict on 2024 
train = df %>% filter(between(year, 2021, 2023)) 
test = df %>% filter(year == 2024) 

priors = get_prior(
  yearly_gross ~ region + pred_cluster + arma(time = year, gr = cik, p = 1, q = 1), 
  data = df, family = student()
)

priors = priors %>% 
  mutate(
    prior = case_when(
      class == 'Intercept' ~ 'normal(9, 0.5)', 
      class == 'sigma' ~ 'exponential(1)', 
      class == 'nu' ~ 'gamma(2, 0.1)', 
      class %in% c('ar','ma','b') ~ 'normal(0,2)'
    )
  )

fit = brm(
  yearly_gross ~ region + pred_cluster + arma(time = year, gr = cik, p = 1, q = 1), 
  data = df, family = student(), 
  chains = 4, cores = 4, 
  warmup = 1000, iter = 2500, seed = 76
)

# model summary 
# nu = 1 lol 
summary(fit)

# posterior predictive check 
# changing to student's t did wonders 
pp_check(fit) + xlim(-200, 200) 

# how good is the model? 
test_actual = test$yearly_gross 
test$yearly_gross = NA 

pred_data = bind_rows(train, test) 
epreds = posterior_epred(
  fit, newdata = pred_data, ndraws = 2000
)
test_rows = which(is.na(pred_data$yearly_gross)) 
test_preds = colMeans(epreds[, test_rows]) 

mae = mean(abs(test_actual - test_preds))
cat("Test MAE:", mae)  
# MAE is around 3.1 billion - not too bad 

# let's also plot the results 
results = tibble(
  actual = test_actual, 
  preds = test_preds
)

ggplot(results, aes(x=actual, y=preds)) + 
  geom_point() + 
  geom_abline(slope = 1, linetype = "dashed", color = "red") + 
  coord_fixed() + 
  theme_bw() + 
  labs(
    x = "Actual Gross Income", 
    y = "Predicted Gross Income", 
    title = "Model Results on 2024", 
    subtitle = "Reported in Billions of $"
  ) 

# save the model for later 
saveRDS(fit, "test_arma_bayesian_model.rds") 
