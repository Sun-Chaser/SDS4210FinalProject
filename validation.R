library(tidyverse)
library(brms)
library(cmdstanr)
library(jsonlite)

options(brms.backend = "cmdstanr")

df <- read_csv("yearly_gross_with_cluster.csv")

df$pred_cluster <- factor(df$pred_cluster)
df$cik          <- factor(df$cik)

df <- df %>% 
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
      TRUE ~ 'International'
    )
  )

df$region <- factor(df$region)

# Ensure ordered by panel id and time
df <- df %>% arrange(cik, year)

# Numeric time variable for ARMA
df$year_numeric <- as.numeric(df$year)

# Train / Val / Test split
train     <- df %>% filter(year <= 2022)
val       <- df %>% filter(year == 2023)
train_val <- df %>% filter(year <= 2023)
test      <- df %>% filter(year == 2024)

metrics <- function(actual, pred) {
  rmse <- sqrt(mean((actual - pred)^2))
  mae  <- mean(abs(actual - pred))
  list(rmse = rmse, mae = mae)
}

p_values <- c(1, 2, 3)
q_values <- c(1, 2, 3)

search_grid  <- expand.grid(p = p_values, q = q_values)
results_list <- list()

for (i in 1:nrow(search_grid)) {
  
  p_val <- search_grid$p[i]
  q_val <- search_grid$q[i]
  
  cat("\n=====================================\n")
  cat("Fitting p =", p_val, " q =", q_val, "\n")
  cat("=====================================\n")
  
  form <- bf(
    yearly_gross ~ pred_cluster +
      arma(time = year_numeric, gr = cik, p = p_val, q = q_val)
  )

  priors <- get_prior(
    formula = form,
    data = train,
    family = student()
    )

   priors <- priors %>%
   mutate(
       prior = case_when(
       class == "Intercept" ~ "normal(9, 0.5)",
       class == "sigma"     ~ "exponential(1)",
       class == "nu"        ~ "gamma(2, 0.1)",
       class %in% c("ar", "ma", "b") ~ "normal(0, 2)",
       TRUE ~ prior
       )
    )
  
  fit_tmp <- brm(
    formula = form,
    data    = train,
    family  = student(),
    prior   = priors,
    chains  = 2, cores = 2,
    iter    = 1500, warmup = 700,
    seed    = 4210,
    silent  = TRUE, refresh = 0
  )
  
  # Use all data up to 2023 so ARMA sees full history
  val_data <- df %>% filter(year <= 2023)
  
  # Actual 2023 outcomes
  val_actual <- val_data %>%
    filter(year == 2023) %>%
    pull(yearly_gross)
  
  # Mask 2023 so brms predicts
  val_data$yearly_gross[val_data$year == 2023] <- NA
  
  val_epred <- posterior_epred(fit_tmp, newdata = val_data, ndraws = 1000)
  val_rows  <- which(is.na(val_data$yearly_gross))
  val_pred_mean <- colMeans(val_epred[, val_rows])
  
  m <- metrics(val_actual, val_pred_mean)
  
  results_list[[i]] <- list(
    p    = p_val,
    q    = q_val,
    rmse = m$rmse,
    mae  = m$mae
  )
}

write_json(results_list, "hyperparam_results_cluster_only.json", pretty = TRUE)
cat("\nSaved hyperparam grid results to hyperparam_results_cluster_only.json\n")

df_results <- bind_rows(results_list)

best <- df_results %>% arrange(rmse) %>% slice(1)

cat("\nBest hyperparameters (cluster-only model):\n")
print(best)

cat("\nUse these p and q values in the evaluation script.\n")