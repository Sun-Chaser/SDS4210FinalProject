
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

# Train / Val / Test splits
train     <- df %>% filter(year <= 2022)
val       <- df %>% filter(year == 2023)
train_val <- df %>% filter(year <= 2023)
test      <- df %>% filter(year == 2024)

metrics <- function(actual, pred) {
  rmse <- sqrt(mean((actual - pred)^2))
  mae  <- mean(abs(actual - pred))
  list(rmse = rmse, mae = mae)
}

# best config (change based on validation.R)
best_p <- 1
best_q <- 1

cat("\nUsing best hyperparameters: p =", best_p, " q =", best_q, "\n")

final_form <- bf(
  yearly_gross ~ pred_cluster +
    arma(time = year_numeric, gr = cik, p = best_p, q = best_q)
)

priors <- get_prior(
  formula = final_form,
  data = train_val,
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

final_fit <- brm(
  formula = final_form,
  data    = train_val,
  family  = student(),
  prior   = priors,
  chains  = 4, cores = 4,
  iter    = 2500, warmup = 1000,
  seed    = 4210
)

saveRDS(final_fit, "final_panel_arma_cluster_only_model.rds")
cat("\nSaved final model as final_panel_arma_cluster_only_model.rds\n")

# Evaluate on TEST (2024) using full history (<= 2024)
test_data <- df %>% filter(year <= 2024)

# Actual 2024 values
test_actual <- test_data %>%
  filter(year == 2024) %>%
  pull(yearly_gross)

# Mask 2024
test_data$yearly_gross[test_data$year == 2024] <- NA

test_epred <- posterior_epred(final_fit, newdata = test_data, ndraws = 2000)
test_rows  <- which(is.na(test_data$yearly_gross))

test_pred_mean <- colMeans(test_epred[, test_rows])

test_metrics <- metrics(test_actual, test_pred_mean)

final_results <- list(
  best_hyperparams = list(p = best_p, q = best_q),
  test_rmse        = test_metrics$rmse,
  test_mae         = test_metrics$mae
)

write_json(final_results, "final_test_results_cluster_only.json", pretty = TRUE)

cat("\nFinal Test Metrics (cluster-only model):\n")
print(final_results)