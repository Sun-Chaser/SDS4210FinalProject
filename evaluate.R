# evaluate_cluster_only.R
# ============================================================
# Libraries
# ============================================================
library(tidyverse)
library(brms)
library(cmdstanr)
library(jsonlite)

options(brms.backend = "cmdstanr")

# ============================================================
# Load and prepare data
# ============================================================
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

# ============================================================
# Metric function
# ============================================================
metrics <- function(actual, pred) {
  rmse <- sqrt(mean((actual - pred)^2))
  mae  <- mean(abs(actual - pred))
  list(rmse = rmse, mae = mae)
}

# ============================================================
# Plug in best hyperparameters from hyperparam_search_cluster_only.R
#   >>> REPLACE THESE WITH YOUR SELECTED p, q <<<
# ============================================================
best_p <- 1  # <-- change this
best_q <- 1  # <-- change this

cat("\nUsing best hyperparameters: p =", best_p, " q =", best_q, "\n")

# ============================================================
# Refit final model on TRAIN + VAL (<= 2023)
# ============================================================
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

# ============================================================
# Evaluate on TEST (2024) using full history (<= 2024)
# ============================================================
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

# ============================================================
# MCMC Diagnostics (Traceplots, Rhat, ESS)
# ============================================================

plot(final_fit)

summary(final_fit)


# Predicted vs Actual Plot

library(ggplot2)

pred_df <- data.frame(
  actual = test_actual,
  predicted = test_pred_mean
)

ggplot(pred_df, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Predicted vs Actual Gross Income (2024)",
       x = "Actual (billions of $)",
       y = "Predicted (billions of $)") +
  theme_minimal()

# ============================================================
# Residual Diagnostics (check if temporal autocorrelation was handled)
# ============================================================

fitted_vals <- fitted(final_fit, summary = TRUE)[,"Estimate"]
train_data <- train_val %>% drop_na()

residuals <- train_data$yearly_gross - fitted_vals[1:nrow(train_data)]

# ACF plot
acf(residuals, main="Residual Autocorrelation")

# Residual vs Time plot

res_df <- train_data %>% mutate(resid = residuals)

ggplot(res_df, aes(x = year, y = resid)) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals vs Time") +
  theme_minimal()


#ljung-Box test
Box.test(residuals, lag=1, type="Ljung-Box")


# ============================================================
# Predict 2025 
# ============================================================

# generate 2025 data
future <- df %>% filter(year == 2024) %>%
  mutate(year = 2025, year_numeric = year_numeric + 1, yearly_gross = NA)

forecast_epred <- posterior_epred(final_fit, newdata = bind_rows(df, future)) 
forecast_mean <- colMeans(forecast_epred[, 1665:2080])


# aggregate by industry
future$predicted_2025 <- forecast_mean

future %>%
  group_by(pred_cluster) %>%
  summarise(mean_income = mean(predicted_2025)) %>% 
  ggplot(aes(x=reorder(pred_cluster, -mean_income), y=mean_income)) + 
  geom_col() + 
  labs(
    x = "Labeled Cluster", 
    y = "Predicted 2025 Gross Income", 
    title = "Our Predictions for 2025"
  ) + 
  theme_bw() 


# END OF FILE