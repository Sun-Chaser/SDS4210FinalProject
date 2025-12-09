library(tidyverse)
library(brms)
library(cmdstanr)

options(brms.backend = "cmdstanr")

df <- read_csv("yearly_gross_with_cluster.csv")

df$pred_cluster <- factor(df$pred_cluster)
df$cik          <- factor(df$cik)

df <- df %>%
  mutate(
    region = case_when(
      loc %in% c('US-CT','US-ME','US-MA','US-RI','US-VT') ~ 'New England',
      loc %in% c('US-NJ','US-NY','US-PA')                 ~ 'Mid-Atlantic',
      loc %in% c('US-IL','US-IN','US-MI','US-OH','US-WI') ~ 'East North Central',
      loc %in% c('US-IA','US-KS','US-MN','US-MO','US-NE','US-SD') ~ 'West North Central',
      loc %in% c('US-DE','US-FL','US-GA','US-MD','US-NC','US-SC','US-VA','US-WV') ~ 'South Atlantic',
      loc %in% c('US-AL','US-KY','US-MI','US-TN')         ~ 'East South Central',
      loc %in% c('US-AR','US-LA','US-OK','US-TX')         ~ 'West South Central',
      loc %in% c('US-AZ','US-CO','US-ID','US-MT','US-NV','US-UT') ~ 'Mountain',
      loc %in% c('US-AK','US-CA','US-HI','US-OR','US-WA') ~ 'Pacific',
      TRUE                                                ~ 'International'
    )
  )

df$region <- factor(df$region)
df$year_numeric <- as.numeric(df$year)

# Train for this experiment: 2008–2023
train_mean <- df %>% filter(year >= 2008, year <= 2023)

# Keep 2024 aside
test_2024 <- df %>% filter(year == 2024)

cat("Training rows (2008–2023):", nrow(train_mean), "\n")
cat("Held-out 2024 rows (not used here):", nrow(test_2024), "\n")

# Fixed ARMA(1,1) structure
p_fix <- 1
q_fix <- 1

# helper
fit_and_report <- function(formula_bf, data, model_name,
                           chains = 4, cores = 4, iter = 2000, warmup = 1000,
                           seed = 4210,
                           save_rds = FALSE, rds_path = NULL) {
  cat("\n=====================================\n")
  cat("Fitting model:", model_name, "\n")
  cat("Formula:", deparse(formula_bf$formula), "\n")
  cat("=====================================\n")

  fit <- brm(
    formula   = formula_bf,
    data      = data,
    family    = student(),
    chains    = chains,
    cores     = cores,
    iter      = iter,
    warmup    = warmup,
    seed      = seed,
    silent    = TRUE,
    refresh   = 0,
    save_pars = save_pars(all = TRUE)   # keep everything for LOO
  )

  # Add LOO criterion
  fit <- add_criterion(fit, "loo")

  loo_obj  <- fit$criteria$loo
  looic    <- loo_obj$estimates["looic", "Estimate"]
  looic_se <- loo_obj$estimates["looic", "SE"]

  cat("\n--- Model:", model_name, "---\n")
  cat("LOOIC:", looic, "\n")
  cat("SE(LOOIC):", looic_se, "\n")

  if (!isFALSE(save_rds) && !is.null(rds_path)) {
    saveRDS(fit, rds_path)
    cat("Saved fit to:", rds_path, "\n")
  }

  invisible(fit)
}

# Model 1: Intercept only + ARMA(1,1)
form_intercept <- bf(
  yearly_gross ~ 1 +
    arma(time = year, gr = cik, p = p_fix, q = q_fix)
)

fit_and_report(
  formula_bf = form_intercept,
  data       = train_mean,
  model_name = "Intercept_only",
  save_rds   = TRUE,
  rds_path   = "fit_mean_arma11_intercept_only.rds"
)

# Model 2: Intercept + Cluster + ARMA(1,1)
form_cluster <- bf(
  yearly_gross ~ pred_cluster +
    arma(time = year, gr = cik, p = p_fix, q = q_fix)
)

fit_and_report(
  formula_bf = form_cluster,
  data       = train_mean,
  model_name = "Intercept_plus_cluster",
  save_rds   = TRUE,
  rds_path   = "fit_mean_arma11_intercept_cluster.rds"
)

# Model 3: Intercept + Region + ARMA(1,1) 
form_region <- bf(
  yearly_gross ~ region +
    arma(time = year, gr = cik, p = p_fix, q = q_fix)
)

fit_and_report(
  formula_bf = form_region,
  data       = train_mean,
  model_name = "Intercept_plus_region",
  save_rds   = TRUE,
  rds_path   = "fit_mean_arma11_intercept_region.rds"
)

# Model 4: Intercept + Region + Cluster + ARMA(1,1)
form_region_cluster <- bf(
  yearly_gross ~ region + pred_cluster +
    arma(time = year, gr = cik, p = p_fix, q = q_fix)
)

fit_and_report(
  formula_bf = form_region_cluster,
  data       = train_mean,
  model_name = "Intercept_plus_region_cluster",
  save_rds   = TRUE,
  rds_path   = "fit_mean_arma11_intercept_region_cluster.rds"
)

cat("\nAll models fit. Check LOOIC values above in the console output.\n")
