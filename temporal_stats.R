library(tidyverse)

df <- read_csv("yearly_gross_with_cluster.csv")

# Ensure types
df <- df %>%
  mutate(
    cik = factor(cik),
    pred_cluster = factor(pred_cluster),
    year = as.numeric(year)
  )

df_yoy <- df %>%
  arrange(cik, year) %>%
  group_by(cik) %>%
  mutate(
    yoy_change      = yearly_gross - lag(yearly_gross),           # raw difference
    yoy_abs_change  = abs(yoy_change),                            # absolute difference
    yoy_pct_change  = (yoy_change / lag(yearly_gross)) * 100      # percent change
  ) %>%
  ungroup()

overall_income_stats <- df %>%
  summarise(
    n_obs   = n(),
    min     = min(yearly_gross, na.rm = TRUE),
    q1      = quantile(yearly_gross, 0.25, na.rm = TRUE),
    median  = median(yearly_gross, na.rm = TRUE),
    mean    = mean(yearly_gross, na.rm = TRUE),
    q3      = quantile(yearly_gross, 0.75, na.rm = TRUE),
    max     = max(yearly_gross, na.rm = TRUE),
    sd      = sd(yearly_gross, na.rm = TRUE)
  )

cat("\n================ Overall Yearly Gross Stats ================\n")
print(overall_income_stats)

yoy_stats <- df_yoy %>%
  summarise(
    n_yoy              = sum(!is.na(yoy_change)),
    mean_abs_change    = mean(yoy_abs_change, na.rm = TRUE),
    median_abs_change  = median(yoy_abs_change, na.rm = TRUE),
    p90_abs_change     = quantile(yoy_abs_change, 0.90, na.rm = TRUE),
    max_abs_change     = max(yoy_abs_change, na.rm = TRUE),

    mean_pct_change    = mean(yoy_pct_change, na.rm = TRUE),
    median_pct_change  = median(yoy_pct_change, na.rm = TRUE),
    p90_pct_change     = quantile(yoy_pct_change, 0.90, na.rm = TRUE),
    max_pct_change     = max(yoy_pct_change, na.rm = TRUE)
  )

cat("\n================ YoY Change Stats (Absolute & Percent) ================\n")
print(yoy_stats)

firm_volatility <- df %>%
  group_by(cik) %>%
  summarise(
    n_years          = n(),
    mean_income      = mean(yearly_gross, na.rm = TRUE),
    sd_income        = sd(yearly_gross, na.rm = TRUE),
    min_income       = min(yearly_gross, na.rm = TRUE),
    max_income       = max(yearly_gross, na.rm = TRUE)
  ) %>%
  ungroup()

cat("\n================ Per-Firm Income Volatility (SD over Years) ================\n")
print(head(firm_volatility, 10))  # show first 10 firms as a preview

write_csv(overall_income_stats, "overall_income_stats.csv")
write_csv(yoy_stats,           "yoy_stats_overall.csv")
write_csv(firm_volatility,     "firm_volatility_by_cik.csv")

cat("\nSaved:\n",
    "- overall_income_stats.csv\n",
    "- yoy_stats_overall.csv\n",
    "- firm_volatility_by_cik.csv\n")
    
p_yoy_abs <- ggplot(df_yoy, aes(x = yoy_abs_change)) +
  geom_histogram(bins = 50) +
  scale_x_continuous(labels = scales::comma) +
  theme_bw() +
  labs(
    title = "Distribution of Year-over-Year Absolute Income Changes",
    x     = "Absolute YoY Change (same units as yearly_gross)",
    y     = "Count"
  )

ggsave("yoy_abs_change_histogram.png", p_yoy_abs, width = 8, height = 5, dpi = 300)

cat("\nSaved histogram: yoy_abs_change_histogram.png\n")