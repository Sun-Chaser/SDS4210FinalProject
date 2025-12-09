csv_path    <- "/Data/cleaned_data_with_missing.csv"
output_root <- "diagnostics"

library(dplyr)
library(tidyr)
library(stringr)
library(zoo)
library(forecast)  # for na.interp()


dat_wide <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)

time_cols <- grep("^[0-9]{4}Q[1-4]$", names(dat_wide), value = TRUE)

dat_long <- dat_wide %>%
  pivot_longer(
    cols      = all_of(time_cols),
    names_to  = "quarter",
    values_to = "value"
  )

firm_missing_summary <- dat_wide %>%
  mutate(
    na_count = rowSums(is.na(across(all_of(time_cols))))
  ) %>%
  arrange(na_count)

top_n_firms <- 5
top_firms   <- firm_missing_summary[1:top_n_firms, ]

safe_name <- function(x) {
  x %>%
    str_trim() %>%
    str_replace_all("[^A-Za-z0-9]+", "_") %>%
    str_sub(1, 80)  # avoid super-long folder names
}

if (!dir.exists(output_root)) dir.create(output_root, recursive = TRUE)

for (i in seq_len(nrow(top_firms))) {

  firm_row   <- top_firms[i, ]
  cik        <- firm_row$cik
  entity     <- firm_row$entityName
  entity_dir <- file.path(output_root, safe_name(entity))

  if (!dir.exists(entity_dir)) dir.create(entity_dir, recursive = TRUE)

  cat("Processing:", entity, "(CIK:", cik, ")\n")

  y_vec <- as.numeric(firm_row[time_cols])

  start_year   <- as.numeric(substr(time_cols[1], 1, 4))
  start_quarter<- as.numeric(substr(time_cols[1], 6, 6))
  y_ts <- ts(y_vec, frequency = 4, start = c(start_year, start_quarter))

  y_ts_clean <- forecast::na.interp(y_ts)

  # ACF plot (temporal autocorrelation)

  acf_file <- file.path(entity_dir, "acf_plot.png")
  png(acf_file, width = 800, height = 600)
  acf(y_ts_clean,
      main = paste("ACF -", entity),
      xlab = "Lag (quarters)")
  dev.off()

  # Seasonal decomposition plot
  decomp <- stl(y_ts_clean, s.window = "periodic")

  decomp_file <- file.path(entity_dir, "seasonal_decomposition.png")
  png(decomp_file, width = 800, height = 600)
  plot(decomp, main = paste("Seasonal decomposition -", entity))
  dev.off()


  # Rolling mean & rolling SD

  k <- 4
  roll_mean <- zoo::rollmean(y_ts_clean, k = k, align = "right", fill = NA)
  roll_sd   <- zoo::rollapply(y_ts_clean, width = k, FUN = sd,
                              align = "right", fill = NA)

  roll_file <- file.path(entity_dir, "rolling_mean_sd.png")
  png(roll_file, width = 800, height = 800)
  par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))


  plot(y_ts_clean,
       main = paste("Series & Rolling Mean (window =", k, ") -", entity),
       ylab = "Value", xlab = "Time")
  lines(roll_mean, col = "red", lwd = 2)
  legend("topleft",
         legend = c("Series", "Rolling mean"),
         col    = c("black", "red"),
         lty    = c(1, 1),
         bty    = "n")

  plot(roll_sd,
       main = paste("Rolling SD (window =", k, ") -", entity),
       ylab = "SD", xlab = "Time", type = "l")
  par(mfrow = c(1, 1))
  dev.off()
}

cat("Done. Plots saved under:", output_root, "\n")