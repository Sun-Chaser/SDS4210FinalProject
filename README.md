# SDS4210FinalProject
This is the final project for SDS 4210 Statistical Computation course at Washington University in Saint Louis

This project uses a geospatial, industry-aware, time-series income model for US publicly traded companies using SEC filings, filled missing values using EM, and predicted future income using Bayesian ARMA models.

##Code Reproduction Guide (Jack Yang, Brian Wei,)##
#Dataset clean and Imputation#
DataAggregation.Rmd
-For grab, combine, clean, and create data frames shown in data frame preview
DataCleanKalman.Rmd
-For Kalman Filter data imputation and performance metrics calculation
DataCleanGaussian.Rmd
-For Finite Gaussian Mixture model data imputation and performance metrics calculation

#Dataset visualization, Model Tuning, and Evaluation#
eda.ipynb
-For dataset visualizations as seen in Figure 2-6
diagnostics.R
-For ACF plots in Figure 8
naive_comparisons.R
-For Table 5, the experiment comparing ARMA to naive model baselines using LOOIC
predictors_experiment.R
-For Table 6, the experiment comparing different combinations of predictors under ARMA(1, 1)
validation.R
-Grid search for hyperparameters p and q
evaluate.R
-Evaluation of ARMA model on test set
