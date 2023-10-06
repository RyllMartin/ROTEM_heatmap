# ROTEM_heatmap

Function to generate a plot-heatmap visualizing the significance level, p-value and the mean change in percent comparing 2 different groups or 2 different points in time via ROTEM-Data.

Alpha/the significance level is used to decide whether a test is displayed as a point at all. The color intensity represents the p-value with more intense colors the smaller the p-value. Point size represents the mean change in percent (calculated as (G1-G0)/G0 so as change from G0) with larger points the larger the change in percent.

If unpaired data, then the data of each parameter of each group are analyzed regarding normal distribution via the Shapiro-Wilk test. If the data of both groups follow a normal distribution then the independent t-test (Welch t-test, assuming unequal variance) is used to compare the data and calculate the statistics. If the data from either group is not normally distributed then the Mann-Whitney-U test is used to compare the data and calculate the statistics. The change in percent is calculated as the difference in means (G1.mean() - GO.mean()) divided by G0.mean if both G1 and G0 data are normally distributed and calculated as the difference in medians divided by G0.median if either data is not normally distributed.

If paired data, then the array of the change (difference) of each parameter in each unit (G1 - G0 for each unit) is analyzed regarding normal distribution via Shapiro-Wilk test. If the data follows a normal distribution then the paired-sample t-test is used to compare the data and calculate the statistics. If the data is not normally distributed then the Wilcoxon signed-rank test is used to compare the data and calculate the statistics. The change in percent is calculated as the mean of the differences divided by G0.mean if the differences are normally distributed and calculated as the median of the difference divided by G0.median if the differences are not normally distributed. All statistical tests are from the scipy library.

The code is a 3-step process:

Calculating an initial dataframe with various statistics for all comparisons
Creating the final dataframe from the first dataframe that is passed to the plotter
Creating the visualization from the final dataframe
Dependencies:

import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats as stats

import matplotlib.pyplot as plt
