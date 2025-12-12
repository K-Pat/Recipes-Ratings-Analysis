# Assets Folder

This folder contains the HTML files for all interactive Plotly visualizations used in the website.

## How to Export Plots

After running your notebook and creating all the plots, export them using the following code in your notebook:

```python
# For each plot figure, run:
fig.write_html('assets/plot-name.html', include_plotlyjs='cdn')
```

## Required Plot Files

Make sure you export the following plots from your notebook:

1. **calories_distribution.html** - Distribution of calories (Univariate Analysis)
2. **calories_vs_ingredients.html** - Scatter plot of calories vs ingredients (Bivariate Analysis)
3. **calories_by_category.html** - Box plot of calories by category (Bivariate Analysis)
4. **missingness_ingredients.html** - Permutation test for missingness dependency (Step 3)
5. **hypothesis_dessert.html** - Permutation test for dessert vs non-dessert (Step 4)
6. **hypothesis_ingredients.html** - Permutation test for ingredients (Step 4)
7. **fairness_rmse_by_category.html** - RMSE by category bar chart (Step 8)
8. **fairness_actual_vs_predicted.html** - Actual vs predicted means (Step 8)
9. **fairness_percent_difference.html** - Percent difference bar chart (Step 8)

## Quick Export Script

You can also use the `export_plots.py` script in this folder, but note that it only exports the basic EDA plots. For the permutation test plots and fairness analysis plots, you'll need to export them directly from your notebook after running those cells.

