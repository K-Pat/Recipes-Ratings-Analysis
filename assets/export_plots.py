"""
Script to export all plots from the notebook to HTML files for the website.
Run this script from the notebook directory after all plots have been created.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import ast

# Set up paths
assets_dir = Path(__file__).parent
notebook_dir = assets_dir.parent / 'dsc80-2025-fa' / 'projects' / 'proj04'

# Load data (you may need to adjust this path)
recipes_df = pd.read_csv(notebook_dir / 'RAW_recipes.csv')

# Parse nutrition column
def parse_nutrition(nutrition_str):
    try:
        nutrition_list = ast.literal_eval(nutrition_str)
        return {
            'calories': nutrition_list[0] if len(nutrition_list) > 0 else np.nan,
            'total_fat_pdv': nutrition_list[1] if len(nutrition_list) > 1 else np.nan,
            'sugar_pdv': nutrition_list[2] if len(nutrition_list) > 2 else np.nan,
            'sodium_pdv': nutrition_list[3] if len(nutrition_list) > 3 else np.nan,
            'protein_pdv': nutrition_list[4] if len(nutrition_list) > 4 else np.nan,
            'saturated_fat_pdv': nutrition_list[5] if len(nutrition_list) > 5 else np.nan,
            'carbs_pdv': nutrition_list[6] if len(nutrition_list) > 6 else np.nan
        }
    except:
        return {
            'calories': np.nan, 'total_fat_pdv': np.nan, 'sugar_pdv': np.nan,
            'sodium_pdv': np.nan, 'protein_pdv': np.nan, 'saturated_fat_pdv': np.nan,
            'carbs_pdv': np.nan
        }

nutrition_parsed = recipes_df['nutrition'].apply(parse_nutrition)
nutrition_df = pd.DataFrame(nutrition_parsed.tolist())

recipes_cleaned = recipes_df.copy()
for col in nutrition_df.columns:
    recipes_cleaned[col] = nutrition_df[col]

def parse_tags(tags_str):
    try:
        return ast.literal_eval(tags_str)
    except:
        return []

recipes_cleaned['tags_list'] = recipes_cleaned['tags'].apply(parse_tags)
recipes_cleaned['n_tags'] = recipes_cleaned['tags_list'].apply(len)
recipes_cleaned['ingredients_list'] = recipes_cleaned['ingredients'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)
recipes_cleaned['submitted'] = pd.to_datetime(recipes_cleaned['submitted'], errors='coerce')
recipes_cleaned['calories'] = recipes_cleaned['calories'].apply(
    lambda x: x if (pd.notna(x) and 0 <= x <= 10000) else np.nan
)

# Create primary_category
common_categories = ['desserts', 'main-dish', 'breakfast', 'lunch', 'dinner', 
                     'appetizers', 'side-dishes', 'salads', 'soups', 'beverages']

def get_primary_category(tags_list):
    if not isinstance(tags_list, list):
        return 'other'
    for cat in common_categories:
        if any(cat in tag.lower() for tag in tags_list):
            return cat
    return 'other'

recipes_cleaned['primary_category'] = recipes_cleaned['tags_list'].apply(get_primary_category)

print("Exporting plots to assets folder...")

# Plot 1: Distribution of calories
fig1 = px.histogram(
    recipes_cleaned.dropna(subset=['calories']), 
    x='calories',
    nbins=50,
    title='Distribution of Calories per Recipe',
    labels={'calories': 'Calories', 'count': 'Number of Recipes'},
    color_discrete_sequence=['#2E86AB']
)
fig1.update_layout(
    xaxis_title='Calories',
    yaxis_title='Number of Recipes',
    showlegend=False
)
fig1.write_html(assets_dir / 'calories_distribution.html', include_plotlyjs='cdn')
print("✓ Exported calories_distribution.html")

# Plot 2: Calories vs Number of Ingredients
fig4 = px.scatter(
    recipes_cleaned.dropna(subset=['calories', 'n_ingredients']),
    x='n_ingredients',
    y='calories',
    title='Calories vs Number of Ingredients',
    labels={'n_ingredients': 'Number of Ingredients', 'calories': 'Calories'},
    color_discrete_sequence=['#C73E1D'],
    opacity=0.5
)
fig4.update_layout(
    xaxis_title='Number of Ingredients',
    yaxis_title='Calories'
)
fig4.write_html(assets_dir / 'calories_vs_ingredients.html', include_plotlyjs='cdn')
print("✓ Exported calories_vs_ingredients.html")

# Plot 3: Box plot of calories by category
fig6 = px.box(
    recipes_cleaned.dropna(subset=['calories']),
    x='primary_category',
    y='calories',
    title='Distribution of Calories by Recipe Category',
    labels={'primary_category': 'Recipe Category', 'calories': 'Calories'},
    color_discrete_sequence=['#8B5A3C']
)
fig6.update_layout(
    xaxis_title='Recipe Category',
    yaxis_title='Calories',
    xaxis={'categoryorder': 'total descending'}
)
fig6.update_xaxes(tickangle=45)
fig6.write_html(assets_dir / 'calories_by_category.html', include_plotlyjs='cdn')
print("✓ Exported calories_by_category.html")

# Note: For the missingness and hypothesis test plots, you'll need to run those cells
# in your notebook and export them manually, or add the code here to recreate them.
# The plots require running the permutation tests first.

print("\nNote: You'll need to export the following plots from your notebook:")
print("  - missingness_ingredients.html (from Step 3)")
print("  - hypothesis_dessert.html (from Step 4)")
print("  - hypothesis_ingredients.html (from Step 4)")
print("  - fairness_rmse_by_category.html (from Step 8)")
print("  - fairness_actual_vs_predicted.html (from Step 8)")
print("  - fairness_percent_difference.html (from Step 8)")
print("\nTo export a plot from your notebook, use:")
print("  fig.write_html('assets/plot-name.html', include_plotlyjs='cdn')")

