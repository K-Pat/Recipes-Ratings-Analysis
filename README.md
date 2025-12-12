# Predicting Recipe Calories: A Machine Learning Approach

### Name: Kavyan Patel

# Overview
This data science project, conducted at UCSD, focuses on predicting the calorie content of recipes using machine learning techniques. By analyzing recipe characteristics such as ingredients, cooking time, and recipe categories, we build predictive models to estimate calorie counts and evaluate their fairness across different recipe types.

# Table of Contents
- [Introduction](#introduction)
- [Data Cleaning and Exploratory Data Analysis](#datacleaning)
- [Assessment of Missingness](#nmaranalysis)
- [Hypothesis Testing](#hypothesistesting)
- [Framing a Prediction Problem](#framingpredictionproblem)
- [Baseline Model](#baselinemodel)
- [Final Model](#finalmodel)
- [Fairness Analysis](#fairnessanalysis)

<!-- #region -->
# Introduction <a name="Introduction"></a>

This data science project investigates what types of recipes tend to have the most calories and builds a predictive model to estimate calorie content. With growing health consciousness and the need for nutritional information, understanding the factors that influence recipe calories is valuable for both consumers and food platforms. Our goal is to explore the relationship between recipe characteristics (such as ingredients, cooking time, and recipe categories) and calorie content, then develop a machine learning model that can accurately predict calories for new recipes.

The dataset, sourced from food.com, includes over 83,000 recipes collected since 2008. It provides detailed information on nutrition, ingredients, cooking instructions, and recipe categories. By analyzing this data, we can identify patterns in calorie content and build predictive models that help users estimate the nutritional value of recipes before preparing them.

### This is a description of what the recipes dataframe contains (83,782 rows):

|    Column  |  Description |
|-----------:|------------:|
|    'name' |       Recipe name |
|     'id' |       Recipe ID |
|      'minutes' |       Minutes to prepare recipe |
|     'submitted' |      Date recipe was submitted |
| 'tags' |      Food.com tags for recipe (stored as string representation of list) |
|     'nutrition' |      Nutrition information in this form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for "percentage of daily value" |
|     'n_steps' |      Number of steps in the recipe |
|    'steps' |      Step by step instructions to follow |
|     'description' |      A description of what the recipe makes |
|     'ingredients' |      List of ingredients (stored as string representation of list) |
|     'n_ingredients' |      Number of ingredients |

The relevant columns for answering our question are: `nutrition` (contains calorie information), `tags` (contains recipe categories/types), `ingredients` (list of ingredients), `n_ingredients` (number of ingredients), and `minutes` (cooking time).

<!-- #endregion -->
<!-- #region -->

# Data Cleaning and Exploratory Data Analysis <a name="datacleaning"></a>

## Data Cleaning

The data cleaning process involved several key steps to prepare the dataset for analysis:

1. **Parsing the nutrition column**: The `nutrition` column was stored as a string representation of a list. We used `ast.literal_eval()` to parse it into individual components, extracting calories and other nutritional values (total fat, sugar, sodium, protein, saturated fat, and carbohydrates) as separate columns.

2. **Parsing the tags column**: Similar to nutrition, the `tags` column was stored as a string representation of a list. We parsed it to create a `tags_list` column containing actual Python lists, and created a `n_tags` column counting the number of tags per recipe.

3. **Parsing the ingredients column**: The `ingredients` column was also stored as a string representation of a list. We parsed it to create an `ingredients_list` column for easier analysis.

4. **Converting date column**: The `submitted` column was converted from string format to datetime using `pd.to_datetime()` with error handling.

5. **Cleaning calorie values**: We replaced any negative or unrealistic calorie values (greater than 10,000 calories per serving) with NaN, as these likely represent data entry errors or parsing failures.

After cleaning, the dataset contained 83,782 rows with 22 columns. Only 39 recipes (0.05%) had missing calorie values, which were handled appropriately in subsequent analyses.

### Head of cleaned DataFrame:

| name                                 |   calories |   minutes |   n_ingredients |   n_tags |
|:-------------------------------------|-----------:|----------:|----------------:|---------:|
| 1 brownies in the world    best ever |      138.4 |        40 |               9 |       11 |
| 1 in canada chocolate chip cookies   |      595.1 |        45 |              11 |       12 |
| 412 broccoli casserole               |      194.8 |        40 |               9 |       12 |
| millionaire pound cake               |      878.3 |       120 |               7 |       13 |
| 2000 meatloaf                        |      267.0 |        90 |              13 |       15 |

## Univariate Analysis

We examined the distributions of key variables to understand the data structure. The distribution of calories shows that most recipes have moderate calorie counts, with the majority falling below 500 calories. The distribution is right-skewed, with some recipes having very high calorie counts (up to nearly 10,000 calories per serving).

<iframe src="assets/calories_distribution.html" width=800 height=600 frameBorder=0></iframe>

The histogram above shows the distribution of calories per recipe. Most recipes cluster in the 0-500 calorie range, with fewer recipes having extremely high calorie counts. This right-skewed distribution is expected, as most everyday recipes tend to be moderate in calories, while some special occasion or complex recipes can be much higher.

## Bivariate Analysis

We explored relationships between calories and other variables to identify potential predictors. The scatter plot below shows the relationship between calories and the number of ingredients.

<iframe src="assets/calories_vs_ingredients.html" width=800 height=600 frameBorder=0></iframe>

The scatter plot reveals a weak positive relationship between the number of ingredients and calories. While recipes with more ingredients tend to have slightly higher calories on average, there is substantial variability, indicating that the number of ingredients alone is not a strong predictor of calorie content.

We also examined calories by recipe category using a box plot, which revealed significant differences in calorie distributions across categories. Main dishes and desserts tend to have higher median calories compared to salads, soups, and beverages.

<iframe src="assets/calories_by_category.html" width=800 height=600 frameBorder=0></iframe>

The box plot shows that main-dish recipes have the highest median calories, followed by desserts. Salads, soups, and beverages have the lowest median calories. This suggests that recipe category is an important factor in predicting calorie content.

## Interesting Aggregates

We aggregated the data to identify patterns in calorie content across different groupings:

### Average Calories by Recipe Category:

| primary_category   |   Mean Calories |   Median Calories |   Std Calories |   Count |
|:-------------------|----------------:|------------------:|---------------:|--------:|
| main-dish          |          501.52 |            422.1  |         391.57 |   25190 |
| desserts           |          492.67 |            286.6  |         819.54 |   13481 |
| other              |          455.84 |            274.9  |         630.27 |   12862 |
| lunch              |          394.31 |            299.8  |         434.38 |    4292 |
| breakfast          |          361.2  |            281.6  |         360.24 |    4612 |
| appetizers         |          335.66 |            202.5  |         472.19 |    3403 |
| soups              |          332.88 |            279.45 |         279.98 |    3054 |
| salads             |          310.85 |            238.3  |         327.6  |    2849 |
| dinner             |          310.59 |            209    |         401.37 |    4245 |
| side-dishes        |          280    |            227.3  |         258.79 |    6223 |
| beverages          |          231.29 |            154.75 |         355.85 |    3532 |

This table shows that main-dish recipes have the highest average calories (501.52), while beverages have the lowest (231.29). Interestingly, desserts have a very high standard deviation (819.54), indicating wide variability in calorie content within this category.

### Average Calories by Number of Ingredients:

| ingredients_bin   |   Mean Calories |   Median Calories |   Count |
|:------------------|----------------:|------------------:|--------:|
| 1-5               |          328.03 |            200.2  |   14159 |
| 6-10              |          398.66 |            288.1  |   41614 |
| 11-15             |          486.46 |            368.4  |   22794 |
| 16-20             |          582.39 |            455.35 |    4496 |
| 21+               |          702.02 |            543.8  |     680 |

There is a clear positive relationship between the number of ingredients and average calories. Recipes with 21+ ingredients have more than double the average calories of recipes with 1-5 ingredients, suggesting that recipe complexity (measured by ingredient count) is associated with higher calorie content.

<!-- #endregion -->
<!-- #region -->
# Assessment of Missingness <a name="nmaranalysis"></a>

## NMAR Analysis

I do not believe there is a column in this dataset that is NMAR (Not Missing At Random). The missingness in the dataset appears to be either MCAR (Missing Completely At Random) or MAR (Missing At Random).

The `calories` column has 39 missing values out of 83,782 recipes (0.05% missing). The missingness likely occurred due to data entry errors, parsing failures when extracting nutrition information from the original nutrition string, or cases where the nutrition data was incomplete in the source. This type of missingness is not related to the actual calorie value itself - recipes with high calories are not more or less likely to have missing data than recipes with low calories. If we had access to additional data such as the original source of the nutrition information, whether the recipe was user-submitted or from a professional database, or the date when the recipe was added, we might be able to explain the missingness and make it MAR.

## Missingness Dependency

We performed permutation tests to analyze whether the missingness of the `calories` column depends on other columns in the dataset.

### Test 1: Missingness of calories depends on n_ingredients

We tested whether recipes with missing calories have different numbers of ingredients compared to recipes with non-missing calories. The permutation test revealed:

- **Observed difference in means**: 3.10 ingredients
- **P-value**: 0.0000
- **Conclusion**: The missingness of calories is **dependent** on `n_ingredients` (p < 0.05)

This suggests that recipes with more ingredients are more likely to have missing calorie data, possibly because more complex recipes are harder to calculate nutrition for, or because they were entered differently in the source system.

<iframe src="assets/missingness_ingredients.html" width=800 height=600 frameBorder=0></iframe>

The histogram above shows the empirical distribution of the test statistic from the permutation test. The observed difference (3.10) falls far outside the null distribution, indicating strong evidence that the missingness depends on the number of ingredients.

### Test 2: Missingness of calories depends on is_dessert

We also tested whether dessert recipes are more or less likely to have missing calories:

- **Observed difference in means**: 0.38
- **P-value**: 0.0000
- **Conclusion**: The missingness of calories is **dependent** on whether a recipe is a dessert (p < 0.05)

This indicates that dessert recipes have a different pattern of missing calorie data compared to non-dessert recipes, which could be related to how dessert recipes are categorized or entered in the system.

<!-- #endregion -->
<!-- #region -->

# Hypothesis Testing <a name="hypothesistesting"></a>

We performed two hypothesis tests to investigate relationships between recipe characteristics and calorie content.

## Hypothesis Test 1: Do dessert recipes have higher average calories than non-dessert recipes?

**Null Hypothesis (H₀)**: Dessert recipes and non-dessert recipes have the same average calories.

**Alternative Hypothesis (Hₐ)**: Dessert recipes have higher average calories than non-dessert recipes.

**Test Statistic**: Difference in means (dessert - non-dessert)

**Significance Level**: α = 0.05

**Results**:
- Observed difference: 83.11 calories
- P-value: 0.0000

**Conclusion**: We reject the null hypothesis. There is strong evidence that dessert recipes have higher average calories than non-dessert recipes. The observed difference of 83.11 calories is statistically significant, with dessert recipes averaging 492.67 calories compared to 409.56 calories for non-dessert recipes.

<iframe src="assets/hypothesis_dessert.html" width=800 height=600 frameBorder=0></iframe>

The histogram above shows the empirical distribution of the test statistic from the permutation test. The observed difference (83.11) falls far in the right tail of the null distribution, providing strong evidence against the null hypothesis.

## Hypothesis Test 2: Do recipes with 10+ ingredients have higher average calories than recipes with fewer than 10 ingredients?

**Null Hypothesis (H₀)**: Recipes with 10+ ingredients and recipes with fewer than 10 ingredients have the same average calories.

**Alternative Hypothesis (Hₐ)**: Recipes with 10+ ingredients have higher average calories than recipes with fewer than 10 ingredients.

**Test Statistic**: Difference in means (10+ ingredients - <10 ingredients)

**Significance Level**: α = 0.05

**Results**:
- Observed difference: 121.93 calories
- P-value: 0.0000

**Conclusion**: We reject the null hypothesis. There is strong evidence that recipes with 10+ ingredients have higher average calories than recipes with fewer than 10 ingredients. Recipes with 10+ ingredients average 492.46 calories, while recipes with fewer than 10 ingredients average 370.53 calories.

<iframe src="assets/hypothesis_ingredients.html" width=800 height=600 frameBorder=0></iframe>

The permutation test distribution shows that the observed difference of 121.93 calories is highly unlikely under the null hypothesis, providing strong evidence that recipe complexity (measured by ingredient count) is associated with higher calorie content.

<!-- #endregion -->

<!-- #region -->
# Framing a Prediction Problem <a name="framingpredictionproblem"></a>

In this project, our objective is to create a model that predicts the calorie count in a recipe based on its characteristics. Since the target variable, calories, is a continuous numeric value, this is a **regression** task.

- **Response Variable**: The target variable for our model is **calories**, the number of calories per serving for each recipe. Calories is a fundamental nutritional metric that many people care about when choosing recipes. It's directly available in the dataset and represents a meaningful target for prediction. Understanding what factors influence calorie content can help people make informed dietary choices.

- **Evaluation Metric**: We use **Root Mean Squared Error (RMSE)** to evaluate our model. RMSE is a standard metric for regression problems that penalizes larger errors more heavily than smaller ones, which is important for calorie prediction where being off by 500 calories is much worse than being off by 50 calories. RMSE is in the same units as the response variable (calories), making it interpretable. Compared to Mean Absolute Error (MAE), RMSE gives more weight to outliers, which is appropriate when large prediction errors are particularly problematic. Compared to R², RMSE provides an absolute measure of error that is easier to interpret in the context of the problem (e.g., "our model is off by an average of X calories").

- **Features Available at Time of Prediction**: At the time of prediction, we would have access to recipe characteristics that are known before the recipe is prepared:
  - `n_ingredients`: Number of ingredients (quantitative)
  - `ingredients_list`: List of ingredients (nominal - can be used to create features)
  - `minutes`: Cooking time in minutes (quantitative)
  - `n_steps`: Number of steps in the recipe (quantitative)
  - `tags_list`: Recipe tags/categories (nominal)
  - `primary_category`: Primary recipe category (nominal)
  - `n_tags`: Number of tags (quantitative)
  - `description`: Recipe description text (can be used for text-based features)
  - `name`: Recipe name (can potentially be used for text-based features)

- **Features NOT Used**: We do not use other nutrition values (`total_fat_pdv`, `sugar_pdv`, `sodium_pdv`, `protein_pdv`, `saturated_fat_pdv`, `carbs_pdv`) because these are derived from the same nutrition calculation as calories and would create circular reasoning. If we could predict these, we could likely predict calories directly. We also do not use `submitted` (submission date), `contributor_id`, or `id` as these are not relevant for predicting calories.

<!-- #endregion -->
<!-- #region -->

# Baseline Model <a name="baselinemodel"></a>

- **Description**: The baseline model is a Linear Regression model that predicts recipe calories using two features: the number of ingredients (`n_ingredients`) and the primary recipe category (`primary_category`).

- **Features Used**:
  - **Quantitative Features (1)**: `n_ingredients` - Number of ingredients in the recipe. This feature is used as-is without transformation.
  - **Nominal Features (1)**: `primary_category` - The primary category of the recipe (e.g., "main-dish", "desserts", "breakfast", etc.). This categorical feature is encoded using OneHotEncoder, which creates binary indicator variables for each category (with one category dropped to avoid multicollinearity).

- **Feature Transformations**: The `primary_category` feature is one-hot encoded using sklearn's `OneHotEncoder` with `drop='first'` to avoid the dummy variable trap. This transforms the single categorical column into multiple binary columns, one for each category (minus one reference category). The `n_ingredients` feature is passed through without transformation.

- **Performance**: The baseline model achieves a test RMSE of 496.00 calories. This means that on average, the model's predictions are off by approximately 496 calories from the true calorie values. The training RMSE is 516.51 calories, and the test RMSE is actually slightly lower, indicating good generalization with no signs of overfitting.

- **Model Assessment**: This baseline model is a reasonable starting point, but there is significant room for improvement. The model uses only two simple features, and while these features do show associations with calories (as demonstrated in the exploratory data analysis), they capture only a limited amount of the variation in calorie content. The relatively high RMSE (496 calories) compared to the mean calorie count (approximately 419 calories) represents a 118.5% error rate, suggesting that many other factors influence recipe calories that are not captured by these two features alone. In the final model, we will add more features and potentially use a more sophisticated modeling algorithm to improve performance.

<!-- #endregion -->

# Final Model <a name="finalmodel"></a>

For our final model, we transitioned from the Linear Regression model to a **Random Forest Regressor** due to the limitations of the baseline model in terms of performance. The Random Forest Regressor was chosen because it handles non-linear relationships effectively, is robust to overfitting, and performs well on datasets with mixed feature types.

- **Description**: Our final model uses additional features beyond the baseline, including `minutes` (cooking time), `n_steps` (number of steps), and several engineered features derived from the ingredients list. These features were added based on domain knowledge and exploratory data analysis, which showed that recipe complexity and ingredient composition are important factors in calorie content.

- **New Features Added**:
  1. **`minutes` (Cooking Time)** - Quantitative feature transformed with QuantileTransformer. Cooking time can indicate recipe complexity and preparation method, which may correlate with calorie content. QuantileTransformer handles outliers and non-linear relationships.
  
  2. **`n_steps` (Number of Steps)** - Quantitative feature transformed with StandardScaler. The number of steps indicates recipe complexity, and more complex recipes often have different calorie profiles.
  
  3. **`n_high_cal_ingredients`** - Quantitative feature counting high-calorie ingredients (butter, oil, cheese, cream, sugar, chocolate, bacon, meat, etc.) transformed with StandardScaler. This directly captures calorie-relevant information from the ingredients list.
  
  4. **`n_proteins`** - Quantitative feature counting protein sources (chicken, beef, pork, eggs, beans, etc.) transformed with StandardScaler.
  
  5. **`n_carbs`** - Quantitative feature counting carbohydrate sources (flour, rice, pasta, bread, potatoes, etc.) transformed with StandardScaler.
  
  6. **`n_fats`** - Quantitative feature counting fat sources (butter, oil, cream, cheese, mayonnaise, etc.) transformed with StandardScaler.
  
  7. **`high_cal_density`** - Ratio feature (n_high_cal_ingredients / n_ingredients) transformed with StandardScaler. This captures the proportion of high-calorie ingredients in a recipe.

- **Feature Transformations**: 
  - `n_ingredients` and `minutes`: QuantileTransformer (handles outliers, non-linear relationships)
  - `n_steps`, `n_high_cal_ingredients`, `n_proteins`, `n_carbs`, `n_fats`, `high_cal_density`: StandardScaler (standardizes for models sensitive to scale)
  - `primary_category`: OneHotEncoder (categorical encoding, same as baseline)

- **Reasons for Feature Selection**:
  - The ingredient-based features (`n_high_cal_ingredients`, `n_proteins`, `n_carbs`, `n_fats`) directly capture macronutrient composition, which is fundamental to calorie content. These features extract meaningful information from the raw ingredients list.
  - The `high_cal_density` ratio feature captures the proportion of calorie-dense ingredients, which is more informative than just the count.
  - Cooking time and number of steps provide additional context about recipe complexity that may relate to calorie content.

## Algorithm and Hyperparameters 

We chose the **Random Forest Regressor** for its ability to handle non-linear relationships, reduced susceptibility to overfitting compared to individual decision trees, and effectiveness in modeling complex datasets with mixed feature types.

We used **GridSearchCV** with 5-fold cross-validation for hyperparameter tuning. The hyperparameter grid included:
- **Number of Estimators (n_estimators)**: [50, 100, 200]
- **Maximum Depth of Trees (max_depth)**: [10, 20, 30, None]
- **Minimum Samples Split (min_samples_split)**: [2, 5, 10]

The final model utilized the following best hyperparameters:
- **Number of Estimators**: 200
- **Maximum Depth**: 10
- **Minimum Samples Split**: 10

## Performance

The performance metrics for the final model are shown below:

| **Metric** | **Train Score**       | **Test Score**        |
|------------|-----------------------|-----------------------|
| RMSE       | 474.61 calories    | 491.62 calories    |

### Comparison to Baseline Model
- **Baseline Model Test RMSE**: 496.00 calories
- **Final Model Test RMSE**: 491.62 calories
- **Improvement**: 4.38 calories (0.9% reduction in error)

The final model shows a modest improvement over the baseline. While the improvement is small, it demonstrates that the additional features and more sophisticated algorithm do contribute to better predictions. The model shows good generalization, with test RMSE (491.62) being only slightly higher than training RMSE (474.61), indicating minimal overfitting.

The relatively high RMSE (around 490 calories) is expected given the limitations of the available data. Without ingredient quantities, which are the primary determinant of calories, the model can only make approximate predictions based on recipe characteristics. However, the model is fair in its predictions, with mean predictions closely matching actual means across different recipe categories, as shown in the fairness analysis.

<!-- #endregion -->
<!-- #region -->
# Fairness Analysis <a name="fairnessanalysis"></a>

In this analysis, we compare the model's performance across different recipe categories to determine if the model performs differently for different types of recipes. We analyze whether the model is fair by comparing RMSE and mean prediction bias across all recipe categories.

**Groups Compared**: We compare each recipe category (appetizers, beverages, breakfast, desserts, dinner, lunch, main-dish, other, salads, side-dishes, soups) against all other categories combined.

**Evaluation Metric**: Root Mean Squared Error (RMSE)

**Question**: Does the model perform worse for certain recipe categories than it does for others?

**Hypotheses**:
- **Null Hypothesis (H₀)**: The model is fair. The RMSE for each recipe category and all other categories are roughly the same, and any differences are due to random chance.
- **Alternative Hypothesis (Hₐ)**: The model is unfair. The RMSE for certain recipe categories is different from the RMSE for all other categories.

**Test Statistic**: Difference in RMSE (Category RMSE - Others RMSE)

**Significance Level**: α = 0.05

## Results

The fairness analysis revealed significant differences in model performance across recipe categories:

### Categories with Significantly Worse Performance (Higher RMSE):

- **Desserts**: RMSE is 347.49 calories higher than all other categories (p = 0.0000). The model has an RMSE of 765.50 calories for desserts compared to 418.01 calories for non-desserts. Mean prediction error: +2.82%.

- **Other**: RMSE is 67.66 calories higher than all other categories (p = 0.0000). Mean prediction error: +2.24%.

- **Lunch**: RMSE is 201.64 calories higher than all other categories (p = 0.0000). Mean prediction error: -16.11% (model under-predicts).

### Categories with Significantly Better Performance (Lower RMSE):

- **Salads**: RMSE is 106.27 calories lower than all other categories (p = 0.0000). Mean prediction error: -0.40%.

<iframe src="assets/fairness_rmse_by_category.html" width=800 height=600 frameBorder=0></iframe>

The bar chart above shows RMSE by recipe category. Desserts have the highest RMSE (765.50 calories), while salads have the lowest (244.51 calories).

<iframe src="assets/fairness_actual_vs_predicted.html" width=800 height=600 frameBorder=0></iframe>

The grouped bar chart above compares actual mean calories vs predicted mean calories for each category. While the model's mean predictions are generally close to actual means, there are some discrepancies, particularly for lunch recipes where the model under-predicts.

<iframe src="assets/fairness_percent_difference.html" width=800 height=600 frameBorder=0></iframe>

The bar chart above shows the percent difference between predicted and actual mean calories. Lunch recipes show the largest under-prediction (-16.11%), while beverages show a slight over-prediction (+7.94%).

## Conclusion

The permutation tests reveal that the model is **not fair** across all recipe categories. The model performs significantly worse for dessert recipes, with an RMSE that is 347.49 calories higher than for non-dessert recipes. This indicates that the model makes larger errors when predicting calories for dessert recipes, which is a fairness concern.

The poor performance for desserts is likely due to the high variability in dessert calorie content (as seen in the EDA, desserts have a standard deviation of 819.54 calories) and the fact that desserts can range from simple fruit-based recipes (low calories) to rich, complex baked goods (very high calories). The model struggles to capture this wide variability using the available features.

Additionally, the model shows systematic bias for lunch recipes, under-predicting their mean calories by 16.11%. This suggests that lunch recipes may have characteristics that the model does not adequately capture.

These findings indicate that while the model provides reasonable predictions on average, it has fairness issues that should be addressed, particularly for dessert recipes. Future improvements could include adding more dessert-specific features or using category-specific models.

<!-- #endregion -->

