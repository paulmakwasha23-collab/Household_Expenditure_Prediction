# Household Expenditure Prediction: A Machine Learning Analysis

## 📌 Project Overview
This project applies machine learning techniques to predict weekly household expenditure (`exp_pw`) across different demographic groups in New Zealand. The analysis traverses a complete end-to-end data science pipeline: from exploratory data analysis and data preprocessing to resolving critical data leakage, handling multicollinearity, and evaluating predictive models.

## 📊 The Dataset
The dataset contains estimated weekly household expenditure and price index weights for 13 distinct household groups (e.g., Beneficiaries, Superannuitants, Income Quintiles, Māori). 
* **Target Variable:** `exp_pw` (Expenditure Per Week in NZD)
* **Features:** Over 80 encoded variables representing specific demographic groups and expenditure categories (e.g., Food, Housing, Transport).

## 🚀 Key Milestones & Methodology

### 1. Identifying and Resolving Data Leakage
Initial model testing yielded an unrealistic R² score of 1.0. A feature audit revealed a data leakage issue: the `weight` column was mathematically derived directly from the target variable (`exp_pw`). Dropping `weight` and `eqv_exp_pw` ensured the model learned genuine patterns rather than reverse-engineering the target.

### 2. The Dummy Variable Trap
Due to the highly categorical nature of the data, applying One-Hot Encoding created perfect multicollinearity across the 80+ features. 
* **Impact:** The standard Linear Regression model suffered catastrophic failure (generating an astronomically negative R² score) due to matrix inversion breakdowns.
* **Solution:** Transitioned to tree-based ensemble learning, which is inherently robust to the Dummy Variable Trap.

### 3. Algorithm Selection: Random Forest
A `RandomForestRegressor` was successfully deployed to navigate the high-dimensional, sparse feature space. By utilizing decision trees, the model effectively mapped the complex, non-linear relationships between specific household demographics and their corresponding expenditure categories.

## 📈 Model Evaluation & Results
The Random Forest model demonstrated strong predictive power on unseen test data:

* **R² Score (0.83):** The model successfully explains 83% of the variance in weekly household spending.
* **Mean Absolute Error (MAE - $11.05):** On average, the model's expenditure predictions deviate by only $11.05 per week.
* **Root Mean Squared Error (RMSE - $19.07):** Indicates that while the model is highly accurate for standard expenses, it appropriately captures the natural variance and occasional extreme outliers inherent in financial data (e.g., housing or vehicle purchases).

### Visual Diagnostics
* **Residual Plot:** Demonstrated tight clustering around the baseline for lower-cost categories, with expected heteroscedasticity emerging in higher-cost categories (like rent).
* **Actual vs. Predicted:** Displayed a strong diagonal alignment, visually confirming the model's high R² score and realistic variance.

## 🛠️ Technologies Used
* **Python**
* **Pandas & NumPy** (Data manipulation and cleaning)
* **Scikit-Learn** (Machine Learning, Train/Test Split, Random Forest)
* **Matplotlib & Seaborn** (Data Visualization)

## 📂 How to View This Project
This project has been exported as a static HTML file for easy viewing of the code, methodology, and final visual diagnostic graphs.

1. Click on the `Household_Expenditure_Analysis.html` file in this repository.
2. **To view the fully rendered notebook with graphs:** [Click here to use the HTML Preview tool](https://htmlpreview.github.io/) and paste the URL of the HTML file into the search bar.
