response_names = [
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]
# normalized_histogram, count_histogram, density_plot
dict_columns_type_plots = {
    "Churn": ["count_histogram", "churn_distribution.png"],
    "Customer_Age": ["count_histogram", "customer_age_distribution.png"],
    "Marital_Status": ["normalized_histogram", "marital_status_distribution.png"],
    "Total_Trans_Ct": ["density_plot", "total_transaction_distribution.png"],
}
keep_cols = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]
param_grid_input = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt"],
    "max_depth": [4, 5, 100],
    "criterion": ["gini", "entropy"],
}


categorical_variables = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
