# ML_Project-Car_Price_Prediction

This project explores the task of predicting car prices using two regression techniques: Linear Regression and Lasso Regression. It leverages a dataset containing information about various car features to estimate their market value.

### data:
  This directory stores the car data in CSV format. It's assumed the dataset contains attributes like:
  - Car_name
  - year
  - Selling_Price
  - Present_Price
  - Kms_Driven
  - Fuel_Type
  - Seller_Type
  - Transmission
  - Owner

**notebooks**: This directory contains the Jupyter Notebook (`car_price_prediction.ipynb`) for data exploration, preprocessing, model training, evaluation, and visualization.

### Ruuning the Project
The Jupyter Notebook (car_price_prediction.ipynb) should guide you through the following steps:

 1. Data Loading and Exploration: This section loads the car data (assumed to be in a CSV file named car_data.csv in the data directory) and explores its contents, identifying missing values, data types, and distribution of features.
 2. Data Preprocessing: This section handles missing values, encodes categorical variables (like fuel type) into numerical representations, and potentially scales features to ensure they are on a similar range.
 3. Model Training: The notebook will likely have separate sections for Linear Regression and Lasso Regression. Each section will train the respective model on the preprocessed data.
 4. Model Evaluation: This section evaluates the performance of each model using metrics like mean squared error (MSE) and R-squared. It might also compare the predicted vs. actual car prices to visualize the model's accuracy.
 5. Visualization: The notebook might include visualizations to explore the relationship between features and car prices, especially for Lasso Regression, which highlights the most important features.

### Comparison and Benefits

- The project allows you to compare the performance of Linear Regression, which captures the linear relationships between features and price, with Lasso Regression.
- Lasso Regression introduces a regularization penalty that shrinks some coefficient values towards zero, potentially leading to:
    - Improved model interpretability (focusing on the most important features)
    - Reduced risk of overfitting (generalizing better to unseen data)

### Resources
  -  Sklearn Linear Regression Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
  -  Sklearn Lesso Regression Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
  -  Kaggle Car price prediction dataset Description: [https://www.kaggle.com/datasets/bhavikjikadara/car-price-prediction-dataset](https://www.kaggle.com/datasets/bhavikjikadara/car-price-prediction-dataset)

### Customization

- You can modify the Jupyter Notebook to:
    - Experiment with different feature engineering techniques (e.g., creating interaction terms, handling outliers).
    - Try other regression algorithms like Random Forest or Support Vector Regression for comparison.
    - Explore advanced techniques like model selection and hyperparameter tuning for optimal performance.

By exploring both Linear Regression and Lasso Regression, you can gain insights into car price prediction and understand the trade-offs between model complexity and interpretability. This project provides a valuable foundation for further exploration in car price estimation and machine learning applications.
