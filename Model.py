# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Step 2: Upload the File
from google.colab import files

# This will prompt you to upload an Excel file from your local machine
uploaded = files.upload()

# Step 3: Load the Dataset (read_excel for .xlsx file)
df = pd.read_excel(list(uploaded.keys())[0])  # Automatically picks the uploaded Excel file
print("\nDataset Preview:")
print(df.head())

# Step 4: Check Missing Data
missing_values = df.isnull().sum()  # Total missing values in each column
total_values = len(df)  # Total rows
missing_percentage = (missing_values / total_values) * 100  # Percentage of missing data in each column
missing_data_df = pd.DataFrame({'Column': df.columns, 'Missing Values': missing_values, 'Percentage': missing_percentage})

print("\n Missing Data Information:")
print(missing_data_df)

# Step 5: Fill Missing Data Using Imputers
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Select numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns  # Select categorical columns

# Imputers for handling missing data
mean_imputer = SimpleImputer(strategy='mean')  # For numerical data
mode_imputer = SimpleImputer(strategy='most_frequent')  # For categorical data

# Fill missing values in numerical columns (only if they exist)
if len(numerical_cols) > 0:
    df[numerical_cols] = mean_imputer.fit_transform(df[numerical_cols])

# Fill missing values in categorical columns (only if they exist)
if len(categorical_cols) > 0:
    df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Step 6: Select Features (X) and Target (y)
# You need to manually specify which columns to use as features (X) and which as target (y)
# Here, we're selecting all columns except the last one as features, and the last column as the target
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]  # The last column is the target variable

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Step 7: Split Data into Train and Test Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Split Complete:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Step 8: Train the Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("\nLinear Regression Model Trained")
print("Coefficients (weights):", regressor.coef_)
print("Intercept (bias):", regressor.intercept_)

# Step 9: Make Predictions on the Test Set
y_pred = regressor.predict(X_test)

# Step 10: Calculate Performance Metrics (R2 Score and Mean Squared Error)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

