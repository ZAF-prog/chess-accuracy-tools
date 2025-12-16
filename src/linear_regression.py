#!/usr/bin/env python
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to perform linear regression and save results
def perform_linear_regression(input_file, y_column, x_column):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Select X and Y columns
    X = df[x_column].values.reshape(-1, 1)
    y = df[y_column].values
    
    # Create a linear regression model
    model = LinearRegression()
    
    # Fit the model to the data
    model.fit(X, y)
    
    # Get the coefficients and intercept
    coef = model.coef_[0]
    intercept = model.intercept_
    
    # Predict using the model
    y_pred = model.predict(X)
    
    # Calculate R² (coefficient of determination)
    r2 = model.score(X, y)
    
    # Calculate Y-error (mean absolute error)
    y_error = np.mean(np.abs(y - y_pred))
    
    # Prepare a DataFrame for output
    output_data = {
        'Coefficient': [coef],
        'Intercept': [intercept],
        'R2': [r2],
        'Y-Error': [y_error]
    }
    output_df = pd.DataFrame(output_data)
    
    # Extract the base filename without extension
    import os
    base_filename = os.path.splitext(input_file)[0]
    output_file = f"{base_filename}_{y_column}_vs_{x_column}_linear_regression_results.csv"
    
    # Save the results to a CSV file
    output_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <y_column> <x_column>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    y_column = sys.argv[2]
    x_column = sys.argv[3]
    
    perform_linear_regression(input_file, y_column, x_column)
