import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# --- V V V V V ---
# Set the input CSV filename here
# input_csv_filename = 'input.csv'
input_csv_filename = sys.argv[1]
feature_column = sys.argv[2]
target_column = sys.argv[3]

# Load data from CSV
data = pd.read_csv(input_csv_filename)
X = data[[feature_column]].values
y = data[target_column].values

# Create a linear regression model and fit it
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Generate predictions
y_pred = model.predict(X)

# Add predictions to the original dataframe
data[f'predicted_{target_column}'] = y_pred

# Get the directory and basename of the input file for output filenames
input_dir = os.path.dirname(input_csv_filename)
base = os.path.splitext(os.path.basename(input_csv_filename))[0]

# Save the updated dataframe to a new CSV file in the same directory
output_csv_path = os.path.join(input_dir, f'{base}_predictions.csv')
data.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")

# Plot the data and the fit
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Fit')
plt.xlabel(feature_column)
plt.ylabel(target_column)
plt.title('Linear Regression Fit')
plt.legend()

# Save the plot as a PNG file with matching basename in the same directory
output_png_path = os.path.join(input_dir, f'{base}_plot.png')
plt.savefig(output_png_path)
print(f"Plot saved to {output_png_path}")

# Show the plot (optional)
plt.show()