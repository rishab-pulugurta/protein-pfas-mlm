import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Read the original CSV file
data = pd.read_csv('/workspace/rtp/CF-PepMLM/data/data.csv')

# First, split off the test set (10% of the data)
train_val, test = train_test_split(data, test_size=0.1, random_state=42)

# Then split the remaining data into train and validation sets (80% train, 10% validation)
train, val = train_test_split(train_val, test_size=0.11111, random_state=42)  # 0.11111 of 90% is 10% of total

# Create a datasets directory if it doesn't exist
os.makedirs('datasets', exist_ok=True)

# Save the splits to CSV files
train.to_csv('./datasets/train.csv', index=False)
val.to_csv('./datasets/val.csv', index=False)
test.to_csv('./datasets/test.csv', index=False)

print(f"Original data shape: {data.shape}")
print(f"Train set shape: {train.shape}")
print(f"Validation set shape: {val.shape}")
print(f"Test set shape: {test.shape}")