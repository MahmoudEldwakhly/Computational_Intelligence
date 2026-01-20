# ================================================================
# CSE473: Computational Intelligence
# Lab Assignment #02
# Name: Mahmoud Elsayd Eldwakhly
# ID : 21P0017
# ================================================================
# Objective:
# Use the pandas library to load and analyze the Iris dataset.
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------

# Option 1: Load directly from an online source
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
# df = pd.read_csv(url)

# Option 2: If Dataset downloaded locally (iris.csv):
# Define column names as the data file doesn't have a header
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv("iris/iris.data", header=None, names=column_names)


print(" Dataset successfully loaded!\n")

# ------------------------------------------------
# 2. Explore the dataset
# ------------------------------------------------

print("---- First 10 rows ----")
print(df.head(10))
print("\n")

print("---- Summary Info ----")
print(df.info())
print("\n")

print("---- Basic Statistics ----")
print(df.describe())
print("\n")

# ------------------------------------------------
# 3. Data Cleaning
# ------------------------------------------------

# Check for missing values
print("---- Missing Values ----")
print(df.isnull().sum())
print("\n")

# Handle missing values if any 
df = df.dropna()

# Ensure data types are appropriate
print("---- Data Types ----")
print(df.dtypes)
print("\n")

# ------------------------------------------------
# 4. Data Analysis
# ------------------------------------------------

# Mean and median for each numerical column
print("---- Mean of Numerical Columns ----")
print(df.mean(numeric_only=True))
print("\n")

print("---- Median of Numerical Columns ----")
print(df.median(numeric_only=True))
print("\n")

# Correlation between numerical features
print("---- Correlation Matrix ----")
print(df.corr(numeric_only=True))
print("\n")

# Group by species and calculate mean for each feature
print("---- Mean Features by Species ----")
print(df.groupby("species").mean(numeric_only=True))
print("\n")

# ------------------------------------------------
# 5. Visualization (Histograms)
# ------------------------------------------------

df.hist(figsize=(10, 8), bins=15, edgecolor='black')
plt.suptitle("Histograms of Iris Dataset Features", fontsize=14)
plt.show()

