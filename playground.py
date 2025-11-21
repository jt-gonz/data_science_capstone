import pandas as pd

df = pd.read_csv('data/fds.csv')
series = df["Annual Salary"].dropna()

# Compute Q1, Q3, and IQR
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Filter dataframe
mask = df["Annual Salary"].between(lower, upper) | df["Annual Salary"].isna()
filtered_df = df[mask]

# Print remaining count
print(f"Remaining rows after removing outliers in '{"Annual Salary"}': {filtered_df["Annual Salary"].notna().sum()}")
