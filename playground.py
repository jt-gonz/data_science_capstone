import pandas as pd
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
from datetime import datetime

df = pd.read_csv('data/clean.csv')

# --- Step 1: Categorize outcomes ---
def categorize_outcome(o):
    if pd.isna(o):
        return None
    o = str(o).lower()
    if o in ["still looking", "seeking employment"]:
        return "Negative"
    elif "pursuing or furthering education" in o or o == "not seeking":
        return "Inconclusive"
    else:
        return "Positive"

# --- Step 2: Prepare data ---
df = df.copy()
df["Outcome Category"] = df["Outcome"].apply(categorize_outcome)

# --- Step 3: Split by sports participation ---
sports_df = df[df["SPORT_1"].notna()].copy()
no_sports_df = df[df["SPORT_1"].isna()].copy()

# --- Step 4: Count outcomes ---
sports_counts = sports_df["Outcome Category"].value_counts().reindex(["Positive", "Negative", "Inconclusive"], fill_value=0)
no_sports_counts = no_sports_df["Outcome Category"].value_counts().reindex(["Positive", "Negative", "Inconclusive"], fill_value=0)

# --- Step 5: Plot side-by-side pie charts ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

colors = ["#4CAF50", "#F44336", "#9E9E9E"]  # Positive / Negative / Inconclusive

# Pie 1: Students who play sports
axes[0].pie(
    sports_counts,
    labels=sports_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    explode=[0.05]*3
)
axes[0].set_title("Outcomes — Students Who Play Sports")

# Pie 2: Students who don't play sports
axes[1].pie(
    no_sports_counts,
    labels=no_sports_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    explode=[0.05]*3
)
axes[1].set_title("Outcomes — Students Who Don't Play Sports")

plt.tight_layout()
plt.show()