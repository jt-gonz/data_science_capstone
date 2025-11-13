"""
Data Visualization Module
Contains all plotting and visualization functions.
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set color palette
gen_palette = sns.color_palette(["#e41a1c", "#000000"])  # red & black
sns.set_palette(gen_palette)


def plot_top_majors(clean_df, n=5):
    """Plot top N primary majors."""
    top_majors = clean_df["Recipient Primary Major Abbreviation"].value_counts().nlargest(n)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=top_majors.index, y=top_majors.values)
    ax.set_title(f"Top {n} Primary Majors")
    ax.set_xlabel("Major")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_majors_by_term(clean_df, top_n=10):
    """Plot top majors for each graduation term."""
    plot_df = clean_df.dropna(subset=["Recipient Primary Major", "Recipient Graduation Date"]).copy()

    counts = (
        plot_df.groupby(["Recipient Graduation Date", "Recipient Primary Major"])
        .size()
        .reset_index(name="Count")
    )

    season_order = {"Winter": 1, "Summer": 2, "Fall": 3}

    def term_sort_key(term):
        season, year = term.split()
        return (int(year), season_order[season])

    counts = counts.sort_values(by="Recipient Graduation Date", key=lambda x: x.map(term_sort_key))
    terms = counts["Recipient Graduation Date"].unique()

    for term in terms:
        subset = counts[counts["Recipient Graduation Date"] == term].nlargest(top_n, "Count")

        plt.figure(figsize=(8, 5))
        sns.barplot(data=subset, x="Recipient Primary Major", y="Count")
        plt.title(f"Top {top_n} Majors – {term}")
        plt.xlabel("Primary Major")
        plt.ylabel("Number of Graduates")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_top_secondary_majors(clean_df, n=5):
    """Plot top N secondary majors."""
    sec_majors_count = {}
    for sec_major in clean_df["Recipient Secondary Majors"]:
        if pd.isna(sec_major):
            continue
        if "," in sec_major:
            tmp = sec_major.split(",")
            for i in tmp:
                i = i.strip()
                sec_majors_count[i] = sec_majors_count.get(i, 0) + 1
        else:
            sec_majors_count[sec_major] = sec_majors_count.get(sec_major, 0) + 1

    top_sec_majors = Counter(sec_majors_count).most_common(n)
    df = pd.DataFrame(top_sec_majors, columns=["Major", "Count"])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Major", y="Count")
    ax.set_title(f"Top {n} Secondary Majors")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_top_schools(clean_df, n=5):
    """Plot top N primary colleges."""
    top_schools = clean_df["Recipient Primary College"].value_counts().nlargest(n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_schools.index, y=top_schools.values)
    plt.title(f"Top {n} Primary Colleges")
    plt.xlabel("College")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_outcomes_by_term(clean_df):
    """Plot graduate outcomes for each term."""
    plot_df = clean_df.dropna(subset=["Recipient Graduation Date", "Outcome Category"]).copy()

    outcome_counts = (
        plot_df.groupby(["Recipient Graduation Date", "Outcome Category"])
        .size()
        .reset_index(name="Count")
    )

    all_terms = plot_df["Recipient Graduation Date"].dropna().unique()
    outcome_categories = ["Positive", "Negative", "Inconclusive"]

    full_index = pd.MultiIndex.from_product(
        [all_terms, outcome_categories],
        names=["Recipient Graduation Date", "Outcome Category"]
    )
    outcome_counts = (
        outcome_counts.set_index(["Recipient Graduation Date", "Outcome Category"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    season_order = {"Winter": 1, "Summer": 2, "Fall": 3}

    def term_sort_key(term):
        season, year = term.split()
        return (int(year), season_order[season])

    outcome_counts = outcome_counts.sort_values(
        by="Recipient Graduation Date",
        key=lambda x: x.map(term_sort_key)
    )

    sns.set(style="whitegrid")
    palette = {"Positive": "green", "Negative": "red", "Inconclusive": "gray"}
    terms = outcome_counts["Recipient Graduation Date"].unique()

    for term in terms:
        subset = outcome_counts[outcome_counts["Recipient Graduation Date"] == term]

        plt.figure(figsize=(6, 5))
        sns.barplot(
            data=subset,
            x="Outcome Category",
            y="Count",
            hue="Outcome Category",
            palette=palette,
            order=outcome_categories,
            legend=False
        )

        plt.title(f"Graduate Outcomes – {term}")
        plt.xlabel("Outcome Category")
        plt.ylabel("Number of Graduates")
        plt.tight_layout()
        plt.show()


def plot_outcomes_by_major(clean_df, all_years=True, top_n=10):
    """Plot graduate outcomes by major."""
    if all_years:
        plot_df = clean_df.dropna(subset=["Outcome Category", "Recipient Primary Major"]).copy()
    else:
        plot_df = clean_df.dropna(
            subset=["Recipient Graduation Date", "Outcome Category", "Recipient Primary Major"]
        ).copy()

    counts = (
        plot_df.groupby(["Recipient Primary Major", "Outcome Category"])
        .size()
        .reset_index(name="Count")
    )

    top_majors = (
        counts.groupby("Recipient Primary Major")["Count"]
        .sum()
        .nlargest(top_n)
        .index
    )
    counts = counts[counts["Recipient Primary Major"].isin(top_majors)]

    outcome_categories = ["Positive", "Negative", "Inconclusive"]
    palette = {"Positive": "green", "Negative": "red", "Inconclusive": "gray"}

    pivot_df = counts.pivot_table(
        index="Recipient Primary Major",
        columns="Outcome Category",
        values="Count",
        fill_value=0
    )

    for o in outcome_categories:
        if o not in pivot_df.columns:
            pivot_df[o] = 0

    pivot_df = pivot_df[outcome_categories]

    sns.set(style="whitegrid")
    pivot_df.plot(
        kind="bar",
        stacked=True,
        color=[palette[o] for o in outcome_categories],
        figsize=(10, 6)
    )

    plt.title("Graduate Outcomes by Major (All Years Combined)")
    plt.xlabel("Primary Major")
    plt.ylabel("Number of Graduates")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Outcome Category")
    plt.tight_layout()
    plt.show()


def plot_outcomes_by_residency(clean_df):
    """Plot outcome distribution by residency type."""
    df = clean_df.dropna(subset=["RESIDENCY", "Outcome Category"]).copy()

    counts = (
        df.groupby(["RESIDENCY", "Outcome Category"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["Positive", "Negative", "Inconclusive"]:
        if col not in counts.columns:
            counts[col] = 0

    outcome_colors = {
        "Positive": "#4CAF50",
        "Negative": "#F44336",
        "Inconclusive": "#9E9E9E"
    }

    residencies = ["Resident", "Out of State", "International", "Undeclared", "Permanent Resident"]
    residencies_present = [r for r in residencies if r in counts.index]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
    axes = axes.flatten()

    for i, residency in enumerate(residencies_present):
        values = counts.loc[residency, ["Positive", "Negative", "Inconclusive"]]
        total = values.sum()

        axes[i].pie(
            values,
            labels=values.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=[outcome_colors[k] for k in values.index],
            textprops={'fontsize': 10}
        )
        axes[i].set_title(f"{residency} (n={total})", fontsize=14)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Outcome Distribution by Residency Type", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_salary_by_major(clean_df, top_n=5):
    """Plot median salary for top and bottom majors."""
    df = clean_df.dropna(subset=["Recipient Primary Major", "Annual Salary"]).copy()

    df["Annual Salary"] = (
        df["Annual Salary"]
        .astype(str)
        .replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

    Q1 = df["Annual Salary"].quantile(0.25)
    Q3 = df["Annual Salary"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df[(df["Annual Salary"] >= lower_bound) & (df["Annual Salary"] <= upper_bound)]

    salary_by_major = (
        df_filtered.groupby("Recipient Primary Major")["Annual Salary"]
        .median()
        .reset_index()
        .sort_values(by="Annual Salary", ascending=False)
    )

    top5 = salary_by_major.head(top_n)
    bottom5 = salary_by_major.tail(top_n)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def k_formatter(x, pos):
        return f"${x / 1000:.0f}k"

    formatter = FuncFormatter(k_formatter)

    sns.barplot(data=top5, x="Recipient Primary Major", y="Annual Salary", color="#2E7D32", ax=axes[0])
    axes[0].set_title(f"Top {top_n} Majors by Median Salary", fontsize=13, weight="bold")
    axes[0].set_xlabel("Major")
    axes[0].set_ylabel("Median Annual Salary")
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=bottom5, x="Recipient Primary Major", y="Annual Salary", color="#C62828", ax=axes[1])
    axes[1].set_title(f"Bottom {top_n} Majors by Median Salary", fontsize=13, weight="bold")
    axes[1].set_xlabel("Major")
    axes[1].set_ylabel("")
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_employment_type(clean_df):
    """Plot employment type distribution."""
    type_counts = clean_df["Employment Type"].value_counts()

    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(type_counts, colors=gen_palette, autopct="%1.1f%%")

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    plt.legend(
        wedges,
        ["Full Time", "Part Time"],
        title="Job Type",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    plt.title("Employment Type Distribution")
    plt.tight_layout()
    plt.show()


def plot_sex_distribution(clean_df):
    """Plot sex distribution."""
    palette = {
        "F": "#f4a582",
        "M": "#92c5de",
        "N": "#fddbc7"
    }

    sex_counts = clean_df["SEX"].dropna().value_counts()

    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        sex_counts,
        colors=[palette[g] for g in sex_counts.index],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.legend(
        wedges,
        ["Female", "Male", "N"],
        title="Sex",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    plt.title("Sex Distribution")
    plt.tight_layout()
    plt.show()


def plot_top_sports(clean_df, n=5):
    """Plot top N sports."""

    def remove_sex(sport):
        if pd.isna(sport):
            return sport
        sport = sport.split("(")[0].strip()
        return sport

    sports = clean_df["SPORT_1"].map(remove_sex)
    top_sports = sports.dropna().value_counts().nlargest(n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_sports.index, y=top_sports.values)
    plt.title(f"Top {n} Sports")
    plt.xlabel("Sport")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_outcomes_over_time(clean_df):
    """Plot percentage of each outcome category over time."""
    counts = clean_df.groupby(["YearSeason", "Outcome Category"]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    def label_for(value):
        year = int(value)
        season_num = int(round((value - year) * 10))
        season = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}.get(season_num, "")
        return f"{season} {year}"

    labels = [label_for(v) for v in percentages.index]
    x_positions = np.arange(len(percentages.index))

    plt.figure(figsize=(12, 6))
    for cat in percentages.columns:
        plt.plot(x_positions, percentages[cat], marker="o", label=cat)

    plt.title("Percentage of Each Outcome Category Over Time")
    plt.xlabel("Graduation Term (Season/Year)")
    plt.ylabel("Percentage (%)")
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Outcome Category")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load cleaned data
    clean_data = pd.read_csv('data/clean.csv')

    print("Generating visualizations...")

    # Example: Generate a few key plots
    plot_top_majors(clean_data)
    plot_outcomes_by_major(clean_data)
    plot_salary_by_major(clean_data)
