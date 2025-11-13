"""
Data Preparation Module
Handles data cleaning, mapping, and transformation operations.
"""

import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser
from rapidfuzz import fuzz


def clean(x):
    """Clean text data by removing parentheses content and converting to lowercase."""
    if pd.isnull(x):
        return np.nan
    index = x.find("(")
    if index != -1:
        return x[:index].lower().strip()
    return x.lower().strip()


def abb_clean(x):
    """Clean abbreviations by converting to lowercase."""
    if pd.isnull(x):
        return np.nan
    return x.lower().strip()


def get_term(date_str):
    """Extract season and year from date string."""
    if pd.isnull(date_str):
        return np.nan
    date_str = date_str.strip()
    try:
        date = parser.parse(date_str, fuzzy=True, default=datetime(2000, 1, 1))
    except Exception:
        raise ValueError(f"Unrecognized date format: {date_str}")

    month = date.month
    year = date.year

    if 1 <= month <= 5:
        season = "Winter"
    elif 6 <= month <= 8:
        season = "Summer"
    else:
        season = "Fall"

    return f"{season} {year}"


def categorize_outcome(o):
    """Categorize outcome into Positive, Negative, or Inconclusive."""
    if pd.isna(o):
        return None
    o = str(o).lower()
    if o in ["still looking", "seeking employment"]:
        return "Negative"
    elif "planning to further education" in o or o == "not seeking":
        return "Inconclusive"
    else:
        return "Positive"


def group_outcomes(row):
    outcome = row["Outcome"].strip().lower()
    work = ["working", "employed"]
    education = ["education"]
    volunteer = ["volunteer"]
    military = ["military"]

    if row["Outcome Category"] == "Positive":
        for w in work:
            if w in outcome:
                row["Outcome Group"] = "Employed"

        for e in education:
            if e in outcome:
                row["Outcome Group"] = "Education"

        for v in volunteer:
            if v in outcome:
                row["Outcome Group"] = "Volunteer"

        for m in military:
            if m in outcome:
                row["Outcome Group"] = "Military"
    elif row["Outcome Category"] == "Negative":
        row["Outcome Group"] = "Looking"
    elif row["Outcome Category"] == "Inconclusive":
        row["Outcome Group"] = "Inconclusive"
    else:
        row["Outcome Group"] = "Unknown"

    return row


def map_to_abb(major, mappings):
    """Map major name to abbreviation."""
    if pd.isna(major):
        return np.nan
    major = major.lower().strip()
    pattern = r'\(([^()]*)\)(?!.*\([^()]*\))'
    paren_match = re.search(pattern, major)
    if paren_match:
        return paren_match.group(1).upper()
    else:
        if major in mappings:
            return mappings[major].upper()
        else:
            return major.upper()


def school_to_code(school, school_map):
    """Map school name to code."""
    if pd.isna(school):
        return np.nan
    school = school.lower().strip()
    if school in school_map:
        return school_map[school].upper()
    else:
        return school.upper()


def remove_sex(sport):
    """Remove gender designation from sport name."""
    if pd.isna(sport):
        return sport
    sport = sport.split("(")[0].strip()
    return sport


def create_employer_mapping(employer_names, threshold=90):
    """Create mapping for similar employer names using fuzzy matching."""
    company_names = employer_names.dropna().tolist()

    groups = []
    used = set()

    for i, name in enumerate(company_names):
        if i in used:
            continue

        group = [name]
        used.add(i)

        for j, other in enumerate(company_names):
            if j in used:
                continue

            # Compare similarity
            similarity = fuzz.ratio(name.lower(), other.lower())
            if similarity >= threshold or name.lower() in other.lower() or other.lower() in name.lower():
                group.append(other)
                used.add(j)

        groups.append(group)

    # Create mapping dictionary
    mapping = {}
    for group in groups:
        counts = Counter(group)
        most_common = counts.most_common(1)[0][0]
        for name in group:
            mapping[name] = most_common

    return mapping


def load_and_prepare_data(fds_path='data/fds.csv',
                          mapping_path='data/mapping.csv',
                          school_map_path='data/school_map.csv'):
    """
    Main function to load and prepare the dataset.

    Parameters:
    -----------
    fds_path : str
        Path to the main FDS CSV file
    mapping_path : str
        Path to the major mapping CSV file
    school_map_path : str
        Path to the school mapping CSV file

    Returns:
    --------
    pd.DataFrame
        Cleaned and prepared dataframe
    """
    # Load data
    fds = pd.read_csv(fds_path)

    # Load mappings
    mappings = pd.read_csv(mapping_path).map(abb_clean)
    mappings = pd.Series(mappings["CODE"].values,
                         index=mappings["MAJOR MINOR OR CONCENTRATION"]).to_dict()

    school_map = pd.read_csv(school_map_path)
    school_map = school_map.map(clean)
    school_map = pd.Series(school_map["CODE"].values,
                           index=school_map["SCHOOL"]).to_dict()

    # Create clean dataframe
    final_df = fds

    # Apply transformations
    final_df["Recipient Primary Major Abbreviation"] = final_df["Recipient Primary Major"].apply(
        lambda x: map_to_abb(x, mappings)
    )
    final_df["Recipient Secondary Majors Abbreviation"] = final_df["Recipient Secondary Majors"].apply(
        lambda x: map_to_abb(x, mappings)
    )
    final_df["Recipient Primary College Abbreviation"] = final_df["Recipient Primary College"].apply(
        lambda x: school_to_code(x, school_map)
    )
    final_df["Recipient Graduation Date Season"] = final_df["Recipient Graduation Date"].map(get_term)
    final_df["Outcome Category"] = final_df["Outcome"].map(categorize_outcome)
    final_df = final_df.apply(group_outcomes, axis=1)

    # Create employer mapping
    # employer_mapping = create_employer_mapping(final_df["Employer Name"])
    # final_df["Employer"] = final_df["Employer Name"].map(employer_mapping)

    # Extract season and year for time series analysis
    final_df["Season"] = final_df["Recipient Graduation Date Season"].str.split().str[0]
    final_df["Year"] = (
        final_df["Recipient Graduation Date Season"]
        .str.split()
        .str[1]
        .apply(lambda x: int(x) if pd.notna(x) else None)
    )

    # Order seasons numerically
    season_order = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
    final_df["SeasonOrder"] = final_df["Season"].map(season_order)

    # Create sortable numeric key
    final_df["YearSeason"] = final_df["Year"] + final_df["SeasonOrder"] / 10.0

    return final_df


def save_cleaned_data(final_df, output_path='data/final.csv'):
    """Save cleaned dataframe to CSV."""
    final_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    # Load and prepare data
    clean_data = load_and_prepare_data()

    # Save cleaned data
    save_cleaned_data(clean_data)

    print(f"\nData preparation complete!")
    print(f"Total records: {len(clean_data)}")
    print(f"Columns: {list(clean_data.columns)}")
