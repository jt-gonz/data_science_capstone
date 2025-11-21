import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')


def check_chi_square_assumptions(contingency_table):
    """
    Check assumptions for chi-square test:
    1. Independence of observations
    2. Expected frequency >= 5 in at least 80% of cells
    3. No expected frequency < 1
    """
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    total_cells = expected.size
    cells_below_5 = np.sum(expected < 5)
    cells_below_1 = np.sum(expected < 1)
    percent_above_5 = ((total_cells - cells_below_5) / total_cells) * 100

    assumptions = {
        'expected_freq': expected,
        'cells_below_5': cells_below_5,
        'cells_below_1': cells_below_1,
        'percent_above_5': percent_above_5,
        'assumptions_met': (percent_above_5 >= 80) and (cells_below_1 == 0)
    }

    return assumptions


def calculate_cramers_v(chi2, n, contingency_table):
    """Calculate Cramer's V effect size"""
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    return cramers_v


def interpret_cramers_v(v, df_min):
    """Interpret Cramer's V based on degrees of freedom"""
    if df_min == 1:
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    elif df_min == 2:
        if v < 0.07:
            return "negligible"
        elif v < 0.21:
            return "small"
        elif v < 0.35:
            return "medium"
        else:
            return "large"
    else:
        if v < 0.06:
            return "negligible"
        elif v < 0.17:
            return "small"
        elif v < 0.29:
            return "medium"
        else:
            return "large"


def four_step_hypothesis_test(feature_name, contingency_table, alpha=0.05):
    """
    Perform 4-step hypothesis testing:
    1. State hypotheses
    2. Calculate test statistic
    3. Find p-value
    4. Make decision and interpret
    """
    print(f"\n{'=' * 80}")
    print(f"CHI-SQUARE TEST: {feature_name} vs Outcome Category")
    print(f"{'=' * 80}\n")

    # Step 1: State hypotheses
    print("STEP 1: STATE HYPOTHESES")
    print(f"H₀ (Null): There is no association between {feature_name} and Outcome Category")
    print(f"H₁ (Alternative): There is an association between {feature_name} and Outcome Category")
    print(f"Significance level: α = {alpha}\n")

    # Check assumptions
    print("CHECKING ASSUMPTIONS:")
    assumptions = check_chi_square_assumptions(contingency_table)
    print(f"1. Independence: Assumed (each observation is independent)")
    print(f"2. Expected frequencies ≥ 5: {assumptions['percent_above_5']:.1f}% of cells")
    print(f"3. No expected frequency < 1: {assumptions['cells_below_1']} cells below 1")
    print(f"4. Assumptions met: {'✓ YES' if assumptions['assumptions_met'] else '✗ NO (use with caution)'}\n")

    # Step 2: Calculate test statistic
    print("STEP 2: CALCULATE TEST STATISTIC")
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square statistic (χ²): {chi2:.4f}")
    print(f"Degrees of freedom: {dof}\n")

    # Step 3: Find p-value
    print("STEP 3: FIND P-VALUE")
    print(f"P-value: {p_val:.6f}\n")

    # Step 4: Make decision
    print("STEP 4: MAKE DECISION AND INTERPRET")
    if p_val < alpha:
        decision = "REJECT"
        interpretation = f"There IS sufficient evidence of an association between {feature_name} and Outcome Category"
    else:
        decision = "FAIL TO REJECT"
        interpretation = f"There is NOT sufficient evidence of an association between {feature_name} and Outcome Category"

    print(f"Decision: {decision} the null hypothesis (p = {p_val:.6f} {'<' if p_val < alpha else '≥'} {alpha})")
    print(f"Interpretation: {interpretation}\n")

    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    cramers_v = calculate_cramers_v(chi2, n, contingency_table)
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    effect_interpretation = interpret_cramers_v(cramers_v, min_dim)

    print(f"EFFECT SIZE:")
    print(f"Cramer's V: {cramers_v:.4f} ({effect_interpretation} effect)\n")

    return {
        'chi2': chi2,
        'p_value': p_val,
        'dof': dof,
        'cramers_v': cramers_v,
        'effect_size': effect_interpretation,
        'decision': decision,
        'assumptions_met': assumptions['assumptions_met']
    }


def plot_chi_square_distribution(chi2_stat, dof, p_value, feature_name, alpha=0.05):
    """Plot chi-square distribution with test statistic and critical value"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate chi-square distribution
    x = np.linspace(0, chi2_stat + 10, 1000)
    y = stats.chi2.pdf(x, dof)

    # Plot distribution
    ax.plot(x, y, 'b-', linewidth=2, label=f'χ² distribution (df={dof})')

    # Fill rejection region
    critical_value = stats.chi2.ppf(1 - alpha, dof)
    x_fill = x[x >= critical_value]
    y_fill = stats.chi2.pdf(x_fill, dof)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='red', label=f'Rejection region (α={alpha})')

    # Mark test statistic
    ax.axvline(chi2_stat, color='green', linestyle='--', linewidth=2, label=f'Test statistic (χ²={chi2_stat:.2f})')
    ax.axvline(critical_value, color='red', linestyle='--', linewidth=2, label=f'Critical value ({critical_value:.2f})')

    # Add text
    ax.text(chi2_stat, max(y) * 0.8, f'p-value = {p_value:.6f}',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    ax.set_xlabel('χ² Value', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Chi-Square Distribution: {feature_name} vs Outcome Category', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_side_by_side_bars(contingency_table, feature_name):
    """Create side-by-side bar plot for the contingency table"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Grouped bar chart (counts)
    contingency_table.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title(f'{feature_name} by Outcome Category (Counts)', fontsize=12, fontweight='bold')
    ax1.set_xlabel(feature_name, fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Stacked percentage chart
    contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    contingency_pct.plot(kind='bar', stacked=True, ax=ax2, width=0.8)
    ax2.set_title(f'{feature_name} by Outcome Category (Proportions)', fontsize=12, fontweight='bold')
    ax2.set_xlabel(feature_name, fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


def analyze_feature(df, feature_col, outcome_col='Outcome Category', alpha=0.05):
    """Complete analysis for one feature"""
    print(f"\n{'#' * 80}")
    print(f"# ANALYZING: {feature_col}")
    print(f"{'#' * 80}")

    # Filter to only Positive and Negative outcomes
    df_filtered = df[df[outcome_col].isin(['Positive', 'Negative'])].copy()

    # Handle Sport special case (NA means non-athlete)
    if feature_col == 'Sport':
        df_filtered[feature_col] = df_filtered[feature_col].fillna('Non-Athlete')
        df_filtered[feature_col] = df_filtered[feature_col].replace('', 'Non-Athlete')
        df_filtered.loc[df_filtered[feature_col] != 'Non-Athlete', feature_col] = 'Athlete'

    # Remove missing values
    df_clean = df_filtered[[feature_col, outcome_col]].dropna()

    print(f"\nSample size: {len(df_clean)} (after filtering for Positive/Negative outcomes)")
    print(f"Unique values in {feature_col}: {df_clean[feature_col].nunique()}")

    # Create contingency table
    contingency_table = pd.crosstab(df_clean[feature_col], df_clean[outcome_col])

    # Save contingency table to CSV
    safe_filename = feature_col.replace('/', '_').replace(' ', '_')
    csv_filename = f"contingency_table_{safe_filename}.csv"
    contingency_table.to_csv(csv_filename)
    print(f"Contingency table saved to: {csv_filename}")

    print("\n" + "=" * 50)
    print("TWO-WAY CONTINGENCY TABLE")
    print("=" * 50)
    print(contingency_table)
    print("\n")

    # Perform hypothesis test
    results = four_step_hypothesis_test(feature_col, contingency_table, alpha)

    # Create visualizations
    print("Creating visualizations...")
    fig1 = plot_chi_square_distribution(results['chi2'], results['dof'],
                                        results['p_value'], feature_col, alpha)
    plt.show()

    fig2 = plot_side_by_side_bars(contingency_table, feature_col)
    plt.show()

    return results, contingency_table


def run_complete_analysis(df, alpha=0.05):
    """Run analysis for all features and generate summary"""

    # Rename columns for better readability
    column_mapping = {
        'Recipient Primary Major Abbreviation': 'Primary Major',
        'Recipient Primary College Abbreviation': 'Primary College',
        'SPORT_1': 'Sport',
        'RESIDENCY': 'Residency',
        'SEX': 'Sex',
        'FTPT': 'Enrollment Status',
        'Recipient Graduation Date Season': 'Graduation Season'
    }

    df = df.rename(columns=column_mapping)
    df = df[df["Sex"] != "N"]

    features = {
        'Primary Major': 'Does the major matter to predict outcome',
        'Primary College': 'Does the primary college matter',
        'Sport': 'Does being an athlete vs non-athlete matter',
        'Residency': 'Importance of residency type to predict outcome',
        'Sex': 'Does the student\'s sex matter',
        'Enrollment Status': 'Does being full-time vs part-time matter',
        'Graduation Season': 'Does graduation season matter'
    }

    summary_results = []

    for feature, description in features.items():
        print(f"\n\n{'*' * 100}")
        print(f"FEATURE: {feature}")
        print(f"RESEARCH QUESTION: {description}")
        print(f"{'*' * 100}")

        try:
            results, contingency_table = analyze_feature(df, feature, alpha=alpha)
            summary_results.append({
                'Feature': feature,
                'Chi-Square': results['chi2'],
                'P-Value': results['p_value'],
                'DOF': results['dof'],
                'Cramers V': results['cramers_v'],
                'Effect Size': results['effect_size'],
                'Decision': results['decision'],
                'Significant': 'Yes' if results['p_value'] < alpha else 'No',
                'Assumptions Met': 'Yes' if results['assumptions_met'] else 'No'
            })
        except Exception as e:
            print(f"Error analyzing {feature}: {str(e)}")
            summary_results.append({
                'Feature': feature,
                'Chi-Square': np.nan,
                'P-Value': np.nan,
                'DOF': np.nan,
                'Cramers V': np.nan,
                'Effect Size': 'Error',
                'Decision': 'Error',
                'Significant': 'Error',
                'Assumptions Met': 'Error'
            })

    # Create summary table
    print(f"\n\n{'=' * 100}")
    print("SUMMARY OF ALL CHI-SQUARE TESTS")
    print(f"{'=' * 100}\n")

    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))

    # Summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: P-values
    valid_results = summary_df[summary_df['P-Value'].notna()]
    colors = ['red' if p < alpha else 'blue' for p in valid_results['P-Value']]
    axes[0, 0].barh(valid_results['Feature'], valid_results['P-Value'], color=colors)
    axes[0, 0].axvline(alpha, color='black', linestyle='--', linewidth=2, label=f'α = {alpha}')
    axes[0, 0].set_xlabel('P-Value', fontsize=12)
    axes[0, 0].set_title('P-Values for All Features', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Plot 2: Cramer's V
    colors_v = ['green' if v > 0.3 else 'orange' if v > 0.1 else 'gray'
                for v in valid_results['Cramers V']]
    axes[0, 1].barh(valid_results['Feature'], valid_results['Cramers V'], color=colors_v)
    axes[0, 1].set_xlabel("Cramer's V", fontsize=12)
    axes[0, 1].set_title("Effect Sizes (Cramer's V)", fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Plot 3: Significance summary
    sig_counts = summary_df['Significant'].value_counts()
    axes[1, 0].pie(sig_counts, labels=sig_counts.index, autopct='%1.1f%%',
                   colors=['lightcoral', 'lightblue'], startangle=90)
    axes[1, 0].set_title('Proportion of Significant Results', fontsize=14, fontweight='bold')

    # Plot 4: Chi-square statistics
    axes[1, 1].barh(valid_results['Feature'], valid_results['Chi-Square'], color='purple', alpha=0.6)
    axes[1, 1].set_xlabel('Chi-Square Statistic', fontsize=12)
    axes[1, 1].set_title('Chi-Square Statistics for All Features', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

    # Final interpretation
    print(f"\n{'=' * 100}")
    print("KEY FINDINGS")
    print(f"{'=' * 100}\n")

    significant_features = summary_df[summary_df['Significant'] == 'Yes']
    if len(significant_features) > 0:
        print(f"Features with SIGNIFICANT associations (p < {alpha}):")
        for _, row in significant_features.iterrows():
            print(f"  • {row['Feature']}: χ² = {row['Chi-Square']:.2f}, p = {row['P-Value']:.6f}, "
                  f"Cramer's V = {row['Cramers V']:.4f} ({row['Effect Size']} effect)")
    else:
        print(f"No features showed significant associations at α = {alpha}")

    print("\n")
    nonsignificant_features = summary_df[summary_df['Significant'] == 'No']
    if len(nonsignificant_features) > 0:
        print(f"Features with NO significant associations (p ≥ {alpha}):")
        for _, row in nonsignificant_features.iterrows():
            print(f"  • {row['Feature']}: χ² = {row['Chi-Square']:.2f}, p = {row['P-Value']:.6f}")

    return summary_df


# Example usage:
# Assuming you have a DataFrame called 'df' with the required columns
df = pd.read_csv('data/final.csv')
summary = run_complete_analysis(df, alpha=0.05)
