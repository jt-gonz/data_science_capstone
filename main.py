"""
Main Script
Orchestrates the data preparation, visualization, and modeling pipeline.
"""

import argparse

from tools.modeling import run_salary_prediction, run_outcome_prediction
from tools.preparation import load_and_prepare_data, save_cleaned_data
from tools.visualization import (
    plot_top_majors,
    plot_top_schools,
    plot_outcomes_by_major,
    plot_salary_by_major,
    plot_outcomes_over_time,
    plot_employment_type,
    plot_sex_distribution,
    plot_top_sports
)


def run_pipeline(mode='all', save_data=True):
    """
    Run the complete analysis pipeline.

    Parameters:
    -----------
    mode : str
        'prepare' - Only data preparation
        'visualize' - Only visualizations
        'model' - Only modeling
        'all' - Complete pipeline (default)
    save_data : bool
        Whether to save cleaned data to CSV
    """

    # Step 1: Data Preparation
    if mode in ['prepare', 'all']:
        print("=" * 60)
        print("STEP 1: DATA PREPARATION")
        print("=" * 60)

        clean_data = load_and_prepare_data(
            fds_path='data/fds.csv',
            mapping_path='data/mapping.csv',
            school_map_path='data/school_map.csv'
        )

        if save_data:
            save_cleaned_data(clean_data, output_path='data/final.csv')

        print(f"\n✓ Data preparation complete!")
        print(f"  Total records: {len(clean_data)}")
        print(f"  Total columns: {len(clean_data.columns)}")

    # Load cleaned data if only running viz or modeling
    if mode in ['visualize', 'model']:
        import pandas as pd
        clean_data = pd.read_csv('data/clean.csv')

    # Step 2: Data Visualization
    if mode in ['visualize', 'all']:
        print("\n" + "=" * 60)
        print("STEP 2: DATA VISUALIZATION")
        print("=" * 60)

        print("\nGenerating visualizations...")

        # Demographics
        print("  - Top majors")
        plot_top_majors(clean_data, n=10)

        print("  - Top schools")
        plot_top_schools(clean_data, n=10)

        print("  - Sex distribution")
        plot_sex_distribution(clean_data)

        print("  - Top sports")
        plot_top_sports(clean_data, n=10)

        # Outcomes
        print("  - Outcomes by major")
        plot_outcomes_by_major(clean_data, all_years=True, top_n=10)

        print("  - Outcomes over time")
        plot_outcomes_over_time(clean_data)

        # Salary
        print("  - Salary by major")
        plot_salary_by_major(clean_data, top_n=5)

        # Employment
        print("  - Employment type distribution")
        plot_employment_type(clean_data)

        print("\n✓ Visualization complete!")

    # Step 3: Machine Learning Models
    if mode in ['model', 'all']:
        print("\n" + "=" * 60)
        print("STEP 3: MACHINE LEARNING MODELING")
        print("=" * 60)

        print("\n--- Salary Prediction Model ---")
        salary_predictor, salary_results = run_salary_prediction(clean_data)

        print("\n--- Outcome Prediction Model ---")
        outcome_predictor, outcome_results = run_outcome_prediction(clean_data)

        print("\n✓ Modeling complete!")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the First Destination Survey (FDS) analysis pipeline'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['prepare', 'visualize', 'model', 'all'],
        default='all',
        help='Which part of the pipeline to run (default: all)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving cleaned data to CSV'
    )

    args = parser.parse_args()

    run_pipeline(mode=args.mode, save_data=not args.no_save)


if __name__ == "__main__":
    # Run with default settings if executed directly
    # Use command line arguments if available
    try:
        main()
    except SystemExit:
        # If no arguments provided, run with defaults
        run_pipeline(mode='all', save_data=True)
