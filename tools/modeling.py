"""
Modeling Module
Contains machine learning models for predicting salary and outcomes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class SalaryPredictor:
    """Random Forest model for predicting annual salary."""

    def __init__(self, n_estimators=300, random_state=42):
        """
        Initialize the salary predictor.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        random_state : int
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.features = [
            "Recipient Primary Major",
            "Recipient Graduation Date",
            "Outcome",
            "Employer Name",
            "Employer Industry",
            "Job Function",
            "Location",
            "FTPT",
            "RESIDENCY",
            "SEX",
            "SPORT_1"
        ]
        self.target = "Annual Salary"

    def prepare_data(self, df):
        """
        Prepare data for training.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        X, y : Features and target
        """
        df_model = df.dropna(subset=[self.target]).copy()
        X = df_model[self.features]
        y = df_model[self.target]
        return X, y

    def build_pipeline(self):
        """Build the preprocessing and model pipeline."""
        cat_features = self.features

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_features)
            ]
        )

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", rf)
        ])

    def train(self, df, test_size=0.2):
        """
        Train the model.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        test_size : float
            Proportion of data to use for testing

        Returns:
        --------
        dict : Training results including scores and predictions
        """
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.build_pipeline()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results = {
            "r2_score": r2,
            "mae": mae,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "y_test": y_test,
            "y_pred": y_pred
        }

        return results

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from trained model.

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        onehot = self.model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
        feature_names = onehot.get_feature_names_out(self.features)
        importances = self.model.named_steps["regressor"].feature_importances_

        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

        return importance_df

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance."""
        importance_df = self.get_feature_importance(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            color="#2E7D32"
        )
        plt.title("Top 20 Most Important Features for Annual Salary Prediction",
                  fontsize=14, weight="bold")
        plt.xlabel("Feature Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    def print_summary(self, results):
        """Print training summary."""
        print("🌲 Random Forest Regression Summary 🌲")
        print("------------------------------------")
        print(f"R² Score: {results['r2_score']:.3f}")
        print(f"Mean Absolute Error: ${results['mae']:,.2f}")
        print(f"Training samples: {results['n_train']}, Testing samples: {results['n_test']}")


class OutcomePredictor:
    """Random Forest classifier for predicting student outcomes."""

    def __init__(self, n_estimators=300, random_state=42):
        """
        Initialize the outcome predictor.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        random_state : int
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.features = [
            "Recipient Primary Major",
            "Recipient Graduation Date",
            "Employer Name",
            "Employer Industry",
            "Job Function",
            "Location",
            "FTPT",
            "RESIDENCY",
            "SEX",
            "SPORT_1"
        ]
        self.target = "Outcome"

    def prepare_data(self, df):
        """
        Prepare data for training.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        X, y : Features and target
        """
        df_model = df.dropna(subset=[self.target]).copy()
        X = df_model[self.features]
        y = df_model[self.target].astype(str).str.strip()
        return X, y

    def build_pipeline(self):
        """Build the preprocessing and model pipeline."""
        cat_features = self.features

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_features)
            ]
        )

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1
        )

        self.model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf)
        ])

    def train(self, df, test_size=0.2):
        """
        Train the model.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        test_size : float
            Proportion of data to use for testing

        Returns:
        --------
        dict : Training results including predictions and metrics
        """
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.build_pipeline()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        results = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "y_test": y_test,
            "y_pred": y_pred,
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

        return results

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from trained model.

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        onehot = self.model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
        feature_names = onehot.get_feature_names_out(self.features)
        importances = self.model.named_steps["classifier"].feature_importances_

        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

        return importance_df

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance."""
        importance_df = self.get_feature_importance(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            color="#1565C0"
        )
        plt.title("Top 20 Most Important Features for Predicting Student Outcome",
                  fontsize=14, weight="bold")
        plt.xlabel("Feature Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    def print_summary(self, results):
        """Print training summary."""
        print("🎯 Random Forest Classification Summary 🎯")
        print("-----------------------------------------")
        print(results["classification_report"])
        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])


def run_salary_prediction(df):
    """
    Run salary prediction model.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    SalaryPredictor : Trained model
    dict : Training results
    """
    predictor = SalaryPredictor()
    results = predictor.train(df)
    predictor.print_summary(results)
    predictor.plot_feature_importance()

    return predictor, results


def run_outcome_prediction(df):
    """
    Run outcome prediction model.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    OutcomePredictor : Trained model
    dict : Training results
    """
    predictor = OutcomePredictor()
    results = predictor.train(df)
    predictor.print_summary(results)
    predictor.plot_feature_importance()

    return predictor, results


if __name__ == "__main__":
    # Load cleaned data
    clean_data = pd.read_csv('data/clean.csv')

    print("Training Salary Prediction Model...")
    salary_predictor, salary_results = run_salary_prediction(clean_data)

    print("\n" + "=" * 50 + "\n")

    print("Training Outcome Prediction Model...")
    outcome_predictor, outcome_results = run_outcome_prediction(clean_data)
