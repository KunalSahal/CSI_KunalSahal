"""
Advanced ML Models for Loan Approval Prediction
Implements ensemble methods with explainability features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
import warnings

warnings.filterwarnings("ignore")


class LoanApprovalPredictor:
    """
    Advanced ML pipeline for loan approval prediction with explainability
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.shap_explainer = None
        self.is_fitted = False
        self.target_mapping = {}
        self.reverse_target_mapping = {}

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced preprocessing with feature engineering
        """
        df = df.copy()

        # Remove Loan_ID column if it exists (it's not a feature)
        if "Loan_ID" in df.columns:
            df = df.drop("Loan_ID", axis=1)

        # Handle missing values strategically
        df["Gender"].fillna("Male", inplace=True)
        df["Married"].fillna("Yes", inplace=True)
        df["Dependents"].fillna("0", inplace=True)
        df["Self_Employed"].fillna("No", inplace=True)
        df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
        df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
        df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

        # Advanced feature engineering
        df["Income_to_Loan_Ratio"] = (
            df["ApplicantIncome"] + df["CoapplicantIncome"]
        ) / df["LoanAmount"]
        df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (
            df["ApplicantIncome"] + df["CoapplicantIncome"]
        )
        df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["Income_per_Dependent"] = df["Total_Income"] / (
            df["Dependents"].str.replace("3+", "3").astype(int) + 1
        )
        df["Loan_Amount_per_Term"] = df["LoanAmount"] / df["Loan_Amount_Term"]

        # Risk scoring based on multiple factors
        df["Risk_Score"] = (
            df["Credit_History"] * 0.4
            + (df["Total_Income"] > df["Total_Income"].median()).astype(int) * 0.3
            + (df["Education"] == "Graduate").astype(int) * 0.2
            + (df["Property_Area"] == "Urban").astype(int) * 0.1
        )

        # Categorical encoding
        categorical_features = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]

        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
                df[feature] = self.encoders[feature].fit_transform(df[feature])
            else:
                df[feature] = self.encoders[feature].transform(df[feature])

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())

        return df

    def train_models(
        self, df: pd.DataFrame, target_column: str = "Loan_Status"
    ) -> Dict[str, Any]:
        """
        Train multiple models and create ensemble
        """
        # Preprocess data
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        df_processed = self.preprocess_data(df)
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Processed columns: {list(df_processed.columns)}")
        print(
            f"Target column '{target_column}' in processed data: {target_column in df_processed.columns}"
        )

        # Encode target variable
        try:
            print(
                f"Target column '{target_column}' values: {df_processed[target_column].unique()}"
            )
            print(f"Target column dtype: {df_processed[target_column].dtype}")

            # Manual encoding as fallback
            target_values = df_processed[target_column].astype(str)
            unique_values = sorted(target_values.unique())
            print(f"Unique target values: {unique_values}")

            # Create a simple mapping
            value_to_int = {val: idx for idx, val in enumerate(unique_values)}
            print(f"Value mapping: {value_to_int}")

            # Encode manually
            y = np.array([value_to_int[val] for val in target_values])

            # Store the mapping for later use
            self.target_mapping = value_to_int
            self.reverse_target_mapping = {
                idx: val for val, idx in value_to_int.items()
            }

            print(f"Encoded target values: {np.unique(y)}")
            print(f"Target shape: {y.shape}")

        except Exception as e:
            print(f"Error encoding target variable: {e}")
            print(f"Target column data: {df_processed[target_column].head()}")
            raise e

        # Feature selection - exclude non-feature columns
        exclude_columns = [target_column]
        if "Loan_ID" in df_processed.columns:
            exclude_columns.append("Loan_ID")

        feature_columns = [
            col for col in df_processed.columns if col not in exclude_columns
        ]
        X = df_processed[feature_columns]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Scale features
        self.scalers["standard"] = StandardScaler()
        X_train_scaled = self.scalers["standard"].fit_transform(X_train)
        X_test_scaled = self.scalers["standard"].transform(X_test)

        # Model configurations
        model_configs = {
            "random_forest": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "xgboost": {
                "model": xgb.XGBClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0],
                },
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
            },
        }

        # Train and tune models
        results = {}
        for name, config in model_configs.items():
            print(f"Training {name}...")

            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                config["model"], config["params"], cv=5, scoring="roc_auc", n_jobs=-1
            )

            # Use scaled data for better performance
            grid_search.fit(X_train_scaled, y_train)

            # Store best model
            self.models[name] = grid_search.best_estimator_

            # Evaluate model
            y_pred = self.models[name].predict(X_test_scaled)
            y_pred_proba = self.models[name].predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(
                self.models[name], X_train_scaled, y_train, cv=5
            )

            results[name] = {
                "best_params": grid_search.best_params_,
                "roc_auc": roc_auc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

            print(
                f"{name} - ROC AUC: {roc_auc:.4f}, CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})"
            )

        # Create ensemble model
        self.create_ensemble_model(X_train_scaled, y_train, X_test_scaled, y_test)

        # Generate feature importance
        self.calculate_feature_importance(X, feature_columns)

        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.models["random_forest"])

        self.is_fitted = True
        return results

    def create_ensemble_model(self, X_train, y_train, X_test, y_test):
        """
        Create ensemble model using voting
        """
        from sklearn.ensemble import VotingClassifier

        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ("rf", self.models["random_forest"]),
                ("xgb", self.models["xgboost"]),
                ("gb", self.models["gradient_boosting"]),
            ],
            voting="soft",
        )

        # Train ensemble
        voting_clf.fit(X_train, y_train)
        self.models["ensemble"] = voting_clf

        # Evaluate ensemble
        y_pred = voting_clf.predict(X_test)
        y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]

        print(f"Ensemble ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    def calculate_feature_importance(self, X, feature_columns):
        """
        Calculate and store feature importance for each model
        """
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                self.feature_importance[name] = dict(
                    zip(feature_columns, model.feature_importances_)
                )

    def predict(self, X, model_name="ensemble"):
        """
        Make predictions using specified model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train_models() first.")

        # Preprocess input
        X_processed = self.preprocess_data(X)
        feature_columns = [
            col for col in X_processed.columns if col not in ["Loan_Status", "Loan_ID"]
        ]
        X_features = X_processed[feature_columns]

        # Scale features
        X_scaled = self.scalers["standard"].transform(X_features)

        # Make prediction
        predictions = self.models[model_name].predict(X_scaled)
        probabilities = self.models[model_name].predict_proba(X_scaled)

        # Convert predictions back to original labels if mapping exists
        if hasattr(self, "reverse_target_mapping") and self.reverse_target_mapping:
            predictions = np.array(
                [self.reverse_target_mapping.get(pred, pred) for pred in predictions]
            )

        return predictions, probabilities

    def explain_prediction(self, X, instance_idx=0):
        """
        Generate SHAP explanations for predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train_models() first.")

        # Preprocess input
        X_processed = self.preprocess_data(X)
        feature_columns = [
            col for col in X_processed.columns if col not in ["Loan_Status", "Loan_ID"]
        ]
        X_features = X_processed[feature_columns]

        # Generate SHAP values
        shap_values = self.shap_explainer.shap_values(
            X_features.iloc[instance_idx : instance_idx + 1]
        )

        return shap_values, feature_columns

    def plot_feature_importance(self, model_name="random_forest", top_n=10):
        """
        Plot feature importance
        """
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}")
            return

        # Get top N features
        importance_dict = self.feature_importance[model_name]
        top_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        # Plot
        features, importances = zip(*top_features)
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances)
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.tight_layout()
        plt.show()

    def save_models(self, filepath_prefix="loan_approval_model"):
        """
        Save trained models and preprocessing objects
        """
        model_artifacts = {
            "models": self.models,
            "scalers": self.scalers,
            "encoders": self.encoders,
            "feature_importance": self.feature_importance,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(model_artifacts, f"{filepath_prefix}_artifacts.pkl")
        print(f"Models saved to {filepath_prefix}_artifacts.pkl")

    def load_models(self, filepath="loan_approval_model_artifacts.pkl"):
        """
        Load trained models and preprocessing objects
        """
        model_artifacts = joblib.load(filepath)

        self.models = model_artifacts["models"]
        self.scalers = model_artifacts["scalers"]
        self.encoders = model_artifacts["encoders"]
        self.feature_importance = model_artifacts["feature_importance"]
        self.is_fitted = model_artifacts["is_fitted"]

        # Reinitialize SHAP explainer
        if "random_forest" in self.models:
            self.shap_explainer = shap.TreeExplainer(self.models["random_forest"])

        print("Models loaded successfully")


class ModelValidator:
    """
    Comprehensive model validation and testing
    """

    @staticmethod
    def validate_model_performance(model, X_test, y_test, threshold=0.8):
        """
        Validate model performance against thresholds
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)

        validation_results = {
            "roc_auc": roc_auc,
            "passes_threshold": roc_auc >= threshold,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        return validation_results

    @staticmethod
    def perform_stress_test(model, X_test, perturbation_factor=0.1):
        """
        Perform stress testing by adding noise to input data
        """
        original_pred = model.predict_proba(X_test)[:, 1]

        # Add noise to numerical features
        X_test_noisy = X_test.copy()
        numerical_cols = X_test.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            noise = np.random.normal(
                0, perturbation_factor * X_test[col].std(), size=len(X_test)
            )
            X_test_noisy[col] = X_test[col] + noise

        noisy_pred = model.predict_proba(X_test_noisy)[:, 1]

        # Calculate stability metric
        stability_score = 1 - np.mean(np.abs(original_pred - noisy_pred))

        return {
            "stability_score": stability_score,
            "mean_prediction_change": np.mean(np.abs(original_pred - noisy_pred)),
            "max_prediction_change": np.max(np.abs(original_pred - noisy_pred)),
        }


# Usage example
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("loan_approval_dataset.csv")

    # Initialize predictor
    predictor = LoanApprovalPredictor()

    # Train models
    results = predictor.train_models(df)

    # Make predictions
    sample_data = df.head(5).drop("Loan_Status", axis=1)
    predictions, probabilities = predictor.predict(sample_data)

    # Generate explanations
    shap_values, feature_names = predictor.explain_prediction(sample_data, 0)

    # Save models
    predictor.save_models()

    print("Model training and evaluation completed successfully!")
