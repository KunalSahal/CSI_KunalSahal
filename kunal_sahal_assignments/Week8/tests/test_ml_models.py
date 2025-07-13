"""
Unit tests for ML model implementation
Comprehensive testing suite for loan approval prediction models
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml_model_implementation import LoanApprovalPredictor, ModelValidator


class TestLoanApprovalPredictor:
    """Test cases for LoanApprovalPredictor class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample loan data for testing"""
        np.random.seed(42)
        n_samples = 100

        data = {
            "Gender": np.random.choice(["Male", "Female"], n_samples),
            "Married": np.random.choice(["Yes", "No"], n_samples),
            "Dependents": np.random.choice(["0", "1", "2", "3+"], n_samples),
            "Education": np.random.choice(["Graduate", "Not Graduate"], n_samples),
            "Self_Employed": np.random.choice(["Yes", "No"], n_samples),
            "ApplicantIncome": np.random.lognormal(8.5, 0.8, n_samples).astype(int),
            "CoapplicantIncome": np.random.lognormal(7.5, 1.2, n_samples).astype(int),
            "LoanAmount": np.random.lognormal(5.0, 0.5, n_samples).astype(int),
            "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], n_samples),
            "Credit_History": np.random.choice([0.0, 1.0], n_samples),
            "Property_Area": np.random.choice(
                ["Urban", "Semiurban", "Rural"], n_samples
            ),
            "Loan_Status": np.random.choice(["Y", "N"], n_samples),
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def predictor(self):
        """Create predictor instance for testing"""
        return LoanApprovalPredictor(random_state=42)

    def test_initialization(self, predictor):
        """Test predictor initialization"""
        assert predictor.random_state == 42
        assert predictor.models == {}
        assert predictor.scalers == {}
        assert predictor.encoders == {}
        assert predictor.feature_importance == {}
        assert predictor.shap_explainer is None
        assert predictor.is_fitted is False

    def test_preprocess_data(self, predictor, sample_data):
        """Test data preprocessing"""
        processed_data = predictor.preprocess_data(sample_data)

        # Check that preprocessing was applied
        assert "Income_to_Loan_Ratio" in processed_data.columns
        assert "Loan_to_Income_Ratio" in processed_data.columns
        assert "Total_Income" in processed_data.columns
        assert "Risk_Score" in processed_data.columns

        # Check that categorical features are encoded
        assert processed_data["Gender"].dtype in [np.int32, np.int64]
        assert processed_data["Married"].dtype in [np.int32, np.int64]

        # Check that no infinite values remain
        assert not np.isinf(processed_data).any().any()
        assert not np.isnan(processed_data).any().any()

    def test_preprocess_data_with_missing_values(self, predictor):
        """Test preprocessing with missing values"""
        data_with_missing = pd.DataFrame(
            {
                "Gender": ["Male", "Female", None, "Male"],
                "Married": ["Yes", None, "No", "Yes"],
                "Dependents": ["0", "1", None, "2"],
                "Education": ["Graduate", "Not Graduate", "Graduate", None],
                "Self_Employed": ["No", "Yes", None, "No"],
                "ApplicantIncome": [5000, 6000, None, 7000],
                "CoapplicantIncome": [2000, None, 3000, 2500],
                "LoanAmount": [150000, 200000, 180000, None],
                "Loan_Amount_Term": [360, None, 240, 300],
                "Credit_History": [1.0, 0.0, None, 1.0],
                "Property_Area": ["Urban", "Semiurban", None, "Rural"],
                "Loan_Status": ["Y", "N", "Y", "N"],
            }
        )

        processed_data = predictor.preprocess_data(data_with_missing)

        # Check that missing values are handled
        assert not processed_data.isnull().any().any()
        assert len(processed_data) == len(data_with_missing)

    @patch("ml_model_implementation.GridSearchCV")
    @patch("ml_model_implementation.cross_val_score")
    def test_train_models(
        self, mock_cv_score, mock_grid_search, predictor, sample_data
    ):
        """Test model training"""
        # Mock the grid search and cross validation
        mock_grid_search.return_value.best_estimator_ = Mock()
        mock_grid_search.return_value.best_params_ = {"n_estimators": 100}
        mock_cv_score.return_value = np.array([0.8, 0.85, 0.82, 0.87, 0.83])

        # Mock the ensemble model
        with patch("ml_model_implementation.VotingClassifier") as mock_voting:
            mock_voting.return_value.fit.return_value = Mock()
            mock_voting.return_value.predict.return_value = np.array([1, 0, 1])
            mock_voting.return_value.predict_proba.return_value = np.array(
                [[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]]
            )

            results = predictor.train_models(sample_data)

        # Check that models were trained
        assert predictor.is_fitted is True
        assert len(predictor.models) > 0
        assert "ensemble" in predictor.models

        # Check that results contain expected keys
        for model_name in ["random_forest", "xgboost", "gradient_boosting"]:
            assert model_name in results
            assert "roc_auc" in results[model_name]
            assert "cv_mean" in results[model_name]

    def test_predict_without_training(self, predictor, sample_data):
        """Test prediction without training raises error"""
        with pytest.raises(ValueError, match="Model not fitted yet"):
            predictor.predict(sample_data)

    def test_predict_after_training(self, predictor, sample_data):
        """Test prediction after training"""
        # Mock training
        with patch.object(predictor, "train_models") as mock_train:
            mock_train.return_value = {}
            predictor.train_models(sample_data)

        # Mock prediction
        with patch.object(predictor.models["ensemble"], "predict") as mock_predict:
            with patch.object(
                predictor.models["ensemble"], "predict_proba"
            ) as mock_proba:
                mock_predict.return_value = np.array([1, 0, 1])
                mock_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])

                predictions, probabilities = predictor.predict(sample_data)

        assert len(predictions) == len(sample_data)
        assert predictions.shape == (len(sample_data),)
        assert probabilities.shape == (len(sample_data), 2)

    def test_explain_prediction_without_training(self, predictor, sample_data):
        """Test SHAP explanation without training raises error"""
        with pytest.raises(ValueError, match="Model not fitted yet"):
            predictor.explain_prediction(sample_data)

    def test_save_and_load_models(self, predictor, sample_data, tmp_path):
        """Test model saving and loading"""
        # Mock training
        with patch.object(predictor, "train_models") as mock_train:
            mock_train.return_value = {}
            predictor.train_models(sample_data)

        # Test saving
        save_path = tmp_path / "test_model.pkl"
        predictor.save_models(str(save_path))
        assert save_path.exists()

        # Test loading
        new_predictor = LoanApprovalPredictor()
        new_predictor.load_models(str(save_path))

        assert new_predictor.is_fitted is True
        assert len(new_predictor.models) > 0


class TestModelValidator:
    """Test cases for ModelValidator class"""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.predict.return_value = np.array([1, 0, 1, 0, 1])
        model.predict_proba.return_value = np.array(
            [[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]]
        )
        return model

    @pytest.fixture
    def mock_data(self):
        """Create mock test data"""
        X_test = np.random.randn(5, 10)
        y_test = np.array([1, 0, 1, 0, 1])
        return X_test, y_test

    def test_validate_model_performance_above_threshold(self, mock_model, mock_data):
        """Test model validation with good performance"""
        X_test, y_test = mock_data

        # Mock metrics calculation
        with patch("ml_model_implementation.roc_auc_score") as mock_roc:
            mock_roc.return_value = 0.85

            result = ModelValidator.validate_model_performance(
                mock_model, X_test, y_test, threshold=0.8
            )

        assert result["passed"] is True
        assert result["roc_auc"] == 0.85

    def test_validate_model_performance_below_threshold(self, mock_model, mock_data):
        """Test model validation with poor performance"""
        X_test, y_test = mock_data

        # Mock metrics calculation
        with patch("ml_model_implementation.roc_auc_score") as mock_roc:
            mock_roc.return_value = 0.75

            result = ModelValidator.validate_model_performance(
                mock_model, X_test, y_test, threshold=0.8
            )

        assert result["passed"] is False
        assert result["roc_auc"] == 0.75

    def test_perform_stress_test(self, mock_model, mock_data):
        """Test stress testing functionality"""
        X_test, y_test = mock_data

        # Mock the model's predict method to return different results under stress
        original_predict = mock_model.predict

        def stressed_predict(X):
            # Simulate degraded performance under stress
            base_pred = original_predict(X)
            # Add some noise to simulate stress
            noise = np.random.choice([0, 1], size=base_pred.shape, p=[0.1, 0.9])
            return np.logical_xor(base_pred, noise).astype(int)

        mock_model.predict = stressed_predict

        result = ModelValidator.perform_stress_test(
            mock_model, X_test, perturbation_factor=0.1
        )

        assert "stability_score" in result
        assert "performance_degradation" in result
        assert 0 <= result["stability_score"] <= 1


class TestIntegration:
    """Integration tests for the complete ML pipeline"""

    @pytest.fixture
    def complete_pipeline(self):
        """Create complete pipeline for integration testing"""
        return LoanApprovalPredictor(random_state=42)

    def test_end_to_end_pipeline(self, complete_pipeline):
        """Test complete end-to-end pipeline"""
        # Create realistic test data
        np.random.seed(42)
        n_samples = 200

        # Generate realistic loan data
        data = pd.DataFrame(
            {
                "Gender": np.random.choice(["Male", "Female"], n_samples, p=[0.8, 0.2]),
                "Married": np.random.choice(["Yes", "No"], n_samples, p=[0.7, 0.3]),
                "Dependents": np.random.choice(
                    ["0", "1", "2", "3+"], n_samples, p=[0.6, 0.2, 0.15, 0.05]
                ),
                "Education": np.random.choice(
                    ["Graduate", "Not Graduate"], n_samples, p=[0.8, 0.2]
                ),
                "Self_Employed": np.random.choice(
                    ["Yes", "No"], n_samples, p=[0.15, 0.85]
                ),
                "ApplicantIncome": np.random.lognormal(8.5, 0.8, n_samples).astype(int),
                "CoapplicantIncome": np.random.lognormal(7.5, 1.2, n_samples).astype(
                    int
                ),
                "LoanAmount": np.random.lognormal(5.0, 0.5, n_samples).astype(int),
                "Loan_Amount_Term": np.random.choice(
                    [120, 180, 240, 300, 360, 480], n_samples
                ),
                "Credit_History": np.random.choice(
                    [0.0, 1.0], n_samples, p=[0.15, 0.85]
                ),
                "Property_Area": np.random.choice(
                    ["Urban", "Semiurban", "Rural"], n_samples
                ),
                "Loan_Status": np.random.choice(["Y", "N"], n_samples, p=[0.7, 0.3]),
            }
        )

        # Test complete pipeline
        try:
            # Train models
            results = complete_pipeline.train_models(data)

            # Verify training results
            assert complete_pipeline.is_fitted is True
            assert len(complete_pipeline.models) > 0
            assert "ensemble" in complete_pipeline.models

            # Test prediction
            test_data = data.head(10)
            predictions, probabilities = complete_pipeline.predict(test_data)

            # Verify predictions
            assert len(predictions) == 10
            assert predictions.shape == (10,)
            assert probabilities.shape == (10, 2)
            assert np.all((probabilities >= 0) & (probabilities <= 1))
            assert np.allclose(probabilities.sum(axis=1), 1.0)

            # Test feature importance
            if hasattr(complete_pipeline, "feature_importance"):
                assert len(complete_pipeline.feature_importance) > 0

            print(f"✅ Integration test passed! Model performance: {results}")

        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")


class TestDataQuality:
    """Tests for data quality and validation"""

    def test_data_consistency(self):
        """Test data consistency checks"""
        # Test valid data
        valid_data = pd.DataFrame(
            {
                "Gender": ["Male", "Female"],
                "Married": ["Yes", "No"],
                "ApplicantIncome": [5000, 6000],
                "Loan_Status": ["Y", "N"],
            }
        )

        # Test invalid data
        invalid_data = pd.DataFrame(
            {
                "Gender": ["Male", "Invalid"],
                "Married": ["Yes", "No"],
                "ApplicantIncome": [5000, -1000],  # Negative income
                "Loan_Status": ["Y", "Invalid"],
            }
        )

        # Add validation logic here
        assert len(valid_data) > 0
        assert valid_data["ApplicantIncome"].min() >= 0

    def test_feature_engineering_consistency(self):
        """Test that feature engineering produces consistent results"""
        predictor = LoanApprovalPredictor()

        # Create test data
        data1 = pd.DataFrame(
            {
                "ApplicantIncome": [5000, 6000],
                "CoapplicantIncome": [2000, 3000],
                "LoanAmount": [150000, 200000],
                "Loan_Amount_Term": [360, 240],
                "Credit_History": [1.0, 0.0],
                "Gender": ["Male", "Female"],
                "Married": ["Yes", "No"],
                "Dependents": ["0", "1"],
                "Education": ["Graduate", "Not Graduate"],
                "Self_Employed": ["No", "Yes"],
                "Property_Area": ["Urban", "Semiurban"],
                "Loan_Status": ["Y", "N"],
            }
        )

        # Process data twice
        processed1 = predictor.preprocess_data(data1)
        processed2 = predictor.preprocess_data(data1)

        # Results should be identical
        pd.testing.assert_frame_equal(processed1, processed2)


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=ml_model_implementation", "--cov-report=html"])
