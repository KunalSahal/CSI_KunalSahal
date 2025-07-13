#!/usr/bin/env python3
"""
Simplified Setup script for Intelligent Loan Approval Assistant
This script initializes the project structure and handles basic setup.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProjectSetup:
    """Handle project setup and initialization"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"

    def create_directories(self):
        """Create necessary project directories"""
        directories = [
            self.data_dir,
            self.data_dir / "processed",
            self.models_dir,
            self.models_dir / "saved_models",
            self.models_dir / "vector_store",
            self.logs_dir,
            self.project_root / "outputs",
            self.project_root / "outputs" / "visualizations",
            self.project_root / "outputs" / "reports",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def check_python_version(self):
        """Check if Python version is compatible"""
        python_version = sys.version_info
        if python_version.major < 3 or (
            python_version.major == 3 and python_version.minor < 8
        ):
            logger.error("Python 3.8 or higher is required")
            return False
        logger.info(
            f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
        return True

    def install_dependencies(self):
        """Install required packages with error handling"""
        logger.info("Checking dependencies...")

        # Check if requirements.txt exists
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            logger.warning("requirements.txt not found")
            return False

        try:
            logger.info("Installing dependencies (this may take a few minutes)...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.warning(
                    f"⚠️ Some dependencies failed to install: {result.stderr}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Dependency installation timed out")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Error installing dependencies: {e}")
            return False

    def create_sample_data(self):
        """Create sample data if the main dataset is not available"""
        sample_data_path = self.data_dir / "sample_data.csv"

        if not sample_data_path.exists():
            logger.info("Creating sample data...")

            # Generate realistic sample data
            np.random.seed(42)
            n_samples = 1000

            # Define realistic distributions
            genders = np.random.choice(["Male", "Female"], n_samples, p=[0.8, 0.2])
            married = np.random.choice(["Yes", "No"], n_samples, p=[0.7, 0.3])
            dependents = np.random.choice(
                ["0", "1", "2", "3+"], n_samples, p=[0.6, 0.2, 0.15, 0.05]
            )
            education = np.random.choice(
                ["Graduate", "Not Graduate"], n_samples, p=[0.8, 0.2]
            )
            self_employed = np.random.choice(["Yes", "No"], n_samples, p=[0.15, 0.85])

            # Income distributions (log-normal)
            applicant_income = np.random.lognormal(8.5, 0.8, n_samples).astype(int)
            coapplicant_income = np.random.lognormal(7.5, 1.2, n_samples).astype(int)
            coapplicant_income[
                np.random.choice(n_samples, int(0.3 * n_samples), replace=False)
            ] = 0

            # Loan amounts
            loan_amounts = np.random.lognormal(5.0, 0.5, n_samples).astype(int)
            loan_terms = np.random.choice(
                [120, 180, 240, 300, 360, 480],
                n_samples,
                p=[0.1, 0.1, 0.15, 0.15, 0.4, 0.1],
            )

            # Credit history
            credit_history = np.random.choice([0.0, 1.0], n_samples, p=[0.15, 0.85])

            # Property area
            property_area = np.random.choice(
                ["Urban", "Semiurban", "Rural"], n_samples, p=[0.4, 0.4, 0.2]
            )

            # Generate loan status based on realistic criteria
            loan_status = []
            for i in range(n_samples):
                # Complex approval logic
                score = 0

                # Income factor
                total_income = applicant_income[i] + coapplicant_income[i]
                if total_income > 10000:
                    score += 3
                elif total_income > 5000:
                    score += 2
                elif total_income > 2000:
                    score += 1

                # Credit history is most important
                if credit_history[i] == 1.0:
                    score += 4

                # Education
                if education[i] == "Graduate":
                    score += 1

                # Marriage
                if married[i] == "Yes":
                    score += 1

                # Dependents (fewer is better)
                if dependents[i] == "0":
                    score += 1

                # Property area
                if property_area[i] == "Urban":
                    score += 1

                # Loan amount to income ratio
                if loan_amounts[i] < total_income * 0.3:
                    score += 2
                elif loan_amounts[i] < total_income * 0.5:
                    score += 1

                # Add some randomness
                if np.random.random() < 0.1:
                    score += np.random.choice([-2, -1, 1, 2])

                # Final decision
                loan_status.append("Y" if score >= 6 else "N")

            # Create DataFrame
            sample_df = pd.DataFrame(
                {
                    "Loan_ID": [
                        f"LP{str(i).zfill(6)}" for i in range(1, n_samples + 1)
                    ],
                    "Gender": genders,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amounts,
                    "Loan_Amount_Term": loan_terms,
                    "Credit_History": credit_history,
                    "Property_Area": property_area,
                    "Loan_Status": loan_status,
                }
            )

            # Save sample data
            sample_df.to_csv(sample_data_path, index=False)
            logger.info(f"✅ Sample data created: {sample_data_path}")

            # Create data summary
            summary = {
                "total_samples": len(sample_df),
                "approval_rate": (sample_df["Loan_Status"] == "Y").mean(),
                "features": list(sample_df.columns),
                "created_at": datetime.now().isoformat(),
            }

            with open(self.data_dir / "data_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"📊 Data summary: {summary}")
        else:
            logger.info("✅ Sample data already exists")

    def create_config_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")

        # Create .env.example
        env_example = """# Environment Variables
# Copy this file to .env and fill in your values

# OpenAI API Key (optional)
OPENAI_API_KEY=your_openai_key_here

# Hugging Face API Key (optional)
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Model Configuration
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# System Configuration
LOG_LEVEL=INFO
MAX_CONVERSATION_HISTORY=10
VECTOR_STORE_PATH=models/vector_store
MODEL_SAVE_PATH=models/saved_models

# API Configuration
API_HOST=localhost
API_PORT=8000
API_RELOAD=True
"""

        with open(self.project_root / ".env.example", "w") as f:
            f.write(env_example)

        # Create model config
        model_config = {
            "models": {
                "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                },
                "logistic_regression": {"C": 1.0, "max_iter": 1000},
            },
            "preprocessing": {
                "handle_missing": True,
                "encode_categorical": True,
                "scale_numerical": True,
            },
            "evaluation": {"test_size": 0.2, "cv_folds": 5, "random_state": 42},
        }

        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)

        with open(config_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        logger.info("✅ Configuration files created")

    def test_imports(self):
        """Test if key modules can be imported"""
        logger.info("Testing module imports...")

        modules_to_test = [
            ("pandas", "pd"),
            ("numpy", "np"),
            ("sklearn", "sklearn"),
            ("streamlit", "streamlit"),
            ("plotly", "plotly"),
        ]

        successful_imports = []
        failed_imports = []

        for module_name, import_name in modules_to_test:
            try:
                __import__(module_name)
                successful_imports.append(module_name)
                logger.info(f"✅ {module_name} imported successfully")
            except ImportError:
                failed_imports.append(module_name)
                logger.warning(f"⚠️ {module_name} not available")

        if successful_imports:
            logger.info(f"✅ Successfully imported: {', '.join(successful_imports)}")

        if failed_imports:
            logger.warning(f"⚠️ Missing modules: {', '.join(failed_imports)}")
            logger.info(
                "You can install missing modules manually using: pip install <module_name>"
            )

        return len(failed_imports) == 0

    def run_setup(self):
        """Run the complete setup process"""
        logger.info("🚀 Starting Intelligent Loan Approval Assistant Setup...")
        logger.info("=" * 60)

        # Check Python version
        if not self.check_python_version():
            logger.error("❌ Setup failed: Python version incompatible")
            return False

        # Create directories
        self.create_directories()

        # Create configuration files
        self.create_config_files()

        # Install dependencies
        deps_installed = self.install_dependencies()

        # Create sample data
        self.create_sample_data()

        # Test imports
        imports_ok = self.test_imports()

        logger.info("=" * 60)
        logger.info("🎉 Setup completed!")

        if deps_installed and imports_ok:
            logger.info("✅ All components ready!")
        else:
            logger.info("⚠️ Some components may need manual installation")

        logger.info("\n📋 Next steps:")
        logger.info("1. Copy .env.example to .env and add your API keys (optional)")
        logger.info("2. Run 'streamlit run app.py' to start the web interface")
        logger.info("3. Or run 'python ml_model_implementation.py' to train models")
        logger.info("4. Check the README.md for detailed usage instructions")

        logger.info("\n🔗 Quick Start:")
        logger.info("streamlit run app.py")

        return True


if __name__ == "__main__":
    setup = ProjectSetup()
    success = setup.run_setup()

    if success:
        print("\n🎉 Setup completed successfully! You can now run the application.")
    else:
        print("\n⚠️ Setup completed with warnings. Check the logs above for details.")
