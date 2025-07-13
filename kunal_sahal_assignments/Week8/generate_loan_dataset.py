import pandas as pd
import numpy as np
from datetime import datetime


def generate_loan_dataset(n_samples=1000, save_to_file=True):
    """
    Generate a comprehensive loan approval dataset

    Parameters:
    n_samples (int): Number of records to generate
    save_to_file (bool): Whether to save to CSV file

    Returns:
    pd.DataFrame: Generated loan dataset
    """

    np.random.seed(42)  # For reproducible results

    # Generate realistic loan data
    data = {
        "Loan_ID": [f"LP{i:06d}" for i in range(1, n_samples + 1)],
        "Gender": np.random.choice(["Male", "Female"], n_samples, p=[0.65, 0.35]),
        "Married": np.random.choice(["Yes", "No"], n_samples, p=[0.7, 0.3]),
        "Dependents": np.random.choice(
            ["0", "1", "2", "3+"], n_samples, p=[0.4, 0.3, 0.2, 0.1]
        ),
        "Education": np.random.choice(
            ["Graduate", "Not Graduate"], n_samples, p=[0.75, 0.25]
        ),
        "Self_Employed": np.random.choice(["Yes", "No"], n_samples, p=[0.15, 0.85]),
        "ApplicantIncome": np.random.lognormal(10.2, 0.6, n_samples).astype(int),
        "CoapplicantIncome": np.random.lognormal(9.0, 0.7, n_samples).astype(int),
        "LoanAmount": np.random.lognormal(11.3, 0.5, n_samples).astype(int),
        "Loan_Amount_Term": np.random.choice(
            [12, 36, 60, 84, 120, 180, 240, 300, 360],
            n_samples,
            p=[0.05, 0.1, 0.15, 0.1, 0.2, 0.15, 0.1, 0.1, 0.05],
        ),
        "Credit_History": np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
        "Property_Area": np.random.choice(
            ["Urban", "Semiurban", "Rural"], n_samples, p=[0.4, 0.35, 0.25]
        ),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create loan status based on realistic criteria
    approval_conditions = (
        (df["ApplicantIncome"] > 4000)  # Minimum income requirement
        & (df["Credit_History"] == 1)  # Good credit history
        & (
            df["LoanAmount"] < (df["ApplicantIncome"] + df["CoapplicantIncome"]) * 0.6
        )  # Debt-to-income ratio
        & (df["Education"] == "Graduate")  # Education preference
        & (df["Loan_Amount_Term"] >= 60)  # Minimum loan term
    )

    # Additional favorable conditions
    favorable_conditions = (
        (df["Property_Area"] == "Urban")  # Urban preference
        | (df["ApplicantIncome"] > 8000)  # High income
        | (df["Married"] == "Yes")  # Married applicants
    )

    # Calculate approval probability
    base_prob = np.where(approval_conditions, 0.85, 0.25)
    bonus_prob = np.where(favorable_conditions, 0.1, 0)
    final_prob = np.clip(base_prob + bonus_prob, 0.1, 0.95)

    # Generate loan status
    df["Loan_Status"] = np.random.binomial(1, final_prob, n_samples)
    df["Loan_Status"] = df["Loan_Status"].map({1: "Y", 0: "N"})

    # Add some noise to make it more realistic
    noise_mask = np.random.random(n_samples) < 0.05
    df.loc[noise_mask, "Loan_Status"] = np.random.choice(["Y", "N"], noise_mask.sum())

    # Add some missing values for realism
    missing_mask = np.random.random(n_samples) < 0.02
    df.loc[missing_mask, "Credit_History"] = np.nan

    # Reorder columns for better readability
    column_order = [
        "Loan_ID",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        "Loan_Status",
    ]

    df = df[column_order]

    # Save to file if requested
    if save_to_file:
        filename = f"loan_dataset_{n_samples}_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Dataset saved as: {filename}")
        print(f"Dataset shape: {df.shape}")
        print(f"Approval rate: {(df['Loan_Status'] == 'Y').mean() * 100:.1f}%")

    return df


if __name__ == "__main__":
    # Generate different sized datasets
    print("Generating loan datasets...")

    # Small dataset
    small_df = generate_loan_dataset(100, True)

    # Medium dataset
    medium_df = generate_loan_dataset(500, True)

    # Large dataset
    large_df = generate_loan_dataset(1000, True)

    print("\nDataset generation completed!")
    print("Files created:")
    print("- loan_dataset_100_records_*.csv")
    print("- loan_dataset_500_records_*.csv")
    print("- loan_dataset_1000_records_*.csv")

    # Show sample data
    print("\nSample data from large dataset:")
    print(large_df.head())

    print("\nDataset statistics:")
    print(f"Total records: {len(large_df)}")
    print(f"Approval rate: {(large_df['Loan_Status'] == 'Y').mean() * 100:.1f}%")
    print(f"Average income: ₹{large_df['ApplicantIncome'].mean():.0f}")
    print(f"Average loan amount: ₹{large_df['LoanAmount'].mean():.0f}")
