import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report
import streamlit as st


def create_feature_mapping():
    """Create a mapping for feature names to more readable labels"""
    feature_mapping = {
        "checking_account": "Checking Account Status",
        "duration": "Duration (months)",
        "credit_history": "Credit History",
        "purpose": "Purpose",
        "credit_amount": "Credit Amount",
        "savings_account": "Savings Account",
        "employment": "Employment Status",
        "installment_rate": "Installment Rate %",
        "personal_status": "Personal Status",
        "other_parties": "Other Parties",
        "residence_since": "Residence Since",
        "property_magnitude": "Property",
        "age": "Age",
        "other_payment_plans": "Other Payment Plans",
        "housing": "Housing",
        "existing_credits": "Existing Credits",
        "job": "Job",
        "num_dependents": "Number of Dependents",
        "own_telephone": "Own Telephone",
        "foreign_worker": "Foreign Worker",
    }
    return feature_mapping


def create_advanced_visualizations(data, model, feature_names):
    """Create advanced visualizations for the dashboard"""

    # Feature importance pie chart
    feature_importance = (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig_pie = px.pie(
        feature_importance,
        values="importance",
        names="feature",
        title="Feature Importance Distribution (Top 10)",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    # Age distribution by credit risk
    if "age" in data.columns:
        fig_age = px.histogram(
            data,
            x="age",
            color="credit_risk",
            nbins=20,
            title="Age Distribution by Credit Risk",
            barmode="overlay",
            opacity=0.7,
        )
    else:
        fig_age = None

    # Credit amount vs Duration scatter plot
    if "credit_amount" in data.columns and "duration" in data.columns:
        fig_scatter = px.scatter(
            data,
            x="duration",
            y="credit_amount",
            color="credit_risk",
            title="Credit Amount vs Duration",
            hover_data=["age"] if "age" in data.columns else None,
            size_max=60,
        )
    else:
        fig_scatter = None

    return fig_pie, fig_age, fig_scatter


def generate_model_insights(model, feature_names, accuracy, roc_auc):
    """Generate detailed model insights"""

    # Feature importance analysis
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    insights = {
        "top_features": feature_importance.head(5),
        "model_performance": {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "performance_category": "Excellent"
            if roc_auc > 0.8
            else "Good"
            if roc_auc > 0.7
            else "Fair",
        },
        "feature_insights": {
            "most_important": feature_importance.iloc[0]["feature"],
            "least_important": feature_importance.iloc[-1]["feature"],
            "top_3_contribute": feature_importance.head(3)["importance"].sum(),
        },
    }

    return insights


def create_risk_assessment_summary(prediction_proba, input_data):
    """Create a comprehensive risk assessment summary"""

    risk_score = prediction_proba[1] * 100  # Convert to percentage

    # Risk categories
    if risk_score < 30:
        risk_category = "Low Risk"
        risk_color = "#00cc96"
        recommendation = "Approve loan with standard terms"
    elif risk_score < 60:
        risk_category = "Medium Risk"
        risk_color = "#ffa500"
        recommendation = "Consider approval with additional conditions"
    else:
        risk_category = "High Risk"
        risk_color = "#ff6b6b"
        recommendation = "Reject or require additional collateral"

    # Risk factors analysis
    risk_factors = []
    if "credit_amount" in input_data and input_data["credit_amount"] > 10000:
        risk_factors.append("High credit amount requested")
    if "duration" in input_data and input_data["duration"] > 36:
        risk_factors.append("Long loan duration")
    if "age" in input_data and input_data["age"] < 25:
        risk_factors.append("Young age category")

    summary = {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "risk_factors": risk_factors,
    }

    return summary


def create_business_metrics_dashboard(data):
    """Create business-focused metrics and KPIs"""

    total_customers = len(data)
    good_credit_count = len(data[data["credit_risk"] == "Good"])
    bad_credit_count = len(data[data["credit_risk"] == "Bad"])

    # Calculate average credit amounts
    avg_credit_good = (
        data[data["credit_risk"] == "Good"]["credit_amount"].mean()
        if "credit_amount" in data.columns
        else 0
    )
    avg_credit_bad = (
        data[data["credit_risk"] == "Bad"]["credit_amount"].mean()
        if "credit_amount" in data.columns
        else 0
    )

    # Calculate approval rate
    approval_rate = (good_credit_count / total_customers) * 100

    # Age demographics
    if "age" in data.columns:
        avg_age_good = data[data["credit_risk"] == "Good"]["age"].mean()
        avg_age_bad = data[data["credit_risk"] == "Bad"]["age"].mean()
    else:
        avg_age_good = avg_age_bad = 0

    metrics = {
        "total_customers": total_customers,
        "good_credit_count": good_credit_count,
        "bad_credit_count": bad_credit_count,
        "approval_rate": approval_rate,
        "avg_credit_good": avg_credit_good,
        "avg_credit_bad": avg_credit_bad,
        "avg_age_good": avg_age_good,
        "avg_age_bad": avg_age_bad,
    }

    return metrics


def create_genai_recommendations(model_insights, business_metrics):
    """Generate GenAI-powered recommendations"""

    recommendations = []

    # Model performance recommendations
    if model_insights["model_performance"]["roc_auc"] > 0.8:
        recommendations.append(
            "🎯 Model shows excellent performance - ready for production deployment"
        )
    else:
        recommendations.append(
            "⚠️ Consider additional feature engineering or model tuning"
        )

    # Business recommendations
    if business_metrics["approval_rate"] < 60:
        recommendations.append("📊 Low approval rate detected - review credit criteria")
    elif business_metrics["approval_rate"] > 80:
        recommendations.append(
            "🚨 High approval rate - consider tightening credit standards"
        )

    # Feature-based recommendations
    top_feature = model_insights["feature_insights"]["most_important"]
    recommendations.append(f"🔍 Focus on '{top_feature}' during credit assessment")

    # Risk-based recommendations
    if business_metrics["avg_credit_bad"] > business_metrics["avg_credit_good"]:
        recommendations.append(
            "💰 High-risk customers tend to request larger loans - implement amount limits"
        )

    # Age-based recommendations
    if business_metrics["avg_age_bad"] < business_metrics["avg_age_good"]:
        recommendations.append(
            "👥 Younger customers show higher risk - consider age-based policies"
        )

    return recommendations


def format_currency(amount):
    """Format currency values for display"""
    return f"${amount:,.0f}"


def calculate_profit_loss_matrix(y_true, y_pred):
    """Calculate profit/loss based on the German Credit cost matrix"""

    # Cost matrix: actual vs predicted
    # True Negative (predict good, actual good): 0 cost
    # False Positive (predict good, actual bad): 5 cost
    # False Negative (predict bad, actual good): 1 cost
    # True Positive (predict bad, actual bad): 0 cost

    tn = np.sum((y_pred == 0) & (y_true == 0))  # Predict good, actual good
    fp = np.sum((y_pred == 0) & (y_true == 1))  # Predict good, actual bad
    fn = np.sum((y_pred == 1) & (y_true == 0))  # Predict bad, actual good
    tp = np.sum((y_pred == 1) & (y_true == 1))  # Predict bad, actual bad

    total_cost = fp * 5 + fn * 1

    cost_matrix = {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "total_cost": total_cost,
        "average_cost": total_cost / len(y_true),
    }

    return cost_matrix


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)

    sample_data = {
        "duration": np.random.randint(6, 72, 100),
        "credit_amount": np.random.randint(250, 18424, 100),
        "installment_rate": np.random.randint(1, 5, 100),
        "age": np.random.randint(19, 75, 100),
        "credit_risk": np.random.choice(["Good", "Bad"], 100, p=[0.7, 0.3]),
    }

    return pd.DataFrame(sample_data)


def create_interactive_prediction_form():
    """Create an interactive form for predictions"""

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input(
                "Duration (months)", min_value=1, max_value=72, value=12
            )
            credit_amount = st.number_input(
                "Credit Amount ($)", min_value=100, max_value=50000, value=5000
            )
            installment_rate = st.slider("Installment Rate (%)", 1, 4, 2)
            age = st.number_input("Age", min_value=18, max_value=80, value=30)

        with col2:
            checking_account = st.selectbox(
                "Checking Account Status",
                ["< 0 DM", "0-200 DM", "> 200 DM", "No account"],
            )
            credit_history = st.selectbox(
                "Credit History",
                ["No credits", "All paid", "Existing paid", "Delayed", "Critical"],
            )
            purpose = st.selectbox(
                "Purpose",
                [
                    "Car (new)",
                    "Car (used)",
                    "Furniture",
                    "TV/Radio",
                    "Appliances",
                    "Education",
                    "Business",
                    "Other",
                ],
            )
            employment = st.selectbox(
                "Employment Status",
                ["Unemployed", "< 1 year", "1-4 years", "4-7 years", "> 7 years"],
            )

        submitted = st.form_submit_button("🔮 Predict Credit Risk", type="primary")

        if submitted:
            input_data = {
                "duration": duration,
                "credit_amount": credit_amount,
                "installment_rate": installment_rate,
                "age": age,
                "checking_account": checking_account,
                "credit_history": credit_history,
                "purpose": purpose,
                "employment": employment,
            }
            return input_data

    return None
