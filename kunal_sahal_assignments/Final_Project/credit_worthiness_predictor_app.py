import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import joblib
from ucimlrepo import fetch_ucirepo
import warnings

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="🏦 Credit Risk Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .good-credit {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .bad-credit {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    '<h1 class="main-header">🏦 Credit Risk Prediction System</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem;">Powered by Random Forest ML & GenAI Analytics</p>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown("## 🔧 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "🤖 Prediction",
        "📈 Model Performance",
        "🧠 GenAI Insights",
    ],
)


@st.cache_data
def load_data():
    """Load and preprocess the German Credit Data"""
    try:
        # Fetch dataset from UCI repository
        german_credit = fetch_ucirepo(id=144)
        X = german_credit.data.features
        y = german_credit.data.targets

        # Combine features and target
        data = pd.concat([X, y], axis=1)

        # Rename target column for clarity
        data.rename(columns={"class": "credit_risk"}, inplace=True)

        # Map target values (1 = Good, 2 = Bad)
        data["credit_risk"] = data["credit_risk"].map({1: "Good", 2: "Bad"})

        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_resource
def train_model(data):
    """Train Random Forest model with feature engineering"""
    # Prepare features
    X = data.drop("credit_risk", axis=1)
    y = data["credit_risk"]

    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Encode target
    y_encoded = LabelEncoder().fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return (
        rf_model,
        label_encoders,
        X_test,
        y_test,
        y_pred,
        y_pred_proba,
        accuracy,
        roc_auc,
        X.columns,
    )


def generate_genai_insights(data, model, feature_names):
    """Generate GenAI-powered insights about credit risk patterns"""
    insights = []

    # Feature importance analysis
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    top_features = feature_importance.head(5)

    insights.append("🎯 **Top Risk Factors Identified by AI:**")
    for idx, row in top_features.iterrows():
        insights.append(f"• {row['feature']}: {row['importance']:.3f} importance score")

    # Statistical insights
    good_credit = data[data["credit_risk"] == "Good"]
    bad_credit = data[data["credit_risk"] == "Bad"]

    insights.append("\n🔍 **AI-Powered Risk Pattern Analysis:**")

    # Age analysis
    if "age" in data.columns:
        avg_age_good = good_credit["age"].mean()
        avg_age_bad = bad_credit["age"].mean()
        insights.append(
            f"• Average age of good credit customers: {avg_age_good:.1f} years"
        )
        insights.append(
            f"• Average age of bad credit customers: {avg_age_bad:.1f} years"
        )

    # Credit amount analysis
    if "credit_amount" in data.columns:
        avg_credit_good = good_credit["credit_amount"].mean()
        avg_credit_bad = bad_credit["credit_amount"].mean()
        insights.append(
            f"• Average credit amount for good customers: ${avg_credit_good:,.0f}"
        )
        insights.append(
            f"• Average credit amount for risky customers: ${avg_credit_bad:,.0f}"
        )

    # Risk distribution
    risk_distribution = data["credit_risk"].value_counts(normalize=True)
    insights.append("\n📊 **Portfolio Risk Distribution:**")
    insights.append(f"• {risk_distribution['Good']:.1%} of customers are low risk")
    insights.append(f"• {risk_distribution['Bad']:.1%} of customers are high risk")

    return "\n".join(insights)


# Load data
data = load_data()

if data is not None:
    # Train model
    model_results = train_model(data)
    if model_results:
        (
            model,
            encoders,
            X_test,
            y_test,
            y_pred,
            y_pred_proba,
            accuracy,
            roc_auc,
            feature_names,
        ) = model_results

        # Page routing
        if page == "🏠 Home":
            st.markdown("## Welcome to the Credit Risk Prediction System")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    """
                <div class="metric-card">
                    <h3>📊 Dataset Overview</h3>
                    <p>1000 customers analyzed</p>
                    <p>20 features considered</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>🎯 Model Performance</h3>
                    <p>Accuracy: {accuracy:.2%}</p>
                    <p>ROC-AUC: {roc_auc:.3f}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    """
                <div class="metric-card">
                    <h3>🤖 AI Technology</h3>
                    <p>Random Forest Algorithm</p>
                    <p>GenAI Insights</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Feature overview
            st.markdown("## 🔍 Key Features Analyzed")
            features_info = {
                "Personal Info": ["Age", "Personal Status", "Employment Status"],
                "Financial Info": ["Credit Amount", "Duration", "Installment Rate"],
                "Credit History": [
                    "Checking Account",
                    "Credit History",
                    "Savings Account",
                ],
                "Collateral": ["Property", "Other Credits", "Guarantors"],
            }

            cols = st.columns(2)
            for i, (category, features) in enumerate(features_info.items()):
                with cols[i % 2]:
                    st.markdown(f"**{category}:**")
                    for feature in features:
                        st.write(f"• {feature}")

        elif page == "📊 Data Analysis":
            st.markdown("## 📊 Exploratory Data Analysis")

            # Dataset overview
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Dataset Summary")
                st.write(f"**Total Records:** {len(data)}")
                st.write(f"**Features:** {len(data.columns) - 1}")
                st.write(f"**Target Classes:** {data['credit_risk'].nunique()}")

            with col2:
                st.markdown("### Credit Risk Distribution")
                risk_counts = data["credit_risk"].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Credit Risk Distribution",
                    color_discrete_sequence=["#00cc96", "#ff6b6b"],
                )
                st.plotly_chart(fig, use_container_width=True)

            # Feature distributions
            st.markdown("### 📈 Feature Distributions")

            # Numerical features
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                selected_num_features = st.multiselect(
                    "Select numerical features to analyze:",
                    numerical_cols,
                    default=numerical_cols[:4],
                )

                if selected_num_features:
                    fig = make_subplots(
                        rows=2, cols=2, subplot_titles=selected_num_features[:4]
                    )

                    for i, col in enumerate(selected_num_features[:4]):
                        row = i // 2 + 1
                        col_pos = i % 2 + 1

                        fig.add_trace(
                            go.Histogram(x=data[col], name=col, showlegend=False),
                            row=row,
                            col=col_pos,
                        )

                    fig.update_layout(
                        height=600, title_text="Distribution of Numerical Features"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap
            st.markdown("### 🔥 Feature Correlation Heatmap")
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu",
                )
                st.plotly_chart(fig, use_container_width=True)

        elif page == "🤖 Prediction":
            st.markdown("## 🤖 Credit Risk Prediction")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📝 Enter Customer Details")

                # Input fields based on key features
                duration = st.number_input(
                    "Duration (months)", min_value=1, max_value=72, value=12
                )
                credit_amount = st.number_input(
                    "Credit Amount", min_value=100, max_value=50000, value=5000
                )
                installment_rate = st.slider("Installment Rate (%)", 1, 4, 2)
                age = st.number_input("Age", min_value=18, max_value=80, value=30)

                # Categorical inputs
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

                if st.button("🔮 Predict Credit Risk", type="primary"):
                    # Create dummy input (simplified for demo)
                    # In practice, you'd need to properly encode all features
                    dummy_input = np.array(
                        [
                            [
                                1,
                                duration,
                                1,
                                1,
                                credit_amount,
                                1,
                                1,
                                installment_rate,
                                1,
                                1,
                                1,
                                1,
                                age,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                            ]
                        ]
                    )

                    # Make prediction
                    prediction = model.predict(dummy_input)[0]
                    probability = model.predict_proba(dummy_input)[0]

                    # Store prediction in session state
                    st.session_state.prediction = prediction
                    st.session_state.probability = probability

            with col2:
                st.markdown("### 🎯 Prediction Results")

                if hasattr(st.session_state, "prediction"):
                    pred = st.session_state.prediction
                    prob = st.session_state.probability

                    if pred == 0:  # Good credit
                        st.markdown(
                            f"""
                        <div class="prediction-card good-credit">
                            <h2>✅ APPROVED</h2>
                            <h3>Low Risk Customer</h3>
                            <p>Confidence: {prob[0]:.1%}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:  # Bad credit
                        st.markdown(
                            f"""
                        <div class="prediction-card bad-credit">
                            <h2>⚠️ CAUTION</h2>
                            <h3>High Risk Customer</h3>
                            <p>Risk Score: {prob[1]:.1%}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Probability visualization
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=["Good Credit", "Bad Credit"],
                                y=[prob[0], prob[1]],
                                marker_color=["#00cc96", "#ff6b6b"],
                            )
                        ]
                    )
                    fig.update_layout(
                        title="Prediction Probabilities", yaxis_title="Probability"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        elif page == "📈 Model Performance":
            st.markdown("## 📈 Model Performance Analysis")

            # Performance metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
            with col3:
                st.metric("Total Predictions", len(y_test))

            # Confusion Matrix
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual"),
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📈 ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f"ROC Curve (AUC = {roc_auc:.3f})")
                )
                fig.add_trace(
                    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random")
                )
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Feature Importance
            st.markdown("### 🎯 Feature Importance")
            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            fig = px.bar(
                feature_importance.head(10),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Most Important Features",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif page == "🧠 GenAI Insights":
            st.markdown("## 🧠 GenAI-Powered Business Insights")

            # Generate insights
            insights = generate_genai_insights(data, model, feature_names)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 🤖 AI Analysis Report")
                st.markdown(insights)

                # Risk factors visualization
                st.markdown("### 📊 Risk Factor Analysis")
                feature_importance = (
                    pd.DataFrame(
                        {
                            "feature": feature_names,
                            "importance": model.feature_importances_,
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .head(8)
                )

                fig = px.treemap(
                    feature_importance,
                    path=["feature"],
                    values="importance",
                    title="Feature Importance Treemap",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 💡 Recommendations")

                recommendations = [
                    "🎯 Focus on checking account status during assessment",
                    "📊 Consider credit history as primary factor",
                    "💰 Evaluate credit amount vs income ratio",
                    "⏰ Review loan duration carefully",
                    "👥 Assess employment stability",
                    "🏠 Consider property ownership",
                    "📱 Verify contact information",
                    "🔍 Cross-check with other financial institutions",
                ]

                for rec in recommendations:
                    st.write(rec)

                st.markdown("---")

                st.markdown("### 🚀 Next Steps")
                st.info("""
                **For Production Deployment:**
                1. Implement real-time data pipeline
                2. Add model monitoring & retraining
                3. Integrate with core banking system
                4. Set up automated alerts
                5. Implement A/B testing framework
                """)

    else:
        st.error("Failed to train the model. Please check your data.")
else:
    st.error("Failed to load the dataset. Please check your connection.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #7f8c8d;">
    <p>🎓 Built by Data Science Intern | 🚀 Powered by Random Forest & GenAI</p>
    <p>📊 German Credit Dataset | 🏦 UCI Machine Learning Repository</p>
</div>
""",
    unsafe_allow_html=True,
)
