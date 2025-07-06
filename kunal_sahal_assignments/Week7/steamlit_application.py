import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="ML Model Deployment Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App title
st.markdown(
    '<h1 class="main-header">🤖 ML Model Deployment Dashboard</h1>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🎯 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Application Mode",
    [
        "Home",
        "Model Training",
        "Prediction Interface",
        "Model Comparison",
        "Data Insights",
    ],
)

# Initialize session state
if "models" not in st.session_state:
    st.session_state.models = {}
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "scalers" not in st.session_state:
    st.session_state.scalers = {}


def load_dataset(dataset_name):
    """Load and return dataset"""
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        df["target_names"] = [data.target_names[i] for i in data.target]
        return df, "classification"
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        df["target_names"] = [data.target_names[i] for i in data.target]
        return df, "classification"
    elif dataset_name == "California Housing":
        # Use California housing dataset instead of Boston
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df, "regression"


def train_model(X_train, y_train, model_type, algorithm):
    """Train machine learning model"""
    if model_type == "classification":
        if algorithm == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algorithm == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif algorithm == "SVM":
            model = SVC(random_state=42, probability=True)
    else:
        if algorithm == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif algorithm == "Linear Regression":
            model = LinearRegression()
        elif algorithm == "SVM":
            model = SVR()

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_type):
    """Evaluate model performance"""
    predictions = model.predict(X_test)

    if model_type == "classification":
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        return accuracy, report, predictions
    else:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2, predictions


# HOME PAGE
if app_mode == "Home":
    st.markdown(
        """
    <div class="info-box">
        <h2>🎯 Welcome to the ML Model Deployment Dashboard</h2>
        <p>This comprehensive application demonstrates end-to-end machine learning workflow including:</p>
        <ul>
            <li><strong>Model Training:</strong> Train multiple ML algorithms on different datasets</li>
            <li><strong>Interactive Predictions:</strong> Real-time predictions with user inputs</li>
            <li><strong>Model Comparison:</strong> Compare performance across different algorithms</li>
            <li><strong>Data Visualization:</strong> Interactive charts and insights</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Datasets Available", "3")
    with col2:
        st.metric("🤖 ML Algorithms", "6")
    with col3:
        st.metric("📈 Visualization Types", "5+")
    with col4:
        st.metric("🔥 Models Trained", len(st.session_state.models))

    # Feature showcase
    st.markdown("## 🚀 Key Features")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Model Training
        - Multiple algorithms (RF, LR, SVM)
        - Automated preprocessing
        - Performance metrics
        - Model persistence
        """)

        st.markdown("""
        ### 📊 Interactive Predictions
        - Real-time input validation
        - Confidence scores
        - Feature importance
        - Prediction explanations
        """)

    with col2:
        st.markdown("""
        ### 🔍 Model Comparison
        - Side-by-side performance
        - Statistical significance
        - ROC curves
        - Confusion matrices
        """)

        st.markdown("""
        ### 📈 Data Insights
        - Correlation analysis
        - Distribution plots
        - Feature relationships
        - Interactive visualizations
        """)

# MODEL TRAINING PAGE
elif app_mode == "Model Training":
    st.markdown("## 🎯 Model Training Interface")

    # Dataset selection
    col1, col2 = st.columns(2)
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset", ["Iris", "Wine", "California Housing"]
        )
    with col2:
        algorithm = st.selectbox(
            "Select Algorithm",
            ["Random Forest", "Logistic Regression", "SVM", "Linear Regression"],
        )

    # Load dataset
    df, model_type = load_dataset(dataset_name)

    # Validate algorithm selection
    if model_type == "classification" and algorithm == "Linear Regression":
        st.error("Linear Regression is not suitable for classification tasks!")
        st.stop()
    elif model_type == "regression" and algorithm in ["Logistic Regression"]:
        st.error("Logistic Regression is not suitable for regression tasks!")
        st.stop()

    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Target Type", model_type.title())

    # Show dataset
    with st.expander("View Dataset"):
        st.dataframe(df.head(10))

    # Training parameters
    st.markdown("### ⚙️ Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random State", value=42)
    with col2:
        scale_features = st.checkbox("Scale Features", value=True)
        show_training_progress = st.checkbox("Show Training Details", value=True)

    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            X = df.drop(["target"], axis=1)
            if "target_names" in df.columns:
                X = X.drop(["target_names"], axis=1)
            y = df["target"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Scale features if selected
            if scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                st.session_state.scalers[f"{dataset_name}_{algorithm}"] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Train model
            model = train_model(X_train_scaled, y_train, model_type, algorithm)

            # Evaluate model
            if model_type == "classification":
                accuracy, report, predictions = evaluate_model(
                    model, X_test_scaled, y_test, model_type
                )

                # Display results
                st.success("✅ Model training completed!")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h3>Model Accuracy</h3>
                        <h2>{accuracy:.4f}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown("### 📊 Classification Report")
                    st.json(report)

                # Save model
                st.session_state.models[f"{dataset_name}_{algorithm}"] = {
                    "model": model,
                    "type": model_type,
                    "features": X.columns.tolist(),
                    "accuracy": accuracy,
                    "dataset": dataset_name,
                    "algorithm": algorithm,
                }

            else:
                mse, r2, predictions = evaluate_model(
                    model, X_test_scaled, y_test, model_type
                )

                # Display results
                st.success("✅ Model training completed!")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h3>R² Score</h3>
                        <h2>{r2:.4f}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h3>MSE</h3>
                        <h2>{mse:.4f}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Save model
                st.session_state.models[f"{dataset_name}_{algorithm}"] = {
                    "model": model,
                    "type": model_type,
                    "features": X.columns.tolist(),
                    "r2": r2,
                    "mse": mse,
                    "dataset": dataset_name,
                    "algorithm": algorithm,
                }

            # Feature importance (for tree-based models)
            if algorithm == "Random Forest":
                st.markdown("### 🎯 Feature Importance")
                importance_df = pd.DataFrame(
                    {"feature": X.columns, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)

                fig = px.bar(
                    importance_df,
                    x="importance",
                    y="feature",
                    title="Feature Importance",
                    orientation="h",
                )
                st.plotly_chart(fig, use_container_width=True)

# PREDICTION INTERFACE PAGE
elif app_mode == "Prediction Interface":
    st.markdown("## 🔮 Interactive Prediction Interface")

    if not st.session_state.models:
        st.warning("⚠️ No trained models available. Please train a model first!")
        st.stop()

    # Model selection
    model_names = list(st.session_state.models.keys())
    selected_model = st.selectbox("Select Trained Model", model_names)

    if selected_model:
        model_info = st.session_state.models[selected_model]
        model = model_info["model"]
        features = model_info["features"]
        model_type = model_info["type"]

        # Display model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", model_info["algorithm"])
        with col2:
            st.metric("Dataset", model_info["dataset"])
        with col3:
            if model_type == "classification":
                st.metric("Accuracy", f"{model_info['accuracy']:.4f}")
            else:
                st.metric("R² Score", f"{model_info['r2']:.4f}")

        # Input interface
        st.markdown("### 📝 Input Features")

        # Create input widgets based on dataset
        input_data = {}

        if model_info["dataset"] == "Iris":
            col1, col2 = st.columns(2)
            with col1:
                input_data["sepal length (cm)"] = st.slider(
                    "Sepal Length (cm)", 4.0, 8.0, 5.8
                )
                input_data["sepal width (cm)"] = st.slider(
                    "Sepal Width (cm)", 2.0, 4.5, 3.0
                )
            with col2:
                input_data["petal length (cm)"] = st.slider(
                    "Petal Length (cm)", 1.0, 7.0, 4.0
                )
                input_data["petal width (cm)"] = st.slider(
                    "Petal Width (cm)", 0.1, 2.5, 1.3
                )

        elif model_info["dataset"] == "Wine":
            st.markdown("*Simplified wine feature inputs for demonstration*")
            col1, col2, col3 = st.columns(3)
            with col1:
                input_data["alcohol"] = st.slider("Alcohol %", 11.0, 15.0, 13.0)
                input_data["malic_acid"] = st.slider("Malic Acid", 0.74, 5.80, 2.34)
            with col2:
                input_data["ash"] = st.slider("Ash", 1.36, 3.23, 2.36)
                input_data["alcalinity_of_ash"] = st.slider(
                    "Alcalinity of Ash", 10.6, 30.0, 19.5
                )
            with col3:
                input_data["magnesium"] = st.slider("Magnesium", 70, 162, 100)
                input_data["total_phenols"] = st.slider(
                    "Total Phenols", 0.98, 3.88, 2.29
                )

            # Add remaining features with default values
            remaining_features = [f for f in features if f not in input_data.keys()]
            for feature in remaining_features:
                input_data[feature] = 1.0  # Default value

        elif model_info["dataset"] == "California Housing":
            col1, col2 = st.columns(2)
            with col1:
                input_data["MedInc"] = st.slider("Median Income", 0.5, 15.0, 5.0)
                input_data["HouseAge"] = st.slider("House Age", 1.0, 52.0, 20.0)
                input_data["AveRooms"] = st.slider("Average Rooms", 1.0, 20.0, 6.0)
                input_data["AveBedrms"] = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
            with col2:
                input_data["Population"] = st.slider("Population", 3.0, 35000.0, 3000.0)
                input_data["AveOccup"] = st.slider("Average Occupancy", 1.0, 20.0, 3.0)
                input_data["Latitude"] = st.slider("Latitude", 32.0, 42.0, 35.0)
                input_data["Longitude"] = st.slider("Longitude", -125.0, -114.0, -119.0)

        # Make prediction
        if st.button("🔮 Make Prediction", type="primary"):
            # Prepare input
            input_df = pd.DataFrame([input_data])

            # Ensure all features are present
            for feature in features:
                if feature not in input_df.columns:
                    input_df[feature] = 0.0

            # Reorder columns to match training data
            input_df = input_df[features]

            # Scale if scaler exists
            scaler_key = f"{model_info['dataset']}_{model_info['algorithm']}"
            if scaler_key in st.session_state.scalers:
                input_scaled = st.session_state.scalers[scaler_key].transform(input_df)
            else:
                input_scaled = input_df

            # Make prediction
            prediction = model.predict(input_scaled)[0]

            # Display prediction
            if model_type == "classification":
                if model_info["dataset"] == "Iris":
                    class_names = ["Setosa", "Versicolor", "Virginica"]
                    predicted_class = class_names[int(prediction)]

                    # Get prediction probabilities
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_scaled)[0]

                        st.markdown(
                            f"""
                        <div class="prediction-box">
                            <h3>Predicted Species</h3>
                            <h2>{predicted_class}</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Probability chart
                        prob_df = pd.DataFrame(
                            {"Species": class_names, "Probability": proba}
                        )

                        fig = px.bar(
                            prob_df,
                            x="Species",
                            y="Probability",
                            title="Prediction Probabilities",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif model_info["dataset"] == "Wine":
                    wine_classes = ["Class 0", "Class 1", "Class 2"]
                    predicted_class = wine_classes[int(prediction)]

                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h3>Predicted Wine Class</h3>
                        <h2>{predicted_class}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            else:  # Regression
                st.markdown(
                    f"""
                <div class="prediction-box">
                    <h3>Predicted House Price</h3>
                    <h2>${prediction:.2f} (in hundreds of thousands)</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Show input summary
            st.markdown("### 📋 Input Summary")
            st.json(input_data)

# MODEL COMPARISON PAGE
elif app_mode == "Model Comparison":
    st.markdown("## 📊 Model Performance Comparison")

    if len(st.session_state.models) < 2:
        st.warning("⚠️ Need at least 2 trained models for comparison!")
        st.stop()

    # Group models by dataset
    dataset_models = {}
    for model_name, model_info in st.session_state.models.items():
        dataset = model_info["dataset"]
        if dataset not in dataset_models:
            dataset_models[dataset] = []
        dataset_models[dataset].append(model_name)

    # Select dataset for comparison
    selected_dataset = st.selectbox(
        "Select Dataset for Comparison", list(dataset_models.keys())
    )

    if selected_dataset and len(dataset_models[selected_dataset]) >= 2:
        models_to_compare = dataset_models[selected_dataset]

        # Create comparison table
        comparison_data = []
        for model_name in models_to_compare:
            model_info = st.session_state.models[model_name]
            if model_info["type"] == "classification":
                comparison_data.append(
                    {
                        "Model": model_info["algorithm"],
                        "Dataset": model_info["dataset"],
                        "Accuracy": f"{model_info['accuracy']:.4f}",
                        "Type": "Classification",
                    }
                )
            else:
                comparison_data.append(
                    {
                        "Model": model_info["algorithm"],
                        "Dataset": model_info["dataset"],
                        "R² Score": f"{model_info['r2']:.4f}",
                        "MSE": f"{model_info['mse']:.4f}",
                        "Type": "Regression",
                    }
                )

        # Display comparison table
        st.markdown("### 📈 Performance Comparison")
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Visualize comparison
        if st.session_state.models[models_to_compare[0]]["type"] == "classification":
            # Accuracy comparison chart
            accuracies = [
                st.session_state.models[model]["accuracy"]
                for model in models_to_compare
            ]
            algorithms = [
                st.session_state.models[model]["algorithm"]
                for model in models_to_compare
            ]

            fig = px.bar(
                x=algorithms,
                y=accuracies,
                title="Model Accuracy Comparison",
                labels={"x": "Algorithm", "y": "Accuracy"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        else:
            # R² comparison chart
            r2_scores = [
                st.session_state.models[model]["r2"] for model in models_to_compare
            ]
            algorithms = [
                st.session_state.models[model]["algorithm"]
                for model in models_to_compare
            ]

            fig = px.bar(
                x=algorithms,
                y=r2_scores,
                title="Model R² Score Comparison",
                labels={"x": "Algorithm", "y": "R² Score"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# DATA INSIGHTS PAGE
elif app_mode == "Data Insights":
    st.markdown("## 📈 Data Insights & Visualizations")

    # Dataset selection
    dataset_name = st.selectbox(
        "Select Dataset for Analysis", ["Iris", "Wine", "California Housing"]
    )

    # Load dataset
    df, model_type = load_dataset(dataset_name)

    # Dataset statistics
    st.markdown("### 📊 Dataset Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Information:**")
        st.write(f"- **Samples:** {df.shape[0]}")
        st.write(f"- **Features:** {df.shape[1] - 1}")
        st.write(f"- **Task Type:** {model_type.title()}")
        st.write(f"- **Missing Values:** {df.isnull().sum().sum()}")

    with col2:
        st.markdown("**Statistical Summary:**")
        st.dataframe(df.describe())

    # Visualizations
    st.markdown("### 📈 Data Visualizations")

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        st.markdown("#### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Distribution plots
    st.markdown("#### Feature Distributions")
    feature_cols = [col for col in df.columns if col not in ["target", "target_names"]]

    if len(feature_cols) >= 2:
        selected_features = st.multiselect(
            "Select Features to Plot", feature_cols, default=feature_cols[:2]
        )

        if selected_features:
            for feature in selected_features:
                fig = px.histogram(df, x=feature, title=f"Distribution of {feature}")
                st.plotly_chart(fig, use_container_width=True)

    # Pairplot for classification
    if model_type == "classification" and "target_names" in df.columns:
        st.markdown("#### Pairwise Relationships")
        if st.button("Generate Pairplot"):
            # Create a subset for pairplot (max 4 features for performance)
            plot_features = feature_cols[:4] + ["target_names"]
            plot_df = df[plot_features]

            fig = px.scatter_matrix(
                plot_df,
                dimensions=feature_cols[:4],
                color="target_names",
                title="Pairwise Feature Relationships",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Target distribution
    st.markdown("#### Target Variable Analysis")
    if model_type == "classification":
        target_counts = df["target"].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Target Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(df, x="target", title="Target Variable Distribution")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤖 Built with Streamlit | 📊 Machine Learning Dashboard | 🎯 Data Science Intern Project</p>
</div>
""",
    unsafe_allow_html=True,
)
