# 🏦 Intelligent Loan Approval Assistant

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [RAG System](#rag-system)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [Future Enhancements](#future-enhancements)

## 🎯 Project Overview

An advanced **Retrieval-Augmented Generation (RAG) Q&A chatbot** system that combines machine learning prediction models with intelligent document retrieval to provide comprehensive loan approval insights. This project demonstrates end-to-end ML pipeline development, from data preprocessing to deployment-ready applications.

### 🚀 Key Innovations
- **Hybrid Intelligence**: Combines traditional ML models with modern RAG architecture
- **Multi-Modal Analysis**: Statistical analysis, ML predictions, and conversational AI
- **Production-Ready**: Containerized deployment with comprehensive monitoring
- **Interpretable AI**: Model explanations and feature importance analysis

## ✨ Features

### 🤖 Machine Learning Models
- **Random Forest Classifier** with hyperparameter tuning
- **XGBoost** for high-performance predictions
- **Logistic Regression** for interpretable baseline
- **Feature engineering** with domain expertise
- **Cross-validation** and model comparison

### 🔍 RAG System
- **Semantic Search** using SentenceTransformers
- **Vector Database** with FAISS for fast retrieval
- **Context-Aware Responses** with conversation history
- **Multi-Source Integration** (documents, data insights, model predictions)

### 📊 Advanced Analytics
- **Interactive Dashboards** with Streamlit
- **Real-time Predictions** with confidence intervals
- **Feature Importance** visualization
- **SHAP explanations** for model interpretability

### 🛠️ Technical Excellence
- **Modular Architecture** with clean separation of concerns
- **Comprehensive Testing** with pytest
- **API Documentation** with FastAPI
- **Monitoring** with MLflow and Weights & Biases
- **Code Quality** with black, flake8, and mypy

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Ingestion    │    │   ML Pipeline       │    │   RAG System        │
│  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │
│  │ CSV Loader    │  │    │  │ Preprocessor  │  │    │  │ Doc Processor │  │
│  │ Data Cleaner  │  │    │  │ Feature Eng   │  │    │  │ Embeddings    │  │
│  │ Validator     │  │    │  │ Model Training│  │    │  │ Vector Store  │  │
│  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                 ┌─────────────────────┴─────────────────────┐
                 │              Frontend                     │
                 │  ┌─────────────────────────────────────┐  │
                 │  │        Streamlit Interface          │  │
                 │  │  ┌─────────────┐ ┌─────────────────┐ │  │
                 │  │  │ Chat Bot    │ │ ML Dashboard    │ │  │
                 │  │  │ RAG Q&A     │ │ Predictions     │ │  │
                 │  │  │ Analytics   │ │ Visualizations  │ │  │
                 │  │  └─────────────┘ └─────────────────┘ │  │
                 │  └─────────────────────────────────────┘  │
                 └───────────────────────────────────────────┘
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- Git
- Optional: CUDA-enabled GPU for faster processing

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/intelligent-loan-approval-assistant.git
cd intelligent-loan-approval-assistant
```

### Step 2: Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download Data
```bash
# Download the dataset from Kaggle
# Place the CSV file in the data/ directory
mkdir -p data
# Copy your loan_approval_dataset.csv to data/Training_Dataset.csv
```

### Step 4: Environment Variables
```bash
# Create .env file
touch .env

# Add your API keys (optional)
echo "OPENAI_API_KEY=your_openai_key_here" >> .env
echo "HUGGINGFACE_API_KEY=your_hf_key_here" >> .env
```

### Step 5: Initialize System
```bash
# Run the setup script
python setup.py

# Or manually run components
python ml_models.py  # Train ML models
python rag_system.py  # Initialize RAG system
```

## 🚀 Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Running Individual Components

#### 1. Train ML Models
```bash
python ml_models.py
```

#### 2. Initialize RAG System
```bash
python rag_system.py
```

#### 3. API Server
```bash
uvicorn api:app --reload
```

### Using the System

#### 1. Load Data
```python
from ml_models import LoanApprovalML
from rag_system import RAGSystem

# Initialize ML system
ml_system = LoanApprovalML()
ml_system.load_data('data/Training_Dataset.csv')
ml_system.train_models()

# Initialize RAG system
rag_system = RAGSystem()
```

#### 2. Make Predictions
```python
# Single prediction
prediction = ml_system.predict_single({
    'Gender': 'Male',
    'Married': 'Yes',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
})

print(f"Approval Probability: {prediction['probability']:.2f}")
```

#### 3. Ask Questions
```python
# Query the RAG system
response = rag_system.query("What factors most influence loan approval?")
print(response['response'])
```

## 📁 Project Structure

```
intelligent-loan-approval-assistant/
├── 📁 data/
│   ├── Training_Dataset.csv
│   └── processed/
├── 📁 models/
│   ├── saved_models/
│   └── vector_store/
├── 📁 src/
│   ├── ml_models.py          # ML pipeline implementation
│   ├── rag_system.py         # RAG system implementation
│   ├── data_processor.py     # Data preprocessing utilities
│   └── utils.py              # Helper functions
├── 📁 frontend/
│   ├── app.py                # Main Streamlit application
│   ├── components/           # UI components
│   └── pages/                # Multi-page app structure
├── 📁 api/
│   ├── main.py               # FastAPI application
│   ├── endpoints/            # API endpoints
│   └── schemas.py            # Pydantic models
├── 📁 tests/
│   ├── test_ml_models.py     # ML model tests
│   ├── test_rag_system.py    # RAG system tests
│   └── test_api.py           # API tests
├── 📁 docs/
│   ├── api_documentation.md
│   ├── model_performance.md
│   └── deployment_guide.md
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_rag_system_development.ipynb
├── 📁 config/
│   ├── model_config.yaml
│   └── rag_config.yaml
├── 📁 deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes/
├── requirements.txt
├── README.md
├── setup.py
└── .env.example