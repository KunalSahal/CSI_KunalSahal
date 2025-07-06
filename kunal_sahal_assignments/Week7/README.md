
# Machine Learning Model Deployment Dashboard

> *My first Streamlit application - A comprehensive ML dashboard for model training, prediction, and analysis*
> [Click here to launch the app](https://kunalsahal.streamlit.app)

## About This Project

Hello! I'm excited to share my first Streamlit application. As a data science intern, I wanted to create something that demonstrates end-to-end machine learning workflows in an interactive web interface. This project helped me learn how to deploy ML models and make them accessible to non-technical users.

## 🔗 Live Demo
[Click here to launch the app](https://kunalsahal.streamlit.app)

## What This App Does

This dashboard allows users to:
- Train ML models on different datasets with various algorithms
- Make real-time predictions with interactive input controls
- Compare model performance side-by-side
- Explore data insights through interactive visualizations
- Understand model behavior through feature importance and probabilities


## Datasets & Models

### Datasets Available:
- Iris Dataset - Classic flower classification (150 samples, 4 features)
- Wine Dataset - Wine quality classification (178 samples, 13 features)
- California Housing - House price prediction (20,640 samples, 8 features)

### Algorithms Implemented:
- Classification: Random Forest, Logistic Regression, SVM
- Regression: Random Forest, Linear Regression, SVR

## Technical Stack

- Frontend: Streamlit (Python web framework)
- ML Libraries: scikit-learn, pandas, numpy
- Visualization: Plotly, Matplotlib, Seaborn
- Data Processing: StandardScaler for feature scaling
- Model Persistence: Session state management

## Project Structure

```
ML_Streamlit_App/
│
├── ml_app.py              # Main application file
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── screenshots/           # App screenshots (optional)
```

## Features I'm Proud Of

### Model Training Interface
- Drag-and-drop dataset selection
- Algorithm comparison with validation
- Real-time training progress
- Automatic feature scaling
- Performance metrics display

### Interactive Predictions
- User-friendly sliders for input features
- Real-time prediction updates
- Confidence scores for classification
- Input validation and error handling

### Data Visualization
- Correlation heatmaps
- Feature distribution plots
- Pairwise relationship analysis
- Interactive Plotly charts

### Model Comparison
- Side-by-side performance metrics
- Accuracy/R² score comparisons
- Best model recommendations

## What I Learned

This project taught me:
- Streamlit basics: Layout, widgets, and interactivity
- ML pipeline: Data preprocessing, model training, evaluation
- Web deployment: Making ML models accessible through web interfaces
- Data visualization: Creating meaningful charts and insights
- User experience: Designing intuitive interfaces for complex ML concepts

## Installation & Setup

### Prerequisites
```bash
Python 3.7+
```

### Step-by-Step Installation
1. Clone the repository
   ```bash
   git clone [your-repo-url]
   cd ML_Streamlit_App
   ```

2. Create virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application
   ```bash
   streamlit run ml_app.py
   ```

## Dependencies

```
streamlit>=1.28.0
pandas>=1.5.3
numpy>=1.24.3
scikit-learn>=1.3.0
matplotlib>=3.7.1
seaborn>=0.12.2
plotly>=5.15.0
joblib>=1.3.2
```

## How to Use

### 1. Home Page
- Overview of app features and capabilities
- Quick statistics about available datasets and models

### 2. Model Training
- Select dataset and algorithm
- Configure training parameters
- Train model and view performance metrics
- Save trained models for predictions

### 3. Prediction Interface
- Choose from trained models
- Input feature values using interactive sliders
- Get real-time predictions with confidence scores
- View prediction explanations

### 4. Model Comparison
- Compare performance across different algorithms
- View side-by-side accuracy or R² scores
- Identify best performing models

### 5. Data Insights
- Explore dataset statistics and distributions
- View correlation matrices and feature relationships
- Generate interactive visualizations

## Known Issues & Future Improvements

### Current Limitations:
- Limited to built-in datasets (working on file upload feature)
- No model export functionality yet
- Basic error handling (improving in next version)

### Future Enhancements:
- File upload for custom datasets
- Model export/download functionality
- Advanced hyperparameter tuning
- Deep learning model support
- A/B testing framework
- Model monitoring dashboard

## Contributing

This is my learning project, but I'm open to suggestions. If you find bugs or have ideas for improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

*Built by a data science intern learning Streamlit for the first time.*
