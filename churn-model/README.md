# Churn Prediction Model - Client Retention Strategy App

## Overview
This Python-based churn prediction model is designed for an insurance brokering company to identify at-risk clients likely to churn. The model uses machine learning to predict client churn risk based on insurance-specific features and provides actionable insights for retention strategies.

## Key Features
- **Insurance-specific feature engineering**: Policy tenure, claim patterns, engagement scores
- **High accuracy target**: >80% precision and recall for churn prediction
- **Real-time predictions**: <500ms inference time per client
- **Batch processing**: Handle up to 10,000 clients with scheduled updates
- **Model explainability**: SHAP values for understanding churn risk factors
- **REST API integration**: Seamless integration with Spring Boot backend

## Model Performance Targets
- **Precision**: >80%
- **Recall**: >80%
- **F1-Score**: >80%
- **Inference Time**: <500ms per client
- **Batch Processing**: <1 hour for 10k clients

## Project Structure
```
churn-model/
├── data/                   # Data storage and processing
├── models/                 # Trained model artifacts
├── src/                    # Source code
│   ├── data_processing/    # Data preprocessing pipelines
│   ├── feature_engineering/ # Feature engineering modules
│   ├── model_training/    # Model development and training
│   ├── model_deployment/  # API and deployment code
│   └── evaluation/         # Model evaluation and metrics
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit and integration tests
└── api/                    # REST API endpoints
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run data analysis: `python notebooks/data_analysis.ipynb`
3. Train model: `python src/model_training/train_model.py`
4. Start API server: `python api/app.py`

## Integration with Spring Boot
The model provides REST API endpoints for integration:
- `POST /predict` - Single client prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/status` - Model health and performance metrics
- `GET /model/explain/{client_id}` - SHAP explanations for specific client

## Business Impact
- **Churn Reduction**: Target 15% reduction in client churn
- **Retention Success**: 70% success rate for approved retention packages
- **ROI**: Data-driven retention strategies with measurable business impact
