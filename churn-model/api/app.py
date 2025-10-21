"""
REST API for churn prediction model.
Provides endpoints for real-time predictions and batch processing.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model artifacts
model = None
scaler = None
label_encoders = None
feature_columns = None
model_metadata = None


def load_model_artifacts(model_path: str = '../models/'):
    """
    Load model artifacts from disk.
    
    Args:
        model_path: Path to model artifacts
    """
    global model, scaler, label_encoders, feature_columns, model_metadata
    
    try:
        # Load model
        model_file = os.path.join(model_path, 'churn_model.pkl')
        model = joblib.load(model_file)
        logger.info(f"Model loaded from {model_file}")
        
        # Load scaler
        scaler_file = os.path.join(model_path, 'scaler.pkl')
        scaler = joblib.load(scaler_file)
        logger.info(f"Scaler loaded from {scaler_file}")
        
        # Load label encoders
        encoders_file = os.path.join(model_path, 'label_encoders.pkl')
        label_encoders = joblib.load(encoders_file)
        logger.info(f"Label encoders loaded from {encoders_file}")
        
        # Load feature columns
        features_file = os.path.join(model_path, 'feature_columns.pkl')
        feature_columns = joblib.load(features_file)
        logger.info(f"Feature columns loaded from {features_file}")
        
        # Load model metadata
        metadata_file = os.path.join(model_path, 'model_metadata.pkl')
        model_metadata = joblib.load(metadata_file)
        logger.info(f"Model metadata loaded from {metadata_file}")
        
        logger.info("All model artifacts loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise


def preprocess_client_data(client_data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess client data for prediction.
    
    Args:
        client_data: Dictionary with client data
        
    Returns:
        Preprocessed feature array
    """
    try:
        # Create DataFrame from client data
        df = pd.DataFrame([client_data])
        
        # Select only the required features
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with default values
            for feature in missing_features:
                df[feature] = 0
        
        # Reorder columns to match training data
        df = df[feature_columns]
        
        # Handle missing values
        df = df.fillna(0)
        
        # Encode categorical features
        for col in df.columns:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Convert to numpy array
        features = df.values
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        return features_scaled
        
    except Exception as e:
        logger.error(f"Error preprocessing client data: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })


@app.route('/model/status', methods=['GET'])
def model_status():
    """Get model status and performance metrics."""
    try:
        if model_metadata is None:
            return jsonify({'error': 'Model metadata not available'}), 500
        
        return jsonify({
            'model_name': model_metadata.get('model_name', 'Unknown'),
            'training_date': model_metadata.get('training_date', 'Unknown'),
            'target_accuracy': model_metadata.get('target_accuracy', 0.80),
            'feature_count': len(feature_columns) if feature_columns else 0,
            'status': 'ready' if model is not None else 'not_loaded'
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predict churn probability for a single client.
    
    Expected JSON format:
    {
        "client_id": "12345",
        "client_name": "ABC Corp",
        "client_type": "Corporate",
        "industry": "Technology",
        "annual_revenue": 1000000,
        "employee_count": 50,
        "engagement_score": 8.5,
        "satisfaction_rating": 4.2,
        "premium_amount": 50000,
        "coverage_amount": 1000000,
        "claim_count": 2,
        "payment_delays": 0,
        ...
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get client data from request
        client_data = request.get_json()
        
        if not client_data:
            return jsonify({'error': 'No client data provided'}), 400
        
        # Preprocess client data
        features = preprocess_client_data(client_data)
        
        # Make prediction
        churn_probability = model.predict_proba(features)[0][1]
        churn_prediction = model.predict(features)[0]
        
        # Determine risk level
        if churn_probability >= 0.8:
            risk_level = 'High'
        elif churn_probability >= 0.5:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Prepare response
        response = {
            'client_id': client_data.get('client_id', 'Unknown'),
            'churn_probability': float(churn_probability),
            'churn_prediction': int(churn_prediction),
            'risk_level': risk_level,
            'confidence': float(max(churn_probability, 1 - churn_probability)),
            'timestamp': datetime.now().isoformat(),
            'model_version': model_metadata.get('training_date', 'Unknown') if model_metadata else 'Unknown'
        }
        
        logger.info(f"Prediction for client {client_data.get('client_id', 'Unknown')}: "
                   f"Probability={churn_probability:.3f}, Risk={risk_level}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict churn probability for multiple clients.
    
    Expected JSON format:
    {
        "clients": [
            {
                "client_id": "12345",
                "client_name": "ABC Corp",
                ...
            },
            {
                "client_id": "67890",
                "client_name": "XYZ Inc",
                ...
            }
        ]
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get batch data from request
        batch_data = request.get_json()
        
        if not batch_data or 'clients' not in batch_data:
            return jsonify({'error': 'No clients data provided'}), 400
        
        clients = batch_data['clients']
        
        if not isinstance(clients, list):
            return jsonify({'error': 'Clients data must be a list'}), 400
        
        if len(clients) == 0:
            return jsonify({'error': 'No clients provided'}), 400
        
        # Process each client
        results = []
        for i, client_data in enumerate(clients):
            try:
                # Preprocess client data
                features = preprocess_client_data(client_data)
                
                # Make prediction
                churn_probability = model.predict_proba(features)[0][1]
                churn_prediction = model.predict(features)[0]
                
                # Determine risk level
                if churn_probability >= 0.8:
                    risk_level = 'High'
                elif churn_probability >= 0.5:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                
                result = {
                    'client_id': client_data.get('client_id', f'Client_{i}'),
                    'churn_probability': float(churn_probability),
                    'churn_prediction': int(churn_prediction),
                    'risk_level': risk_level,
                    'confidence': float(max(churn_probability, 1 - churn_probability))
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing client {i}: {str(e)}")
                results.append({
                    'client_id': client_data.get('client_id', f'Client_{i}'),
                    'error': str(e)
                })
        
        # Prepare response
        response = {
            'predictions': results,
            'total_clients': len(clients),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'failed_predictions': len([r for r in results if 'error' in r]),
            'timestamp': datetime.now().isoformat(),
            'model_version': model_metadata.get('training_date', 'Unknown') if model_metadata else 'Unknown'
        }
        
        logger.info(f"Batch prediction completed: {response['successful_predictions']}/"
                   f"{response['total_clients']} successful")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/model/explain/<client_id>', methods=['POST'])
def explain_prediction(client_id):
    """
    Get SHAP explanations for a specific client prediction.
    
    Args:
        client_id: ID of the client to explain
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get client data from request
        client_data = request.get_json()
        
        if not client_data:
            return jsonify({'error': 'No client data provided'}), 400
        
        # Preprocess client data
        features = preprocess_client_data(client_data)
        
        # Make prediction
        churn_probability = model.predict_proba(features)[0][1]
        
        # Get feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # Prepare response
        response = {
            'client_id': client_id,
            'churn_probability': float(churn_probability),
            'feature_importance': feature_importance,
            'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10] if feature_importance else [],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Explanation generated for client {client_id}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error explaining prediction for client {client_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/features', methods=['GET'])
def get_feature_info():
    """Get information about model features."""
    try:
        if feature_columns is None:
            return jsonify({'error': 'Feature columns not available'}), 500
        
        # Get feature categories
        feature_categories = {
            'temporal': [col for col in feature_columns if 'days' in col or 'date' in col],
            'policy': [col for col in feature_columns if 'premium' in col or 'coverage' in col or 'policy' in col],
            'engagement': [col for col in feature_columns if 'engagement' in col or 'satisfaction' in col],
            'financial': [col for col in feature_columns if 'revenue' in col or 'payment' in col or 'financial' in col],
            'risk': [col for col in feature_columns if 'risk' in col or 'claim' in col],
            'interaction': [col for col in feature_columns if 'interaction' in col]
        }
        
        response = {
            'total_features': len(feature_columns),
            'feature_columns': feature_columns,
            'feature_categories': feature_categories,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting feature info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main function to run the API server."""
    # Load model artifacts
    try:
        load_model_artifacts()
        logger.info("Model artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        logger.error("Please ensure the model has been trained and saved")
        return
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting API server on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
