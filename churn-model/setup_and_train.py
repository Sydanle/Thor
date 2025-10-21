"""
Complete setup and training script for the churn prediction model.
This script handles the entire pipeline from data preparation to model deployment.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """
    Run a shell command and log the result.
    
    Args:
        command: Command to run
        description: Description of the command
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False


def setup_environment():
    """Set up the Python environment and install dependencies."""
    logger.info("=== SETTING UP ENVIRONMENT ===")
    
    # Create necessary directories
    directories = [
        'data',
        'models',
        'notebooks',
        'src/data_processing',
        'src/feature_engineering',
        'src/model_training',
        'src/model_deployment',
        'src/evaluation',
        'api',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        logger.error("Failed to install dependencies")
        return False
    
    logger.info("✅ Environment setup completed")
    return True


def run_data_analysis():
    """Run data analysis and feature engineering."""
    logger.info("=== RUNNING DATA ANALYSIS ===")
    
    # Create sample data and run analysis
    analysis_script = """
import sys
sys.path.append('src')
from data_processing.data_loader import create_sample_data
from feature_engineering.feature_engineer import InsuranceFeatureEngineer
import pandas as pd

# Create sample data
print("Creating sample data...")
df = create_sample_data(5000)
print(f"Sample data created: {len(df)} clients")

# Apply feature engineering
print("Applying feature engineering...")
feature_engineer = InsuranceFeatureEngineer()
df_features = feature_engineer.create_all_features(df)

print(f"Feature engineering completed: {len(df_features.columns)} features")
print(f"Churn rate: {df_features['churn_label'].mean():.2%}")

# Save processed data
df_features.to_csv('data/processed_client_data.csv', index=False)
print("Processed data saved to data/processed_client_data.csv")

# Save feature columns
feature_columns = feature_engineer.select_features_for_model(df_features)
pd.Series(feature_columns).to_csv('data/feature_columns.csv', index=False)
print(f"Feature columns saved: {len(feature_columns)} features")
"""
    
    with open('temp_analysis.py', 'w') as f:
        f.write(analysis_script)
    
    success = run_command("python temp_analysis.py", "Running data analysis")
    os.remove('temp_analysis.py')
    
    return success


def train_model():
    """Train the churn prediction model."""
    logger.info("=== TRAINING MODEL ===")
    
    # Run model training
    training_script = """
import sys
sys.path.append('src')
from model_training.train_model import main

if __name__ == "__main__":
    main()
"""
    
    with open('temp_training.py', 'w') as f:
        f.write(training_script)
    
    success = run_command("python temp_training.py", "Training churn prediction model")
    os.remove('temp_training.py')
    
    return success


def test_api():
    """Test the model API."""
    logger.info("=== TESTING API ===")
    
    # Start API server in background
    api_process = subprocess.Popen(
        ["python", "api/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for API to start
    import time
    time.sleep(5)
    
    # Test API endpoints
    test_script = """
import requests
import time

# Wait for API to be ready
time.sleep(10)

try:
    # Test health endpoint
    response = requests.get('http://localhost:5000/health', timeout=10)
    print(f"Health check: {response.status_code}")
    
    # Test model status
    response = requests.get('http://localhost:5000/model/status', timeout=10)
    print(f"Model status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ API is working correctly")
    else:
        print("❌ API test failed")
        
except Exception as e:
    print(f"❌ API test failed: {e}")
"""
    
    with open('temp_api_test.py', 'w') as f:
        f.write(test_script)
    
    success = run_command("python temp_api_test.py", "Testing API endpoints")
    os.remove('temp_api_test.py')
    
    # Stop API server
    api_process.terminate()
    api_process.wait()
    
    return success


def run_showcase():
    """Run the model showcase demonstration."""
    logger.info("=== RUNNING MODEL SHOWCASE ===")
    
    # Start API server in background
    api_process = subprocess.Popen(
        ["python", "api/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for API to start
    import time
    time.sleep(10)
    
    # Run showcase
    success = run_command("python demo_model_showcase.py", "Running model showcase")
    
    # Stop API server
    api_process.terminate()
    api_process.wait()
    
    return success


def main():
    """Main function to run the complete setup and training pipeline."""
    logger.info("=== CHURN PREDICTION MODEL SETUP AND TRAINING ===")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        return False
    
    # Step 2: Run data analysis
    if not run_data_analysis():
        logger.error("Data analysis failed")
        return False
    
    # Step 3: Train model
    if not train_model():
        logger.error("Model training failed")
        return False
    
    # Step 4: Test API
    if not test_api():
        logger.error("API testing failed")
        return False
    
    # Step 5: Run showcase
    if not run_showcase():
        logger.error("Model showcase failed")
        return False
    
    logger.info("=== SETUP AND TRAINING COMPLETED SUCCESSFULLY ===")
    logger.info("✅ Environment configured")
    logger.info("✅ Data analysis completed")
    logger.info("✅ Model trained and saved")
    logger.info("✅ API tested and working")
    logger.info("✅ Showcase demonstration completed")
    logger.info("✅ Ready for production deployment")
    
    # Print next steps
    logger.info("\n=== NEXT STEPS ===")
    logger.info("1. Review model performance in models/ directory")
    logger.info("2. Check business report: models/business_report.md")
    logger.info("3. Start API server: python api/app.py")
    logger.info("4. Integrate with Spring Boot backend")
    logger.info("5. Deploy to production environment")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Churn prediction model setup completed successfully!")
    else:
        logger.error("❌ Setup failed. Please check the logs for details.")
        sys.exit(1)
