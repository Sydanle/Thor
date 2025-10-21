"""
Churn prediction model training module.
Implements multiple ML algorithms and selects the best performing model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Comprehensive churn prediction model trainer.
    Implements multiple algorithms and selects the best performing model.
    """
    
    def __init__(self, target_accuracy: float = 0.80):
        """
        Initialize model trainer.
        
        Args:
            target_accuracy: Target precision/recall for model selection
        """
        self.target_accuracy = target_accuracy
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'churn_label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: Processed client data DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of features and target
        """
        logger.info("Preparing data for model training...")
        
        # Select features for training
        exclude_columns = [
            'client_id', 'client_name', 'registration_date', 'last_contact_date',
            'policy_start_date', 'policy_end_date', 'last_claim_date', 'last_payment_date',
            'churn_date', 'churn_reason', 'retention_status', target_column
        ]
        
        # Get feature columns
        self.feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        self.feature_columns = [col for col in self.feature_columns 
                               if df[col].isnull().sum() / len(df) < missing_threshold]
        
        # Remove constant columns
        self.feature_columns = [col for col in self.feature_columns 
                               if df[col].nunique() > 1]
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X = self._encode_categorical_features(X)
        
        logger.info(f"Prepared {len(self.feature_columns)} features for training")
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Churn rate: {y.mean():.2%}")
        
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        
        # For numeric columns, fill with median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            X[col] = X[col].fillna(X[col].median())
        
        # For categorical columns, fill with mode
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        return X
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train multiple models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with model results
        """
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self._handle_class_imbalance(X_train_scaled, y_train)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
            'Logistic Regression': LogisticRegression(
                random_state=random_state, max_iter=1000
            ),
            'SVM': SVC(
                random_state=random_state, probability=True
            )
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Store model
            self.models[name] = model
            
            logger.info(f"{name} - Precision: {metrics['precision']:.3f}, "
                       f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
        
        # Select best model
        self._select_best_model(results, y_test)
        
        # Store results
        self.model_metrics = results
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'results': results,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name
        }
    
    def _handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using SMOTE."""
        
        # Check if imbalance exists
        if y.mean() < 0.1 or y.mean() > 0.9:
            logger.info("Handling class imbalance with SMOTE...")
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            logger.info(f"Balanced dataset: {len(X_balanced)} samples, churn rate: {y_balanced.mean():.2%}")
            return X_balanced, y_balanced
        
        return X, y
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def _select_best_model(self, results: Dict[str, Any], y_test: pd.Series) -> None:
        """Select the best model based on performance criteria."""
        
        best_score = 0
        best_model_name = None
        
        for name, result in results.items():
            metrics = result['metrics']
            
            # Calculate composite score (weighted average of precision and recall)
            composite_score = (metrics['precision'] * 0.5 + metrics['recall'] * 0.5)
            
            # Check if model meets target accuracy
            if (metrics['precision'] >= self.target_accuracy and 
                metrics['recall'] >= self.target_accuracy and 
                composite_score > best_score):
                
                best_score = composite_score
                best_model_name = name
        
        if best_model_name is None:
            # If no model meets target, select best overall
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x]['metrics']['f1'])
            logger.warning(f"No model met target accuracy. Selected best overall: {best_model_name}")
        
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best model metrics: {results[best_model_name]['metrics']}")
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for the best model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with tuning results
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if self.best_model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {self.best_model_name}")
            return {}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle class imbalance
        X_balanced, y_balanced = self._handle_class_imbalance(X_scaled, y)
        
        # Get base model
        base_model = self.models[self.best_model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grids[self.best_model_name],
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_balanced, y_balanced)
        
        # Update best model with tuned parameters
        self.best_model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model performance...")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': auc_score},
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Model evaluation completed:")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1']:.3f}")
        logger.info(f"  AUC: {metrics['auc']:.3f}")
        
        return results
    
    def save_model(self, model_path: str = '../models/') -> None:
        """
        Save the trained model and related artifacts.
        
        Args:
            model_path: Path to save model artifacts
        """
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_path, 'churn_model.pkl')
        joblib.dump(self.best_model, model_file)
        
        # Save scaler
        scaler_file = os.path.join(model_path, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_file)
        
        # Save label encoders
        encoders_file = os.path.join(model_path, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_file)
        
        # Save feature columns
        features_file = os.path.join(model_path, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, features_file)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'target_accuracy': self.target_accuracy,
            'training_date': datetime.now().isoformat(),
            'model_metrics': self.model_metrics
        }
        
        metadata_file = os.path.join(model_path, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_file)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model file: {model_file}")
        logger.info(f"Scaler file: {scaler_file}")
        logger.info(f"Encoders file: {encoders_file}")
        logger.info(f"Features file: {features_file}")
        logger.info(f"Metadata file: {metadata_file}")
    
    def plot_model_performance(self, evaluation_results: Dict[str, Any], 
                              save_path: str = '../models/') -> None:
        """
        Create visualization plots for model performance.
        
        Args:
            evaluation_results: Results from model evaluation
            save_path: Path to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        conf_matrix = evaluation_results['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curve
        roc_data = evaluation_results['roc_curve']
        axes[0,1].plot(roc_data['fpr'], roc_data['tpr'], 
                      label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Feature Importance
        if evaluation_results['feature_importance'] is not None:
            top_features = evaluation_results['feature_importance'].head(15)
            axes[1,0].barh(range(len(top_features)), top_features['importance'])
            axes[1,0].set_yticks(range(len(top_features)))
            axes[1,0].set_yticklabels(top_features['feature'])
            axes[1,0].set_xlabel('Importance')
            axes[1,0].set_title('Top 15 Feature Importance')
            axes[1,0].invert_yaxis()
        
        # Model Comparison
        if self.model_metrics:
            model_names = list(self.model_metrics.keys())
            f1_scores = [self.model_metrics[name]['metrics']['f1'] for name in model_names]
            
            axes[1,1].bar(model_names, f1_scores)
            axes[1,1].set_ylabel('F1-Score')
            axes[1,1].set_title('Model Comparison (F1-Score)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(save_path, 'model_performance.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance plots saved to {plot_file}")


def main():
    """Main function to train the churn prediction model."""
    
    # Create sample data for training
    from data_processing.data_loader import create_sample_data
    from feature_engineering.feature_engineer import InsuranceFeatureEngineer
    
    print("=== CHURN PREDICTION MODEL TRAINING ===")
    
    # Load and prepare data
    print("1. Loading and preparing data...")
    df = create_sample_data(5000)  # 5000 clients for training
    
    # Apply feature engineering
    feature_engineer = InsuranceFeatureEngineer()
    df_features = feature_engineer.create_all_features(df)
    
    print(f"Dataset prepared: {len(df_features)} clients, {len(df_features.columns)} features")
    print(f"Churn rate: {df_features['churn_label'].mean():.2%}")
    
    # Initialize trainer
    trainer = ChurnModelTrainer(target_accuracy=0.80)
    
    # Prepare data
    X, y = trainer.prepare_data(df_features)
    
    # Train models
    print("\n2. Training models...")
    training_results = trainer.train_models(X, y)
    
    # Hyperparameter tuning
    print("\n3. Hyperparameter tuning...")
    tuning_results = trainer.hyperparameter_tuning(X, y)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    evaluation_results = trainer.evaluate_model(training_results['X_test'], training_results['y_test'])
    
    # Save model
    print("\n5. Saving model...")
    trainer.save_model()
    
    # Create performance plots
    print("\n6. Creating performance plots...")
    trainer.plot_model_performance(evaluation_results)
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    metrics = evaluation_results['metrics']
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"AUC: {metrics['auc']:.3f}")
    
    if metrics['precision'] >= 0.80 and metrics['recall'] >= 0.80:
        print("✅ Model meets target accuracy requirements!")
    else:
        print("⚠️ Model does not meet target accuracy requirements.")
    
    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main()
