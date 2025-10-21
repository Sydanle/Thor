"""
Feature engineering module for churn prediction model.
Creates insurance-specific features for improved model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class InsuranceFeatureEngineer:
    """
    Feature engineering for insurance churn prediction.
    Creates domain-specific features that capture client behavior patterns.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_columns = []
        self.categorical_features = []
        self.numeric_features = []
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for churn prediction.
        
        Args:
            df: Client data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Create temporal features
        df = self._create_temporal_features(df)
        
        # Create policy-related features
        df = self._create_policy_features(df)
        
        # Create engagement features
        df = self._create_engagement_features(df)
        
        # Create financial features
        df = self._create_financial_features(df)
        
        # Create risk features
        df = self._create_risk_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Create lag features
        df = self._create_lag_features(df)
        
        # Create trend features
        df = self._create_trend_features(df)
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        
        # Client tenure (days since registration)
        df['client_tenure_days'] = (datetime.now() - df['registration_date']).dt.days
        
        # Days since last contact
        df['days_since_last_contact'] = (datetime.now() - df['last_contact_date']).dt.days
        
        # Days since last claim
        df['days_since_last_claim'] = (datetime.now() - df['last_claim_date']).dt.days
        
        # Days since last payment
        df['days_since_last_payment'] = (datetime.now() - df['last_payment_date']).dt.days
        
        # Policy age
        df['policy_age_days'] = (datetime.now() - df['policy_start_date']).dt.days
        
        # Days until policy renewal
        df['days_until_renewal'] = (df['policy_end_date'] - datetime.now()).dt.days
        
        # Seasonality features
        df['registration_month'] = df['registration_date'].dt.month
        df['registration_quarter'] = df['registration_date'].dt.quarter
        df['registration_year'] = df['registration_date'].dt.year
        
        return df
    
    def _create_policy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create policy-specific features."""
        
        # Premium per coverage ratio
        df['premium_coverage_ratio'] = df['premium_amount'] / df['coverage_amount']
        
        # Premium change rate
        df['premium_change_rate'] = df['premium_changes']
        
        # Coverage change rate
        df['coverage_change_rate'] = df['coverage_changes']
        
        # Renewal frequency (renewals per year)
        df['renewal_frequency'] = df['renewal_count'] / (df['client_tenure_days'] / 365.25)
        
        # Policy type encoding
        df['is_corporate_policy'] = (df['policy_type'] == 'Corporate').astype(int)
        df['is_individual_policy'] = (df['policy_type'] == 'Individual').astype(int)
        
        # Premium tier (based on premium amount)
        df['premium_tier'] = pd.cut(df['premium_amount'], 
                                  bins=[0, 1000, 5000, 10000, float('inf')], 
                                  labels=['Low', 'Medium', 'High', 'Premium'])
        
        return df
    
    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create client engagement features."""
        
        # Overall engagement score (weighted combination)
        df['overall_engagement'] = (
            df['engagement_score'] * 0.3 +
            df['communication_score'] * 0.3 +
            df['satisfaction_rating'] * 0.2 +
            (df['email_opens'] / df['email_opens'].max()) * 0.1 +
            (df['website_visits'] / df['website_visits'].max()) * 0.1
        )
        
        # Engagement trend (recent vs historical)
        df['engagement_trend'] = df['overall_engagement'] - df['overall_engagement'].rolling(30).mean()
        
        # Communication frequency
        df['communication_frequency'] = df['contact_frequency'] / (df['client_tenure_days'] / 30)
        
        # Digital engagement score
        df['digital_engagement'] = (
            df['email_opens'] * 0.4 +
            df['email_clicks'] * 0.3 +
            df['website_visits'] * 0.3
        )
        
        # Support dependency
        df['support_dependency'] = df['support_tickets'] / (df['client_tenure_days'] / 30)
        
        # Meeting attendance rate
        df['meeting_attendance_rate'] = df['meeting_attendance'] / (df['client_tenure_days'] / 90)
        
        return df
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial behavior features."""
        
        # Revenue per employee
        df['revenue_per_employee'] = df['annual_revenue'] / df['employee_count']
        
        # Premium as percentage of revenue
        df['premium_revenue_ratio'] = df['premium_amount'] / df['annual_revenue']
        
        # Payment behavior score
        df['payment_behavior_score'] = np.where(
            df['payment_delays'] == 0, 10,
            np.where(df['payment_delays'] <= 2, 7,
                    np.where(df['payment_delays'] <= 5, 4, 1))
        )
        
        # Payment frequency score
        payment_freq_map = {'Monthly': 12, 'Quarterly': 4, 'Annual': 1}
        df['payment_frequency_score'] = df['payment_frequency'].map(payment_freq_map)
        
        # Claim to premium ratio
        df['claim_premium_ratio'] = df['total_claim_amount'] / df['premium_amount']
        
        # Financial stability score
        df['financial_stability_score'] = (
            (df['annual_revenue'] / df['annual_revenue'].max()) * 0.4 +
            (df['employee_count'] / df['employee_count'].max()) * 0.2 +
            (df['payment_behavior_score'] / 10) * 0.4
        )
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk assessment features."""
        
        # Claim frequency per year
        df['claim_frequency_annual'] = df['claim_count'] / (df['client_tenure_days'] / 365.25)
        
        # Claim severity score
        df['claim_severity_normalized'] = df['claim_severity_score'] / 10
        
        # Risk score (combination of claim frequency and severity)
        df['risk_score'] = (
            df['claim_frequency_annual'] * 0.6 +
            df['claim_severity_normalized'] * 0.4
        )
        
        # High-risk client flag
        df['is_high_risk'] = (df['risk_score'] > df['risk_score'].quantile(0.8)).astype(int)
        
        # Payment risk score
        df['payment_risk_score'] = np.where(
            df['payment_delays'] == 0, 1,
            np.where(df['payment_delays'] <= 2, 2,
                    np.where(df['payment_delays'] <= 5, 3, 4))
        )
        
        # Engagement risk score
        df['engagement_risk_score'] = np.where(
            df['overall_engagement'] >= 7, 1,
            np.where(df['overall_engagement'] >= 5, 2,
                    np.where(df['overall_engagement'] >= 3, 3, 4))
        )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables."""
        
        # Engagement × Risk interaction
        df['engagement_risk_interaction'] = df['overall_engagement'] * df['risk_score']
        
        # Premium × Engagement interaction
        df['premium_engagement_interaction'] = df['premium_amount'] * df['overall_engagement']
        
        # Tenure × Engagement interaction
        df['tenure_engagement_interaction'] = df['client_tenure_days'] * df['overall_engagement']
        
        # Claim frequency × Payment behavior
        df['claim_payment_interaction'] = df['claim_frequency_annual'] * df['payment_behavior_score']
        
        # Revenue × Premium interaction
        df['revenue_premium_interaction'] = df['annual_revenue'] * df['premium_amount']
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series patterns."""
        
        # Sort by client_id and date for proper lag calculation
        df_sorted = df.sort_values(['client_id', 'registration_date'])
        
        # Lag features for engagement
        df_sorted['engagement_lag_1'] = df_sorted.groupby('client_id')['overall_engagement'].shift(1)
        df_sorted['engagement_lag_2'] = df_sorted.groupby('client_id')['overall_engagement'].shift(2)
        
        # Lag features for satisfaction
        df_sorted['satisfaction_lag_1'] = df_sorted.groupby('client_id')['satisfaction_rating'].shift(1)
        
        # Lag features for claims
        df_sorted['claim_frequency_lag_1'] = df_sorted.groupby('client_id')['claim_frequency_annual'].shift(1)
        
        return df_sorted
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend features for time series analysis."""
        
        # Engagement trend (slope over time)
        df['engagement_trend_slope'] = df.groupby('client_id')['overall_engagement'].apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Satisfaction trend
        df['satisfaction_trend_slope'] = df.groupby('client_id')['satisfaction_rating'].apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Premium change trend
        df['premium_trend_slope'] = df.groupby('client_id')['premium_change_rate'].apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        return df
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """
        Get feature categories for model interpretation.
        
        Returns:
            Dictionary mapping categories to feature lists
        """
        return {
            'temporal': [
                'client_tenure_days', 'days_since_last_contact', 'days_since_last_claim',
                'days_since_last_payment', 'policy_age_days', 'days_until_renewal'
            ],
            'policy': [
                'premium_coverage_ratio', 'premium_change_rate', 'coverage_change_rate',
                'renewal_frequency', 'is_corporate_policy', 'is_individual_policy'
            ],
            'engagement': [
                'overall_engagement', 'engagement_trend', 'communication_frequency',
                'digital_engagement', 'support_dependency', 'meeting_attendance_rate'
            ],
            'financial': [
                'revenue_per_employee', 'premium_revenue_ratio', 'payment_behavior_score',
                'payment_frequency_score', 'claim_premium_ratio', 'financial_stability_score'
            ],
            'risk': [
                'claim_frequency_annual', 'claim_severity_normalized', 'risk_score',
                'is_high_risk', 'payment_risk_score', 'engagement_risk_score'
            ],
            'interaction': [
                'engagement_risk_interaction', 'premium_engagement_interaction',
                'tenure_engagement_interaction', 'claim_payment_interaction',
                'revenue_premium_interaction'
            ]
        }
    
    def select_features_for_model(self, df: pd.DataFrame, 
                                target_column: str = 'churn_label') -> List[str]:
        """
        Select the most important features for the model.
        
        Args:
            df: DataFrame with all features
            target_column: Name of the target column
            
        Returns:
            List of selected feature names
        """
        # Remove non-feature columns
        exclude_columns = [
            'client_id', 'client_name', 'registration_date', 'last_contact_date',
            'policy_start_date', 'policy_end_date', 'last_claim_date', 'last_payment_date',
            'churn_date', 'churn_reason', 'retention_status', target_column
        ]
        
        # Get all feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        feature_columns = [col for col in feature_columns 
                          if df[col].isnull().sum() / len(df) < missing_threshold]
        
        # Remove constant columns
        feature_columns = [col for col in feature_columns 
                          if df[col].nunique() > 1]
        
        logger.info(f"Selected {len(feature_columns)} features for model training")
        return feature_columns


if __name__ == "__main__":
    # Example usage
    from data_processing.data_loader import create_sample_data
    
    print("Creating sample data and features...")
    sample_data = create_sample_data(1000)
    
    feature_engineer = InsuranceFeatureEngineer()
    features_df = feature_engineer.create_all_features(sample_data)
    
    print(f"Original features: {len(sample_data.columns)}")
    print(f"Engineered features: {len(features_df.columns)}")
    print(f"Churn rate: {features_df['churn_label'].mean():.2%}")
    
    # Show feature categories
    feature_categories = feature_engineer.get_feature_importance_categories()
    for category, features in feature_categories.items():
        print(f"{category}: {len(features)} features")
