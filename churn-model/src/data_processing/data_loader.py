"""
Data loading and preprocessing module for churn prediction model.
Handles insurance client data from SQL Server with proper data cleaning and validation.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """
    Data loader for insurance client data with churn prediction focus.
    Handles data extraction, cleaning, and preprocessing for ML model.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize data loader with database connection.
        
        Args:
            connection_string: SQL Server connection string
        """
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        
    def load_client_data(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load comprehensive client data for churn prediction.
        
        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
            
        Returns:
            DataFrame with client data and features
        """
        try:
            # Default to last 2 years if no dates provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=730)
                
            query = """
            SELECT 
                c.client_id,
                c.client_name,
                c.registration_date,
                c.client_type,
                c.industry,
                c.annual_revenue,
                c.employee_count,
                c.contact_frequency,
                c.last_contact_date,
                c.engagement_score,
                c.satisfaction_rating,
                c.retention_status,
                c.churn_date,
                c.churn_reason,
                -- Policy features
                p.policy_id,
                p.policy_type,
                p.premium_amount,
                p.coverage_amount,
                p.policy_start_date,
                p.policy_end_date,
                p.renewal_count,
                p.premium_changes,
                p.coverage_changes,
                -- Claim features
                cl.claim_count,
                cl.total_claim_amount,
                cl.claim_frequency,
                cl.last_claim_date,
                cl.claim_severity_score,
                -- Payment features
                pay.payment_frequency,
                pay.payment_delays,
                pay.payment_amount,
                pay.last_payment_date,
                pay.payment_method,
                -- Engagement features
                eng.email_opens,
                eng.email_clicks,
                eng.website_visits,
                eng.support_tickets,
                eng.meeting_attendance,
                eng.communication_score
            FROM clients c
            LEFT JOIN policies p ON c.client_id = p.client_id
            LEFT JOIN claims cl ON c.client_id = cl.client_id
            LEFT JOIN payments pay ON c.client_id = pay.client_id
            LEFT JOIN engagement eng ON c.client_id = eng.client_id
            WHERE c.registration_date BETWEEN ? AND ?
            ORDER BY c.client_id, p.policy_start_date DESC
            """
            
            df = pd.read_sql(query, self.engine, params=[start_date, end_date])
            logger.info(f"Loaded {len(df)} client records from {start_date} to {end_date}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading client data: {str(e)}")
            raise
    
    def create_churn_labels(self, df: pd.DataFrame, 
                           churn_window_days: int = 90) -> pd.DataFrame:
        """
        Create churn labels based on client retention status and dates.
        
        Args:
            df: Client data DataFrame
            churn_window_days: Days to look ahead for churn prediction
            
        Returns:
            DataFrame with churn labels
        """
        # Create churn label based on retention status and dates
        current_date = datetime.now()
        
        # Method 1: Use existing churn_date if available
        df['churn_label'] = 0
        df.loc[df['churn_date'].notna(), 'churn_label'] = 1
        
        # Method 2: Use retention_status
        df.loc[df['retention_status'] == 'churned', 'churn_label'] = 1
        
        # Method 3: Use policy end dates (if no renewal within window)
        policy_end_threshold = current_date - timedelta(days=churn_window_days)
        df.loc[(df['policy_end_date'] < policy_end_threshold) & 
               (df['renewal_count'] == 0), 'churn_label'] = 1
        
        # Remove clients who churned before the prediction window
        df = df[df['churn_date'].isna() | 
                (df['churn_date'] > current_date - timedelta(days=churn_window_days))]
        
        logger.info(f"Created churn labels: {df['churn_label'].sum()} churned out of {len(df)} clients")
        logger.info(f"Churn rate: {df['churn_label'].mean():.2%}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess client data for ML model.
        
        Args:
            df: Raw client data DataFrame
            
        Returns:
            Cleaned DataFrame ready for feature engineering
        """
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['client_id'])
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert date columns
        date_columns = ['registration_date', 'last_contact_date', 'policy_start_date', 
                       'policy_end_date', 'last_claim_date', 'last_payment_date', 'churn_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['annual_revenue', 'employee_count', 'premium_amount', 
                          'coverage_amount', 'claim_count', 'total_claim_amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Data cleaning completed. Final dataset: {len(df)} records")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        
        # For engagement scores, use median
        engagement_cols = ['engagement_score', 'satisfaction_rating', 'communication_score']
        for col in engagement_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # For claim-related features, fill with 0 (no claims)
        claim_cols = ['claim_count', 'total_claim_amount', 'claim_frequency']
        for col in claim_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # For payment features, use forward fill
        payment_cols = ['payment_frequency', 'payment_delays', 'payment_amount']
        for col in payment_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # For categorical features, use mode
        categorical_cols = ['client_type', 'industry', 'policy_type', 'payment_method']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate data summary statistics for model development.
        
        Args:
            df: Processed client data DataFrame
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            'total_clients': len(df),
            'churn_rate': df['churn_label'].mean(),
            'churned_clients': df['churn_label'].sum(),
            'retained_clients': (df['churn_label'] == 0).sum(),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'date_range': {
                'earliest_registration': df['registration_date'].min(),
                'latest_registration': df['registration_date'].max()
            },
            'feature_distributions': {}
        }
        
        # Add feature distributions for key numeric features
        numeric_features = ['annual_revenue', 'premium_amount', 'engagement_score', 'claim_count']
        for feature in numeric_features:
            if feature in df.columns:
                summary['feature_distributions'][feature] = {
                    'mean': df[feature].mean(),
                    'median': df[feature].median(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max()
                }
        
        return summary


def create_sample_data(n_clients: int = 1000) -> pd.DataFrame:
    """
    Create sample insurance client data for testing and development.
    
    Args:
        n_clients: Number of sample clients to generate
        
    Returns:
        DataFrame with sample client data
    """
    np.random.seed(42)
    
    # Generate sample data
    data = {
        'client_id': range(1, n_clients + 1),
        'client_name': [f'Client_{i}' for i in range(1, n_clients + 1)],
        'registration_date': pd.date_range('2020-01-01', periods=n_clients, freq='D'),
        'client_type': np.random.choice(['Corporate', 'SME', 'Individual'], n_clients, p=[0.3, 0.5, 0.2]),
        'industry': np.random.choice(['Manufacturing', 'Technology', 'Healthcare', 'Finance', 'Retail'], n_clients),
        'annual_revenue': np.random.lognormal(12, 1, n_clients),
        'employee_count': np.random.poisson(50, n_clients),
        'contact_frequency': np.random.poisson(4, n_clients),
        'last_contact_date': pd.date_range('2023-01-01', periods=n_clients, freq='D'),
        'engagement_score': np.random.uniform(1, 10, n_clients),
        'satisfaction_rating': np.random.uniform(1, 5, n_clients),
        'retention_status': np.random.choice(['active', 'churned'], n_clients, p=[0.85, 0.15]),
        'churn_date': [None] * n_clients,
        'churn_reason': [None] * n_clients,
        'premium_amount': np.random.lognormal(8, 1, n_clients),
        'coverage_amount': np.random.lognormal(14, 1, n_clients),
        'policy_start_date': pd.date_range('2020-01-01', periods=n_clients, freq='D'),
        'policy_end_date': pd.date_range('2024-01-01', periods=n_clients, freq='D'),
        'renewal_count': np.random.poisson(2, n_clients),
        'premium_changes': np.random.normal(0, 0.1, n_clients),
        'coverage_changes': np.random.normal(0, 0.05, n_clients),
        'claim_count': np.random.poisson(1, n_clients),
        'total_claim_amount': np.random.lognormal(6, 2, n_clients),
        'claim_frequency': np.random.uniform(0, 5, n_clients),
        'last_claim_date': pd.date_range('2023-01-01', periods=n_clients, freq='D'),
        'claim_severity_score': np.random.uniform(1, 10, n_clients),
        'payment_frequency': np.random.choice(['Monthly', 'Quarterly', 'Annual'], n_clients),
        'payment_delays': np.random.poisson(1, n_clients),
        'payment_amount': np.random.lognormal(8, 1, n_clients),
        'last_payment_date': pd.date_range('2023-01-01', periods=n_clients, freq='D'),
        'payment_method': np.random.choice(['Bank Transfer', 'Credit Card', 'Check'], n_clients),
        'email_opens': np.random.poisson(10, n_clients),
        'email_clicks': np.random.poisson(3, n_clients),
        'website_visits': np.random.poisson(5, n_clients),
        'support_tickets': np.random.poisson(1, n_clients),
        'meeting_attendance': np.random.poisson(2, n_clients),
        'communication_score': np.random.uniform(1, 10, n_clients)
    }
    
    df = pd.DataFrame(data)
    
    # Add some churn patterns
    churn_mask = df['retention_status'] == 'churned'
    df.loc[churn_mask, 'churn_date'] = df.loc[churn_mask, 'last_contact_date'] + pd.Timedelta(days=30)
    df.loc[churn_mask, 'churn_reason'] = np.random.choice(['Price', 'Service', 'Coverage', 'Competitor'], churn_mask.sum())
    
    # Create churn labels
    df['churn_label'] = (df['retention_status'] == 'churned').astype(int)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Creating sample data for testing...")
    sample_data = create_sample_data(1000)
    print(f"Sample data created: {len(sample_data)} clients")
    print(f"Churn rate: {sample_data['churn_label'].mean():.2%}")
    print("\nData summary:")
    print(sample_data.describe())
