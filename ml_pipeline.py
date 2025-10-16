"""
Machine Learning Pipeline for Return Abuse Detection
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import logging
from models import RETURN_REASON_ENCODING

logger = logging.getLogger(__name__)

class MLPipeline:
    """Complete ML pipeline for return abuse detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath):
        """Load and validate dataset"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Handle missing values
        data['return_message_sentiment'].fillna(0, inplace=True)
        data['user_return_count'].fillna(0, inplace=True)
        data['return_approval_time'].fillna(data['return_approval_time'].median(), inplace=True)
        
        # Convert dates to datetime
        if 'order_date' in data.columns:
            data['order_date'] = pd.to_datetime(data['order_date'])
        if 'return_date' in data.columns:
            data['return_date'] = pd.to_datetime(data['return_date'])
        
        # Encode categorical variables
        if 'return_reason' in data.columns:
            data['return_reason_encoded'] = data['return_reason'].map(RETURN_REASON_ENCODING)
            data['return_reason_encoded'].fillna(-1, inplace=True)
        
        logger.info("Data preprocessing completed")
        return data
    
    def engineer_features(self, data):
        """Engineer features for ML model"""
        if isinstance(data, dict):
            # Single prediction case
            features = []
            
            # Basic features
            features.extend([
                data.get('image_uploaded', 0),
                data.get('return_message_sentiment', 0),
                data.get('user_return_count', 0),
                data.get('return_approval_time', 0)
            ])
            
            # Encoded return reason
            reason_encoded = RETURN_REASON_ENCODING.get(data.get('return_reason', ''), -1)
            features.append(reason_encoded)
            
            # Derived features
            features.extend([
                1 if data.get('return_message_sentiment', 0) < -0.5 else 0,  # Very negative sentiment
                1 if data.get('user_return_count', 0) > 3 else 0,  # Frequent returner
                1 if data.get('return_approval_time', 0) > 10 else 0,  # Long approval time
                data.get('return_message_sentiment', 0) * data.get('user_return_count', 0),  # Interaction
                1 if reason_encoded in [0, 2, 6, 7] else 0  # Suspicious reasons
            ])
            
            return features
        
        else:
            # Batch processing case
            features_df = pd.DataFrame()
            
            # Basic features
            features_df['image_uploaded'] = data['image_uploaded']
            features_df['return_message_sentiment'] = data['return_message_sentiment']
            features_df['user_return_count'] = data['user_return_count']
            features_df['return_approval_time'] = data['return_approval_time']
            
            # Encoded return reason
            features_df['return_reason_encoded'] = data['return_reason_encoded']
            
            # Derived features
            features_df['very_negative_sentiment'] = (data['return_message_sentiment'] < -0.5).astype(int)
            features_df['frequent_returner'] = (data['user_return_count'] > 3).astype(int)
            features_df['long_approval_time'] = (data['return_approval_time'] > 10).astype(int)
            features_df['sentiment_count_interaction'] = data['return_message_sentiment'] * data['user_return_count']
            features_df['suspicious_reason'] = data['return_reason_encoded'].isin([0, 2, 6, 7]).astype(int)
            
            self.feature_names = features_df.columns.tolist()
            return features_df
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the ML model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        
        # Train model
        logger.info(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        logger.info(f"Model Performance:")
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"Precision: {metrics['precision']:.3f}")
        logger.info(f"Recall: {metrics['recall']:.3f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.3f}")
        
        # Feature importance for Random Forest
        if model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 5 Important Features:")
            for _, row in feature_importance.head().iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.3f}")
        
        return metrics, X_test_scaled, y_test, y_pred_proba
    
    def save_model(self, model_path='models/abuse_detection_model.pkl', 
                   scaler_path='models/feature_scaler.pkl'):
        """Save trained model and scaler"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='models/abuse_detection_model.pkl',
                   scaler_path='models/feature_scaler.pkl'):
        """Load trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Model and scaler loaded successfully")
