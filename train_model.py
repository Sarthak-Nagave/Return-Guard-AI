"""
Enhanced model training script for Flipkart GRiD 7.0 competition
Training Random Forest model with optimized features for return abuse detection
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def load_and_prepare_data():
    """Load and prepare training data"""
    logger.info("Loading training dataset...")
    
    # Load the dataset
    df = pd.read_csv('data/training_data.csv')
    logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Display basic statistics
    logger.info("Dataset info:")
    logger.info(f"- Shape: {df.shape}")
    logger.info(f"- Columns: {list(df.columns)}")
    logger.info(f"- Target distribution: {df['is_abuse'].value_counts().to_dict()}")
    
    return df

def engineer_features(df):
    """Engineer features for the model"""
    logger.info("Engineering features...")
    
    # Create feature columns
    features = df.copy()
    
    # Encode return reason
    le = LabelEncoder()
    features['return_reason_encoded'] = le.fit_transform(features['return_reason'].astype(str))
    
    # Create sentiment categories
    features['sentiment_very_negative'] = (features['return_message_sentiment'] < -0.5).astype(int)
    features['sentiment_negative'] = ((features['return_message_sentiment'] >= -0.5) & 
                                     (features['return_message_sentiment'] < -0.1)).astype(int)
    features['sentiment_neutral'] = ((features['return_message_sentiment'] >= -0.1) & 
                                    (features['return_message_sentiment'] <= 0.1)).astype(int)
    features['sentiment_positive'] = (features['return_message_sentiment'] > 0.1).astype(int)
    
    # Create user behavior features
    features['high_return_user'] = (features['user_return_count'] > 3).astype(int)
    features['fast_approval_seeker'] = (features['return_approval_time'] < 24).astype(int)
    
    # Interaction features
    features['sentiment_count_interaction'] = features['return_message_sentiment'] * features['user_return_count']
    features['image_sentiment_interaction'] = features['image_uploaded'] * features['return_message_sentiment']
    
    # Select final feature set
    feature_columns = [
        'image_uploaded',
        'return_message_sentiment', 
        'user_return_count',
        'return_approval_time',
        'return_reason_encoded',
        'sentiment_very_negative',
        'sentiment_negative',
        'sentiment_neutral',
        'sentiment_positive',
        'high_return_user',
        'fast_approval_seeker',
        'sentiment_count_interaction',
        'image_sentiment_interaction'
    ]
    
    X = features[feature_columns]
    y = features['is_abuse']
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature columns: {feature_columns}")
    
    return X, y, feature_columns

def train_model(X, y):
    """Train Random Forest model with optimized parameters"""
    logger.info("Training Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = rf_model.score(X_test_scaled, y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Model Performance:")
    logger.info(f"- Accuracy: {accuracy:.4f}")
    logger.info(f"- AUC Score: {auc_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    logger.info(f"- Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 Feature Importance:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"- {row['feature']}: {row['importance']:.4f}")
    
    return rf_model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba, feature_importance

def create_visualizations(model, X_test, y_test, y_pred, y_pred_proba, feature_importance):
    """Create model evaluation visualizations"""
    logger.info("Creating visualizations...")
    
    plt.style.use('default')
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abuse'], 
                yticklabels=['Normal', 'Abuse'])
    plt.title('Confusion Matrix - Return Abuse Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Return Abuse Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importance - Return Abuse Detection')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Normal Returns', color='green')
    plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Abusive Returns', color='red')
    plt.xlabel('Abuse Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Abuse Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/plots/probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved to static/plots/")

def save_model_and_metadata(model, scaler, feature_importance):
    """Save trained model and metadata"""
    logger.info("Saving model and metadata...")
    
    # Save model and scaler
    joblib.dump(model, 'models/return_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    # Save training metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'features_used': list(feature_importance['feature']),
        'training_date': datetime.now().isoformat(),
        'model_version': '1.0',
        'competition': 'Flipkart GRiD 7.0'
    }
    
    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Model saved successfully!")
    logger.info("Files created:")
    logger.info("- models/return_model.pkl")
    logger.info("- models/scaler.pkl")
    logger.info("- models/feature_importance.csv")
    logger.info("- models/model_metadata.json")

def main():
    """Main training function"""
    logger.info("Starting Flipkart GRiD 7.0 model training...")
    
    try:
        # Create directories
        create_directories()
        
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Engineer features
        X, y, feature_columns = engineer_features(df)
        
        # Train model
        model, scaler, X_test, y_test, y_pred, y_pred_proba, feature_importance = train_model(X, y)
        
        # Create visualizations
        create_visualizations(model, X_test, y_test, y_pred, y_pred_proba, feature_importance)
        
        # Save model and metadata
        save_model_and_metadata(model, scaler, feature_importance)
        
        logger.info("Training completed successfully!")
        logger.info("Model is ready for deployment in the Return Abuse Detection System")
        
        return {
            'accuracy': model.score(X_test, y_test),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'feature_count': len(feature_columns)
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()