import os
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from ml_pipeline import MLPipeline
from models import db, ReturnRequestDB, PredictionResultDB
from utils import log_abuse_case, log_to_csv, get_csv_logs, calculate_risk_score, log_escalated_case

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "flipkart-grid-7-return-abuse-detection-2025")

# Database configuration
database_url = os.environ.get("DATABASE_URL")
if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
else:
    # Fallback for development
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///return_abuse.db"
    
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)

# Initialize ML Pipeline
ml_pipeline = MLPipeline()

# Create database tables
with app.app_context():
    db.create_all()

# Load trained model on startup
try:
    model = joblib.load('models/return_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("Production model loaded successfully")
except FileNotFoundError:
    logger.warning("Model not found. Please train the model first by running train_model.py")
    model = None
    scaler = None

@app.route('/')
def index():
    """Main dashboard for return abuse detection"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict return abuse probability"""
    try:
        if model is None or scaler is None:
            flash('Model not loaded. Please train the model first.', 'error')
            return redirect(url_for('index'))
        
        # Extract form data
        form_data = {
            'user_id': request.form.get('user_id'),
            'order_id': request.form.get('order_id'),
            'product_id': request.form.get('product_id'),
            'return_reason': request.form.get('return_reason'),
            'image_uploaded': int(request.form.get('image_uploaded', 0)),
            'return_message_sentiment': float(request.form.get('return_message_sentiment', 0)),
            'user_return_count': int(request.form.get('user_return_count', 0)),
            'return_approval_time': int(request.form.get('return_approval_time', 0))
        }
        
        # Create feature vector matching the trained model
        feature_vector = [
            form_data['image_uploaded'],
            form_data['return_message_sentiment'],
            form_data['user_return_count'],
            form_data['return_approval_time'],
            # Encode return reason (simple mapping for demo)
            hash(form_data['return_reason']) % 10,  # Simple categorical encoding
            1 if form_data['return_message_sentiment'] < -0.5 else 0,  # Very negative sentiment
            form_data['return_message_sentiment'] * form_data['user_return_count']  # Interaction feature
        ]
        
        # Scale features
        features_scaled = scaler.transform([feature_vector])
        
        # Make prediction
        abuse_probability = model.predict_proba(features_scaled)[0][1]
        prediction = model.predict(features_scaled)[0]
        
        # Calculate risk score and recommendation
        risk_score, recommendation = calculate_risk_score(abuse_probability)
        
        # Save return request to database
        return_request = ReturnRequestDB(
            user_id=form_data['user_id'],
            order_id=form_data['order_id'],
            product_id=form_data['product_id'],
            return_reason=form_data['return_reason'],
            image_uploaded=form_data['image_uploaded'],
            return_message_sentiment=form_data['return_message_sentiment'],
            user_return_count=form_data['user_return_count'],
            return_approval_time=form_data['return_approval_time']
        )
        db.session.add(return_request)
        db.session.commit()
        
        # Save prediction result to database
        prediction_result = PredictionResultDB(
            return_request_id=return_request.id,
            abuse_probability=float(abuse_probability),
            is_abuse=bool(prediction),
            risk_score=int(risk_score),
            recommendation=str(recommendation)
        )
        db.session.add(prediction_result)
        db.session.commit()
        
        # Log all cases to CSV for competition analysis
        log_to_csv({
            'timestamp': datetime.now().isoformat(),
            'user_id': form_data['user_id'],
            'order_id': form_data['order_id'],
            'product_id': form_data['product_id'],
            'return_reason': form_data['return_reason'],
            'sentiment': form_data['return_message_sentiment'],
            'user_return_count': form_data['user_return_count'],
            'abuse_probability': float(abuse_probability),
            'risk_score': int(risk_score),
            'prediction': 'Abusive' if prediction == 1 else 'Normal',
            'action': str(recommendation),
            'is_abuse': bool(prediction)
        })
        
        # Log if abuse is detected
        if prediction == 1 or abuse_probability > 0.7:
            log_abuse_case({
                **form_data,
                'abuse_probability': abuse_probability,
                'risk_score': risk_score,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            })
        
        return render_template('results.html', 
                             prediction=prediction,
                             abuse_probability=abuse_probability,
                             risk_score=risk_score,
                             recommendation=recommendation,
                             form_data=form_data)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Create feature vector matching the trained model
        feature_vector = [
            data.get('image_uploaded', 0),
            data.get('return_message_sentiment', 0),
            data.get('user_return_count', 0),
            data.get('return_approval_time', 0),
            hash(data.get('return_reason', '')) % 10,
            1 if data.get('return_message_sentiment', 0) < -0.5 else 0,
            data.get('return_message_sentiment', 0) * data.get('user_return_count', 0)
        ]
        
        features_scaled = scaler.transform([feature_vector])
        
        abuse_probability = model.predict_proba(features_scaled)[0][1]
        prediction = model.predict(features_scaled)[0]
        risk_score, recommendation = calculate_risk_score(abuse_probability)
        
        return jsonify({
            'prediction': int(prediction),
            'abuse_probability': float(abuse_probability),
            'risk_score': risk_score,
            'recommendation': recommendation
        })
    
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Trigger model training"""
    try:
        from train_model import main
        metrics = main()
        flash(f'Model trained successfully! Accuracy: {metrics["accuracy"]:.3f}', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        flash(f'Error training model: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/logs')
def view_logs():
    """View abuse detection logs from CSV file for competition analysis"""
    try:
        # Get logs from CSV file
        logs = get_csv_logs()
        
        # Also get database logs for comprehensive view
        try:
            predictions = db.session.query(PredictionResultDB).join(ReturnRequestDB).order_by(PredictionResultDB.created_at.desc()).limit(50).all()
            
            for pred in predictions:
                log_entry = {
                    'timestamp': pred.created_at.isoformat(),
                    'user_id': pred.return_request.user_id,
                    'order_id': pred.return_request.order_id,
                    'product_id': pred.return_request.product_id,
                    'return_reason': pred.return_request.return_reason,
                    'sentiment': str(pred.return_request.return_message_sentiment),
                    'user_return_count': str(pred.return_request.user_return_count),
                    'abuse_probability': str(pred.abuse_probability),
                    'risk_score': str(pred.risk_score),
                    'prediction': 'Abusive' if pred.is_abuse else 'Normal',
                    'action': pred.recommendation,
                    'is_abuse': str(pred.is_abuse).lower()
                }
                logs.append(log_entry)
        except:
            pass  # If database fails, just use CSV logs
        
        return render_template('logs.html', logs=logs)
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        flash(f'Error reading logs: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/database')
def view_database():
    """View all database records"""
    try:
        return_requests = ReturnRequestDB.query.order_by(ReturnRequestDB.created_at.desc()).limit(50).all()
        predictions = PredictionResultDB.query.order_by(PredictionResultDB.created_at.desc()).limit(50).all()
        
        return render_template('database.html', 
                             return_requests=return_requests,
                             predictions=predictions)
    except Exception as e:
        logger.error(f"Error reading database: {str(e)}")
        flash(f'Error reading database: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/escalate', methods=['POST'])
def escalate_case():
    """Escalate a rejected return case to manager"""
    try:
        # Get case data from request
        data = request.json if request.is_json else request.form
        
        case_info = {
            'user_id': data.get('user_id'),
            'order_id': data.get('order_id'),
            'product_id': data.get('product_id'),
            'return_reason': data.get('return_reason'),
            'abuse_probability': data.get('abuse_probability'),
            'risk_score': data.get('risk_score'),
            'escalated_at': datetime.now().isoformat(),
            'escalated_by': 'fraud_analyst'
        }
        
        # Log to escalated cases CSV
        log_escalated_case(case_info)
        
        logger.info(f"Case escalated to manager for user {case_info['user_id']}")
        
        if request.is_json:
            return jsonify({
                'success': True,
                'message': 'Case successfully escalated to manager for review'
            })
        else:
            flash('Case successfully escalated to manager for review', 'success')
            return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Escalation error: {str(e)}")
        if request.is_json:
            return jsonify({
                'success': False,
                'error': f'Failed to escalate case: {str(e)}'
            }), 500
        else:
            flash(f'Failed to escalate case: {str(e)}', 'error')
            return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
