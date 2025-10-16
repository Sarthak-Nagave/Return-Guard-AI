"""
Utility functions for the Return Abuse Detection System
"""
import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_risk_score(abuse_probability):
    """Calculate risk score and recommendation based on abuse probability"""
    # Convert probability to percentage risk score
    risk_score = int(abuse_probability * 100)
    
    # Determine recommendation based on optimized thresholds for competition
    if risk_score >= 75:
        recommendation = "REJECT"
    elif risk_score >= 35:
        recommendation = "MANUAL_REVIEW"
    else:
        recommendation = "APPROVE"
    
    return risk_score, recommendation


def log_to_csv(case_data):
    """Log all predictions to CSV file for competition analysis"""
    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        csv_file = 'logs/abuse_logs.csv'
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'user_id', 'order_id', 'product_id', 'return_reason',
                'sentiment', 'user_return_count', 'abuse_probability', 'risk_score',
                'prediction', 'action', 'is_abuse'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(case_data)
            
    except Exception as e:
        logging.error(f"Failed to log to CSV: {str(e)}")


def get_csv_logs():
    """Read all logs from CSV file"""
    try:
        csv_file = 'logs/abuse_logs.csv'
        logs = []
        
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    logs.append(row)
        
        return logs
    
    except Exception as e:
        logging.error(f"Failed to read CSV logs: {str(e)}")
        return []

def log_abuse_case(case_data):
    """Log abuse cases to file for audit trail"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Prepare log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': case_data.get('user_id'),
            'order_id': case_data.get('order_id'),
            'product_id': case_data.get('product_id'),
            'return_reason': case_data.get('return_reason'),
            'abuse_probability': case_data.get('abuse_probability'),
            'risk_score': case_data.get('risk_score'),
            'recommendation': case_data.get('recommendation'),
            'image_uploaded': case_data.get('image_uploaded'),
            'user_return_count': case_data.get('user_return_count'),
            'return_message_sentiment': case_data.get('return_message_sentiment')
        }
        
        # Write to log file
        with open('logs/abuse_cases.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Logged abuse case for user {case_data.get('user_id')}")
        
    except Exception as e:
        logger.error(f"Error logging abuse case: {str(e)}")

def get_abuse_statistics():
    """Get statistics from logged abuse cases"""
    try:
        if not os.path.exists('logs/abuse_cases.log'):
            return {'total_cases': 0, 'recommendations': {}}
        
        total_cases = 0
        recommendations = {'APPROVE': 0, 'MANUAL_REVIEW': 0, 'REJECT': 0}
        
        with open('logs/abuse_cases.log', 'r') as f:
            for line in f:
                if line.strip():
                    case = json.loads(line.strip())
                    total_cases += 1
                    recommendation = case.get('recommendation', 'UNKNOWN')
                    if recommendation in recommendations:
                        recommendations[recommendation] += 1
        
        return {
            'total_cases': total_cases,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting abuse statistics: {str(e)}")
        return {'total_cases': 0, 'recommendations': {}}

def validate_form_data(form_data):
    """Validate form input data"""
    errors = []
    
    # Required fields
    required_fields = ['user_id', 'order_id', 'product_id', 'return_reason']
    for field in required_fields:
        if not form_data.get(field):
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    # Numeric validations
    try:
        sentiment = float(form_data.get('return_message_sentiment', 0))
        if sentiment < -1 or sentiment > 1:
            errors.append("Return message sentiment must be between -1 and 1")
    except ValueError:
        errors.append("Return message sentiment must be a valid number")
    
    try:
        return_count = int(form_data.get('user_return_count', 0))
        if return_count < 0:
            errors.append("User return count cannot be negative")
    except ValueError:
        errors.append("User return count must be a valid integer")
    
    try:
        approval_time = int(form_data.get('return_approval_time', 0))
        if approval_time < 0:
            errors.append("Return approval time cannot be negative")
    except ValueError:
        errors.append("Return approval time must be a valid integer")
    
    return errors

def format_recommendation_color(recommendation):
    """Get Bootstrap color class for recommendation"""
    color_map = {
        'APPROVE': 'success',
        'MANUAL_REVIEW': 'warning',
        'REJECT': 'danger'
    }
    return color_map.get(recommendation, 'secondary')

def format_risk_score_color(risk_score):
    """Get Bootstrap color class for risk score"""
    if risk_score < 30:
        return 'success'
    elif risk_score < 60:
        return 'warning'
    else:
        return 'danger'

def log_escalated_case(case_data):
    """Log escalated cases to CSV file for manager review"""
    try:
        os.makedirs('logs', exist_ok=True)
        filepath = 'logs/escalated_cases.csv'
        
        # Check if file exists to write header
        file_exists = os.path.exists(filepath)
        
        with open(filepath, 'a', newline='') as csvfile:
            fieldnames = [
                'user_id', 'order_id', 'product_id', 'return_reason',
                'abuse_probability', 'risk_score', 'escalated_at', 'escalated_by'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(case_data)
            
    except Exception as e:
        print(f"Error logging escalated case: {str(e)}")

def get_escalated_cases():
    """Read escalated cases from CSV file"""
    try:
        filepath = 'logs/escalated_cases.csv'
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except Exception as e:
        print(f"Error reading escalated cases: {str(e)}")
        return []
