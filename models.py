"""
Data models for the Return Abuse Detection System
"""
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class ReturnRequestDB(db.Model):
    """Database model for return request data"""
    __tablename__ = 'return_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    order_id = db.Column(db.String(50), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    order_date = db.Column(db.String(50))
    return_date = db.Column(db.String(50))
    return_reason = db.Column(db.String(200))
    image_uploaded = db.Column(db.Integer, default=0)
    return_message_sentiment = db.Column(db.Float, default=0.0)
    user_return_count = db.Column(db.Integer, default=0)
    return_approval_time = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, user_id, order_id, product_id, order_date=None, return_date=None, 
                 return_reason="", image_uploaded=0, return_message_sentiment=0.0, 
                 user_return_count=0, return_approval_time=0):
        self.user_id = user_id
        self.order_id = order_id
        self.product_id = product_id
        self.order_date = order_date
        self.return_date = return_date
        self.return_reason = return_reason
        self.image_uploaded = image_uploaded
        self.return_message_sentiment = return_message_sentiment
        self.user_return_count = user_return_count
        self.return_approval_time = return_approval_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML processing"""
        return {
            'user_id': self.user_id,
            'order_id': self.order_id,
            'product_id': self.product_id,
            'order_date': self.order_date,
            'return_date': self.return_date,
            'return_reason': self.return_reason,
            'image_uploaded': self.image_uploaded,
            'return_message_sentiment': self.return_message_sentiment,
            'user_return_count': self.user_return_count,
            'return_approval_time': self.return_approval_time
        }


class PredictionResultDB(db.Model):
    """Database model for prediction results"""
    __tablename__ = 'prediction_results'
    
    id = db.Column(db.Integer, primary_key=True)
    return_request_id = db.Column(db.Integer, db.ForeignKey('return_requests.id'), nullable=False)
    abuse_probability = db.Column(db.Float, nullable=False)
    is_abuse = db.Column(db.Boolean, nullable=False)
    risk_score = db.Column(db.Integer, nullable=False)
    recommendation = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    return_request = db.relationship('ReturnRequestDB', backref='predictions')
    
    def __init__(self, return_request_id, abuse_probability, is_abuse, risk_score, recommendation):
        self.return_request_id = return_request_id
        self.abuse_probability = abuse_probability
        self.is_abuse = is_abuse
        self.risk_score = risk_score
        self.recommendation = recommendation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'return_request_id': self.return_request_id,
            'abuse_probability': self.abuse_probability,
            'is_abuse': self.is_abuse,
            'risk_score': self.risk_score,
            'recommendation': self.recommendation,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ReturnRequest:
    """Model for return request data"""
    user_id: str
    order_id: str
    product_id: str
    order_date: Optional[str] = None
    return_date: Optional[str] = None
    return_reason: str = ""
    image_uploaded: int = 0
    return_message_sentiment: float = 0.0
    user_return_count: int = 0
    return_approval_time: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML processing"""
        return {
            'user_id': self.user_id,
            'order_id': self.order_id,
            'product_id': self.product_id,
            'order_date': self.order_date,
            'return_date': self.return_date,
            'return_reason': self.return_reason,
            'image_uploaded': self.image_uploaded,
            'return_message_sentiment': self.return_message_sentiment,
            'user_return_count': self.user_return_count,
            'return_approval_time': self.return_approval_time
        }

@dataclass
class PredictionResult:
    """Model for prediction results"""
    abuse_probability: float
    is_abuse: bool
    risk_score: int
    recommendation: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'abuse_probability': self.abuse_probability,
            'is_abuse': self.is_abuse,
            'risk_score': self.risk_score,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp.isoformat()
        }

# Return reason mappings for encoding
RETURN_REASONS = [
    'wrong color', 'poor quality', 'item damaged during delivery', 'received late',
    'unwanted gift', 'wrong item delivered', 'not satisfied with the product',
    'product not as described', 'changed mind', 'duplicate order', 'defective product',
    'accidental purchase', 'size issue', 'incorrect quantity', 'missing accessories'
]

RETURN_REASON_ENCODING = {reason: idx for idx, reason in enumerate(RETURN_REASONS)}

# Risk level thresholds
RISK_THRESHOLDS = {
    'LOW': 0.3,
    'MEDIUM': 0.6,
    'HIGH': 0.8
}

# Recommendation mappings
RECOMMENDATIONS = {
    'LOW': 'APPROVE',
    'MEDIUM': 'MANUAL_REVIEW',
    'HIGH': 'REJECT'
}
