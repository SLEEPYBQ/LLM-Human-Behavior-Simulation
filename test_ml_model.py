#!/usr/bin/env python3
"""
Test the fixed ML model predictions
"""

import sys
import os
import pandas as pd
import json

# Add the project root to the path
sys.path.insert(0, '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP')

from ml_model import LoanApprovalModel
from persona_config import LoanApplication

def test_ml_predictions():
    """Test ML model predictions with sample data"""
    print("=== Testing Fixed ML Model Predictions ===")
    
    # Load the model
    model = LoanApprovalModel()
    model_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl'
    
    if not os.path.exists(model_path):
        print("⚠ Model file not found. Please run 'python ml_model.py' first.")
        return
    
    model.load_model(model_path)
    
    # Load test data
    test_data_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/test_data.csv'
    if not os.path.exists(test_data_path):
        print("⚠ Test data file not found.")
        return
        
    df = pd.read_csv(test_data_path)
    print(f"✓ Loaded {len(df)} test samples")
    print(f"  Columns: {list(df.columns)}")
    
    # Test with first 5 samples
    print("\n=== Testing First 5 Samples ===")
    
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        
        # Create LoanApplication object
        loan_app = LoanApplication(
            loan_id=i,
            no_of_dependents=int(row['no_of_dependents']),
            education=int(row['education']),  # Already encoded
            self_employed=int(row['self_employed']),  # Already encoded
            income_annum=int(row['income_annum']),
            loan_amount=int(row['loan_amount']),
            loan_term=int(row['loan_term']),
            cibil_score=int(row['cibil_score']),
            residential_assets_value=int(row['residential_assets_value']),
            commercial_assets_value=int(row['commercial_assets_value']),
            luxury_assets_value=int(row['luxury_assets_value']),
            bank_asset_value=int(row['bank_asset_value'])
        )
        
        # Test prediction
        try:
            prediction = model.predict_single(loan_app.to_dict())
            true_label = "Approved" if row['true_loan_status'] == 1 else "Rejected"
            
            print(f"\nSample {i+1}:")
            print(f"  True Label: {true_label}")
            print(f"  AI Prediction: '{prediction['prediction']}'")
            print(f"  Confidence: {prediction['confidence']:.3f}")
            print(f"  Probability Approved: {prediction['probability_approved']:.3f}")
            print(f"  Match: {'✓' if prediction['prediction'] == true_label else '✗'}")
            
        except Exception as e:
            print(f"\nSample {i+1}: ERROR - {e}")
    
    # Test with raw dictionary data
    print("\n=== Testing Raw Dictionary Input ===")
    
    test_loan_dict = {
        'no_of_dependents': 2,
        'education': 1,  # Already encoded
        'self_employed': 0,  # Already encoded
        'income_annum': 5000000,
        'loan_amount': 10000000,
        'loan_term': 15,
        'cibil_score': 750,
        'residential_assets_value': 8000000,
        'commercial_assets_value': 2000000,
        'luxury_assets_value': 5000000,
        'bank_asset_value': 3000000
    }
    
    try:
        prediction = model.predict_single(test_loan_dict)
        print(f"Dictionary Input Test:")
        print(f"  AI Prediction: '{prediction['prediction']}'")
        print(f"  Confidence: {prediction['confidence']:.3f}")
        print(f"  Probability Approved: {prediction['probability_approved']:.3f}")
        print("✓ Dictionary input working correctly")
        
    except Exception as e:
        print(f"Dictionary Input Test: ERROR - {e}")
    
    # Calculate overall accuracy on test set
    print("\n=== Overall Test Set Performance ===")
    
    correct_predictions = 0
    total_predictions = 0
    predictions_list = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        loan_dict = {
            'no_of_dependents': int(row['no_of_dependents']),
            'education': int(row['education']),
            'self_employed': int(row['self_employed']),
            'income_annum': int(row['income_annum']),
            'loan_amount': int(row['loan_amount']),
            'loan_term': int(row['loan_term']),
            'cibil_score': int(row['cibil_score']),
            'residential_assets_value': int(row['residential_assets_value']),
            'commercial_assets_value': int(row['commercial_assets_value']),
            'luxury_assets_value': int(row['luxury_assets_value']),
            'bank_asset_value': int(row['bank_asset_value'])
        }
        
        try:
            prediction = model.predict_single(loan_dict)
            true_label = "Approved" if row['true_loan_status'] == 1 else "Rejected"
            
            predictions_list.append({
                'sample': i,
                'true_label': true_label,
                'predicted_label': prediction['prediction'],
                'confidence': prediction['confidence'],
                'match': prediction['prediction'] == true_label
            })
            
            if prediction['prediction'] == true_label:
                correct_predictions += 1
            total_predictions += 1
            
        except Exception as e:
            print(f"Sample {i}: Prediction failed - {e}")
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"Test Set Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        # Show prediction distribution
        approved_predictions = sum(1 for p in predictions_list if p['predicted_label'] == 'Approved')
        rejected_predictions = sum(1 for p in predictions_list if p['predicted_label'] == 'Rejected')
        
        print(f"Prediction Distribution:")
        print(f"  Approved: {approved_predictions}")
        print(f"  Rejected: {rejected_predictions}")
        
        # Show confidence distribution
        confidences = [p['confidence'] for p in predictions_list]
        print(f"Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"Average Confidence: {sum(confidences)/len(confidences):.3f}")
        
    print("\n✓ ML Model testing completed successfully!")

if __name__ == "__main__":
    test_ml_predictions()