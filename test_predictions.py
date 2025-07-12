import pandas as pd
from ml_model import LoanApprovalModel

def test_model_predictions():
    # Load the trained model
    model = LoanApprovalModel()
    model.load_model('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl')
    
    # Check label encodings
    print("=== Label Encoder Mappings ===")
    for key, encoder in model.label_encoders.items():
        print(f"{key}:")
        for i, class_name in enumerate(encoder.classes_):
            print(f"  {i} -> {class_name}")
        print()
    
    # Load a few test cases to verify predictions
    test_data = pd.read_csv('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/experiment_cases.csv')
    
    print("=== Testing Model Predictions ===")
    print("Checking first 5 cases:")
    print()
    
    for i in range(5):
        row = test_data.iloc[i]
        case_id = row['case_id']
        
        # Prepare data for prediction
        loan_data = {
            'no_of_dependents': row['no_of_dependents'],
            'education': row['education'], 
            'self_employed': row['self_employed'],
            'income_annum': row['income_annum'],
            'loan_amount': row['loan_amount'],
            'loan_term': row['loan_term'],
            'cibil_score': row['cibil_score'],
            'residential_assets_value': row['residential_assets_value'],
            'commercial_assets_value': row['commercial_assets_value'],
            'luxury_assets_value': row['luxury_assets_value'],
            'bank_asset_value': row['bank_asset_value']
        }
        
        # Get model prediction
        pred_result = model.predict_single(loan_data)
        
        print(f"Case {case_id}:")
        print(f"  True status: {row['true_loan_status']} ({row['true_loan_status_string']})")
        print(f"  Model prediction: {pred_result['prediction']}")
        print(f"  Confidence: {pred_result['confidence']:.3f}")
        print(f"  Probability approved: {pred_result['probability_approved']:.3f}")
        
        # Convert to binary for comparison
        pred_binary = 1 if pred_result['prediction'] == 'Approved' else 0
        print(f"  Binary prediction: {pred_binary}")
        print(f"  CSV model_prediction: {row['model_prediction']}")
        print(f"  Match: {pred_binary == row['model_prediction']}")
        print(f"  Correct: {row['is_correct']}")
        print("-" * 50)

if __name__ == "__main__":
    test_model_predictions()