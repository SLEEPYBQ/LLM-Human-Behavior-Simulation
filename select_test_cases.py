import pandas as pd
import numpy as np
from ml_model import LoanApprovalModel

def select_test_cases():
    # Load the test data
    test_data = pd.read_csv('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/test_data.csv')
    
    # Load the trained model
    model = LoanApprovalModel()
    model.load_model('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl')
    
    # Get model predictions for all test cases
    predictions = []
    for idx, row in test_data.iterrows():
        # Prepare data for prediction (exclude target columns)
        loan_data = row.drop(['true_loan_status_string', 'true_loan_status']).to_dict()
        pred_result = model.predict_single(loan_data)
        
        # Convert prediction to binary (1=Approved, 0=Rejected)
        pred_binary = 1 if pred_result['prediction'] == 'Approved' else 0
        predictions.append(pred_binary)
    
    # Add predictions to dataframe
    test_data['model_prediction'] = predictions
    
    # Identify correct and incorrect predictions
    test_data['is_correct'] = (test_data['model_prediction'] == test_data['true_loan_status'])
    
    # Separate correct and incorrect predictions
    correct_predictions = test_data[test_data['is_correct'] == True]
    incorrect_predictions = test_data[test_data['is_correct'] == False]
    
    print(f"Total test cases: {len(test_data)}")
    print(f"Correct predictions: {len(correct_predictions)}")
    print(f"Incorrect predictions: {len(incorrect_predictions)}")
    
    # Sample 28 correct and 12 incorrect cases
    np.random.seed(42)  # For reproducibility
    
    if len(correct_predictions) >= 28:
        selected_correct = correct_predictions.sample(n=28, random_state=42)
    else:
        selected_correct = correct_predictions
        print(f"Warning: Only {len(correct_predictions)} correct predictions available")
    
    if len(incorrect_predictions) >= 12:
        selected_incorrect = incorrect_predictions.sample(n=12, random_state=42)
    else:
        selected_incorrect = incorrect_predictions
        print(f"Warning: Only {len(incorrect_predictions)} incorrect predictions available")
    
    # Combine selected cases
    selected_cases = pd.concat([selected_correct, selected_incorrect], ignore_index=True)
    
    # Shuffle the final dataset
    selected_cases = selected_cases.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add case ID for reference
    selected_cases['case_id'] = range(1, len(selected_cases) + 1)
    
    # Reorder columns for clarity
    columns_order = ['case_id', 'no_of_dependents', 'education', 'self_employed', 'income_annum', 
                     'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 
                     'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value',
                     'true_loan_status', 'true_loan_status_string', 'model_prediction', 'is_correct']
    
    selected_cases = selected_cases[columns_order]
    
    # Save to CSV
    output_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/experiment_cases.csv'
    selected_cases.to_csv(output_path, index=False)
    
    print(f"\nSelected {len(selected_cases)} cases saved to: {output_path}")
    print(f"Correct predictions in sample: {len(selected_cases[selected_cases['is_correct'] == True])}")
    print(f"Incorrect predictions in sample: {len(selected_cases[selected_cases['is_correct'] == False])}")
    
    return selected_cases

if __name__ == "__main__":
    selected_cases = select_test_cases()
    print("\nFirst 5 cases:")
    print(selected_cases.head())