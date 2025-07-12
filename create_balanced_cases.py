import pandas as pd
import numpy as np
from ml_model import LoanApprovalModel

def select_balanced_test_cases():
    """Select 40 balanced test cases with equal approve/reject distribution"""
    
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
        
        # Convert to binary (1=Approved, 0=Rejected)
        pred_binary = 1 if pred_result['prediction'] == 'Approved' else 0
        predictions.append(pred_binary)
    
    # Add predictions to dataframe
    test_data['model_prediction'] = predictions
    
    # Identify correct and incorrect predictions
    test_data['is_correct'] = (test_data['model_prediction'] == test_data['true_loan_status'])
    
    # Separate by ground truth and prediction accuracy
    approved_correct = test_data[(test_data['true_loan_status'] == 1) & (test_data['is_correct'] == True)]
    approved_incorrect = test_data[(test_data['true_loan_status'] == 1) & (test_data['is_correct'] == False)]
    rejected_correct = test_data[(test_data['true_loan_status'] == 0) & (test_data['is_correct'] == True)]
    rejected_incorrect = test_data[(test_data['true_loan_status'] == 0) & (test_data['is_correct'] == False)]
    
    print("=== AVAILABLE SAMPLES BY CATEGORY ===")
    print(f"Approved cases - Correct: {len(approved_correct)}, Incorrect: {len(approved_incorrect)}")
    print(f"Rejected cases - Correct: {len(rejected_correct)}, Incorrect: {len(rejected_incorrect)}")
    
    # Target: 40 cases total
    # - 20 Approved, 20 Rejected (balanced ground truth)
    # - 28 correct, 12 incorrect overall
    
    # Calculate distribution for each category
    # We need to distribute 28 correct and 12 incorrect across approved/rejected
    # Target: 14 correct approved + 14 correct rejected + 6 incorrect approved + 6 incorrect rejected
    
    target_approved_correct = 14
    target_approved_incorrect = 6
    target_rejected_correct = 14
    target_rejected_incorrect = 6
    
    print(f"\\n=== TARGET DISTRIBUTION ===")
    print(f"Approved cases: {target_approved_correct + target_approved_incorrect} (correct: {target_approved_correct}, incorrect: {target_approved_incorrect})")
    print(f"Rejected cases: {target_rejected_correct + target_rejected_incorrect} (correct: {target_rejected_correct}, incorrect: {target_rejected_incorrect})")
    print(f"Total: 40 cases (correct: 28, incorrect: 12)")
    
    # Sample from each category
    np.random.seed(42)  # For reproducibility
    
    selected_parts = []
    
    # Sample approved correct cases
    if len(approved_correct) >= target_approved_correct:
        selected_approved_correct = approved_correct.sample(n=target_approved_correct, random_state=42)
        selected_parts.append(selected_approved_correct)
        print(f"✓ Selected {target_approved_correct} approved correct cases")
    else:
        print(f"✗ Only {len(approved_correct)} approved correct cases available (need {target_approved_correct})")
        return None
    
    # Sample approved incorrect cases
    if len(approved_incorrect) >= target_approved_incorrect:
        selected_approved_incorrect = approved_incorrect.sample(n=target_approved_incorrect, random_state=42)
        selected_parts.append(selected_approved_incorrect)
        print(f"✓ Selected {target_approved_incorrect} approved incorrect cases")
    else:
        print(f"✗ Only {len(approved_incorrect)} approved incorrect cases available (need {target_approved_incorrect})")
        return None
    
    # Sample rejected correct cases
    if len(rejected_correct) >= target_rejected_correct:
        selected_rejected_correct = rejected_correct.sample(n=target_rejected_correct, random_state=42)
        selected_parts.append(selected_rejected_correct)
        print(f"✓ Selected {target_rejected_correct} rejected correct cases")
    else:
        print(f"✗ Only {len(rejected_correct)} rejected correct cases available (need {target_rejected_correct})")
        return None
    
    # Sample rejected incorrect cases
    if len(rejected_incorrect) >= target_rejected_incorrect:
        selected_rejected_incorrect = rejected_incorrect.sample(n=target_rejected_incorrect, random_state=42)
        selected_parts.append(selected_rejected_incorrect)
        print(f"✓ Selected {target_rejected_incorrect} rejected incorrect cases")
    else:
        print(f"✗ Only {len(rejected_incorrect)} rejected incorrect cases available (need {target_rejected_incorrect})")
        return None
    
    # Combine all selected cases
    selected_cases = pd.concat(selected_parts, ignore_index=True)
    
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
    
    # Verify final distribution
    print(f"\\n=== FINAL VERIFICATION ===")
    print(f"Total cases: {len(selected_cases)}")
    print(f"Ground truth distribution:")
    print(selected_cases['true_loan_status_string'].value_counts())
    print(f"Prediction accuracy:")
    print(selected_cases['is_correct'].value_counts())
    
    # Save to CSV
    output_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/experiment_cases_balanced.csv'
    selected_cases.to_csv(output_path, index=False)
    
    print(f"\\nBalanced dataset saved to: {output_path}")
    
    return selected_cases

if __name__ == "__main__":
    selected_cases = select_balanced_test_cases()
    if selected_cases is not None:
        print("\\n=== SUCCESS ===")
        print("Balanced test cases created successfully!")
    else:
        print("\\n=== FAILED ===")
        print("Could not create balanced dataset - insufficient samples in some categories")