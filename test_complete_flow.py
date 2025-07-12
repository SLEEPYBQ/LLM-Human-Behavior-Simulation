import pandas as pd
import json
from ml_model import LoanApprovalModel
from persona_config import *

def test_complete_flow():
    """Test the complete flow from ML model to persona decision"""
    
    print("=== TESTING COMPLETE FLOW ===")
    
    # Load test data
    test_data = pd.read_csv('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/experiment_cases_fixed.csv')
    print(f"Loaded {len(test_data)} test cases")
    
    # Load ML model
    model = LoanApprovalModel()
    model.load_model('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl')
    print("ML model loaded")
    
    # Test first case
    case = test_data.iloc[0]
    print(f"\n--- Testing Case {case['case_id']} ---")
    print(f"True status: {case['true_loan_status']} ({case['true_loan_status_string']})")
    print(f"Expected prediction: {case['model_prediction']}")
    print(f"Is correct: {case['is_correct']}")
    
    # Prepare loan data for ML model
    loan_data = {
        'no_of_dependents': case['no_of_dependents'],
        'education': case['education'], 
        'self_employed': case['self_employed'],
        'income_annum': case['income_annum'],
        'loan_amount': case['loan_amount'],
        'loan_term': case['loan_term'],
        'cibil_score': case['cibil_score'],
        'residential_assets_value': case['residential_assets_value'],
        'commercial_assets_value': case['commercial_assets_value'],
        'luxury_assets_value': case['luxury_assets_value'],
        'bank_asset_value': case['bank_asset_value']
    }
    
    # Get ML prediction
    print("\n1. ML Model Prediction:")
    ml_result = model.predict_single(loan_data)
    print(f"   Raw result: {ml_result}")
    
    # Convert to binary for comparison
    ml_binary = 1 if ml_result['prediction'] == 'Approved' else 0
    print(f"   Binary prediction: {ml_binary}")
    print(f"   Matches CSV: {ml_binary == case['model_prediction']}")
    
    # Test LoanApplication creation
    print("\n2. LoanApplication Object:")
    try:
        loan_app = LoanApplication(
            loan_id=int(case['case_id']),
            no_of_dependents=int(case['no_of_dependents']),
            education=case['education'],
            self_employed=case['self_employed'],
            income_annum=int(case['income_annum']),
            loan_amount=int(case['loan_amount']),
            loan_term=int(case['loan_term']),
            cibil_score=int(case['cibil_score']),
            residential_assets_value=int(case['residential_assets_value']),
            commercial_assets_value=int(case['commercial_assets_value']),
            luxury_assets_value=int(case['luxury_assets_value']),
            bank_asset_value=int(case['bank_asset_value'])
        )
        print(f"   LoanApplication created successfully")
        print(f"   loan_app.to_dict() works: {bool(loan_app.to_dict())}")
    except Exception as e:
        print(f"   ERROR creating LoanApplication: {e}")
        return
    
    # Test AIRecommendation creation
    print("\n3. AIRecommendation Object:")
    try:
        ai_recommendation = AIRecommendation(**ml_result)
        print(f"   AIRecommendation created successfully")
        print(f"   prediction: {ai_recommendation.prediction}")
        print(f"   confidence: {ai_recommendation.confidence}")
    except Exception as e:
        print(f"   ERROR creating AIRecommendation: {e}")
        print(f"   Trying to fix...")
        # Try to fix the issue
        try:
            ai_recommendation = AIRecommendation(
                prediction=ml_result['prediction'],
                confidence=ml_result['confidence']
            )
            print(f"   Fixed AIRecommendation created")
        except Exception as e2:
            print(f"   Still failed: {e2}")
            return
    
    # Test if we have valid personas
    print("\n4. Persona Configuration:")
    try:
        with open('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json', 'r') as f:
            persona_data = json.load(f)
        print(f"   Found {len(persona_data)} personas")
        
        # Create a simple persona for testing
        first_persona_data = persona_data[0]
        persona = PersonaConfig(
            expertise_level=ExpertiseLevel(first_persona_data['expertise_level']),
            ai_trust_level=AITrustLevel(first_persona_data['ai_trust_level']),
            need_for_cognition=first_persona_data['need_for_cognition'],
            knowledge_base=first_persona_data['knowledge_base'],
            need_for_cognition_scale=first_persona_data.get('need_for_cognition_scale'),
            general_self_efficacy_scale=first_persona_data.get('general_self_efficacy_scale'),
            thinking_mode_preference=first_persona_data.get('thinking_mode_preference', 'adaptive')
        )
        print(f"   Persona created: {persona.get_persona_description()}")
        
    except Exception as e:
        print(f"   ERROR with personas: {e}")
        return
    
    print("\n=== FLOW TEST COMPLETE ===")
    print("All components seem to work individually.")
    print("Ready for full integration test.")

if __name__ == "__main__":
    test_complete_flow()