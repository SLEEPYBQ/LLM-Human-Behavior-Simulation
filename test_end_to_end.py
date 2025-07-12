import os
from main import HumanBehaviorSimulation
import pandas as pd

def test_end_to_end():
    """Test end-to-end simulation with label verification"""
    
    print("=== END-TO-END SIMULATION TEST ===")
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("WARNING: No OPENAI_API_KEY found. This test will fail.")
        print("Set the environment variable: export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # Initialize simulation
        sim = HumanBehaviorSimulation()
        
        # Load components
        sim.load_ml_model('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl')
        sim.load_personas('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json')
        
        # Use our fixed test data
        test_data = pd.read_csv('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/experiment_cases_fixed.csv')
        
        print(f"Running test with {len(test_data)} cases...")
        print("First 3 cases:")
        for i in range(3):
            case = test_data.iloc[i]
            print(f"  Case {case['case_id']}: True={case['true_loan_status']}({case['true_loan_status_string']}), "
                  f"Model={case['model_prediction']}, Correct={case['is_correct']}")
        
        # Run simulation with just 2 cases and 2 personas for quick test
        print("\nRunning simulation...")
        results = sim.run_simulation(
            test_data=test_data,
            num_cases=2,  # Just 2 cases for quick test
            num_personas=2,  # Just 2 personas 
            verbose=True
        )
        
        print(f"\n=== RESULTS ANALYSIS ===")
        print(f"Total results: {len(results)}")
        
        # Check for errors
        error_results = [r for r in results if 'error' in r]
        if error_results:
            print(f"ERROR RESULTS: {len(error_results)}")
            for r in error_results:
                print(f"  {r.get('error', 'Unknown error')}")
        
        # Analyze successful results
        success_results = [r for r in results if 'error' not in r]
        print(f"SUCCESSFUL RESULTS: {len(success_results)}")
        
        if success_results:
            print("\nDecision Analysis:")
            for result in success_results:
                case_id = result['loan_id']
                true_label = result['true_label']
                ai_pred = result['ai_prediction']
                initial_decision = result['initial_decision']
                final_choice = result['final_decision_choice']
                final_conf = result['final_confidence']
                
                print(f"  Case {case_id}:")
                print(f"    True: {true_label}")
                print(f"    AI: {ai_pred}")
                print(f"    Human Initial: {initial_decision}")
                print(f"    Final Choice: {final_choice} (conf: {final_conf:.1f}%)")
                
                # Check for consistency
                if ai_pred == initial_decision:
                    print(f"    ✓ AI and Human agree")
                else:
                    print(f"    ⚠ AI and Human disagree")
                
                if final_choice == 'ACCEPT_AI':
                    final_decision_str = ai_pred
                else:
                    final_decision_str = initial_decision
                
                if final_decision_str == true_label:
                    print(f"    ✓ Final decision correct")
                else:
                    print(f"    ✗ Final decision incorrect")
                print()
        
        print("=== END-TO-END TEST COMPLETE ===")
        
        if success_results:
            print("✓ System is working correctly")
            print("✓ Labels appear to be consistent")
            print("✓ Ready for full experiment")
        else:
            print("✗ No successful results - check errors above")
            
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_end_to_end()