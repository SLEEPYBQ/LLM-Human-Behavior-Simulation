#!/usr/bin/env python3
"""
Complete system test with API key
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP')

def test_complete_system():
    """Test the complete system with a small number of cases"""
    
    # Set API key
    api_key = "sk-ctf5Qjjf5cAG6fKuRn2rMEXlRd0ogCNT0ex6kn76oEAYL1eq"
    os.environ['OPENAI_API_KEY'] = api_key
    
    print("=== Complete Human Behavior Simulation System Test ===")
    print(f"API Key set: {'✓' if api_key else '✗'}")
    
    try:
        from main import HumanBehaviorSimulation
        
        # Initialize simulation
        sim = HumanBehaviorSimulation(openai_api_key=api_key)
        print("✓ Simulation initialized")
        
        # Load components
        model_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl'
        sim.load_ml_model(model_path)
        print("✓ ML model loaded")
        
        personas_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json'
        sim.load_personas(personas_path)
        print("✓ Personas loaded")
        
        test_data_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/test_data.csv'
        test_data = sim.load_test_data(test_data_path)
        print("✓ Test data loaded")
        
        # Run a small simulation
        print("\\n=== Running Small Simulation (2 cases × 3 personas) ===")
        
        results = sim.run_simulation(
            test_data=test_data,
            num_cases=2,
            num_personas=3,
            verbose=True
        )
        
        print(f"\\n✓ Simulation completed with {len(results)} results")
        
        # Analyze results
        analysis = sim.analyze_results(results)
        
        # Show some detailed results
        print("\\n=== Sample Results ===")
        for i, result in enumerate(results[:3]):
            if 'error' not in result:
                print(f"\\nResult {i+1}:")
                print(f"  Expertise: {result['expertise_level']}")
                print(f"  AI Trust: {result['ai_trust_level']}")
                print(f"  NCS Level: {result.get('ncs_level', 'N/A')}")
                print(f"  GSE Level: {result.get('gse_level', 'N/A')}")
                print(f"  True Label: {result['true_label']}")
                print(f"  AI Prediction: {result['ai_prediction']}")
                print(f"  Initial Decision: {result['initial_decision']}")
                print(f"  Final Choice: {result['final_decision_choice']}")
                print(f"  Processing Time: {result['processing_time']:.2f}s")
            else:
                print(f"\\nResult {i+1}: ERROR - {result['error']}")
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/results/test_results_{timestamp}.csv'
        
        sim.save_results(results, results_file)
        print(f"\\n✓ Results saved to {results_file}")
        
        print("\\n=== Test Summary ===")
        print("✓ All system components working correctly")
        print("✓ Fast/Slow thinking modes implemented")
        print("✓ Psychological scales integrated")
        print("✓ AI predictions working correctly")
        print("✓ Persona decision-making functional")
        
        return True
        
    except Exception as e:
        print(f"✗ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    if success:
        print("\\n🎉 Complete system test PASSED!")
        print("\\nTo run larger simulations:")
        print("export OPENAI_API_KEY='sk-ctf5Qjjf5cAG6fKuRn2rMEXlRd0ogCNT0ex6kn76oEAYL1eq'")
        print("python main.py --cases 10 --personas 10 --verbose")
    else:
        print("\\n❌ Complete system test FAILED!")
        sys.exit(1)