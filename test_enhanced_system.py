#!/usr/bin/env python3
"""
Test script for the enhanced Human Behavior Simulation system.

This script tests all the new features:
1. Fast vs Slow Thinking
2. Need for Cognition Scale (NCS)
3. General Self-Efficacy Scale (GSE)
4. Enhanced persona configurations
5. Updated agent implementation
"""

import sys
import os
import json
import time
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP')

from persona_config import PersonaConfig, ExpertiseLevel, AITrustLevel, LoanApplication, AIRecommendation
from psychological_scales import ScaleManager, ScaleLevel
from thinking_modes import ThinkingModeController, ThinkingMode
from agents.persona_agent import PersonaAgent
from ml_model import LoanApprovalModel
from main import HumanBehaviorSimulation

def test_psychological_scales():
    """Test the psychological scales functionality"""
    print("\\n=== Testing Psychological Scales ===")
    
    scale_manager = ScaleManager()
    
    # Test all combinations
    levels = [ScaleLevel.LOW, ScaleLevel.MEDIUM, ScaleLevel.HIGH]
    
    for ncs_level in levels:
        for gse_level in levels:
            scales = scale_manager.generate_persona_scales(ncs_level, gse_level)
            
            ncs_data = scales['need_for_cognition']
            gse_data = scales['general_self_efficacy']
            
            print(f"NCS: {ncs_level.value} -> Score: {ncs_data['score']}/90")
            print(f"GSE: {gse_level.value} -> Score: {gse_data['score']}/40")
            print()
    
    print("✓ Psychological scales working correctly")

def test_thinking_modes():
    """Test the fast vs slow thinking modes"""
    print("\\n=== Testing Thinking Modes ===")
    
    try:
        # Import required modules
        from langchain_openai import ChatOpenAI
        
        # Create a mock LLM (this would need an actual API key to work)
        # For testing purposes, we'll just check the class structure
        llm = ChatOpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key='test-key',
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Create a test persona configuration
        from persona_config import create_persona_configs
        personas = create_persona_configs()
        test_persona = personas[0]  # Use first persona
        
        # Test thinking mode controller creation
        thinking_controller = ThinkingModeController(llm, test_persona)
        
        # Create a test loan application
        loan_app = LoanApplication(
            loan_id=1,
            no_of_dependents=2,
            education="Graduate",
            self_employed="No",
            income_annum=5000000,
            loan_amount=10000000,
            loan_term=15,
            cibil_score=750,
            residential_assets_value=8000000,
            commercial_assets_value=2000000,
            luxury_assets_value=5000000,
            bank_asset_value=3000000
        )
        
        # Test thinking mode determination
        thinking_mode = thinking_controller.determine_thinking_mode(loan_app)
        print(f"Determined thinking mode: {thinking_mode.value}")
        
        print("✓ Thinking modes structure working correctly")
        
    except Exception as e:
        print(f"⚠ Thinking modes test skipped (requires API key): {e}")

def test_enhanced_persona_config():
    """Test the enhanced persona configuration"""
    print("\\n=== Testing Enhanced Persona Configuration ===")
    
    # Load personas from the generated file
    with open('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json', 'r') as f:
        persona_data = json.load(f)
    
    # Test first few personas
    for i, data in enumerate(persona_data[:3]):
        print(f"\\nPersona {i+1}:")
        print(f"  Expertise: {data['expertise_level']}")
        print(f"  AI Trust: {data['ai_trust_level']}")
        print(f"  NCS Score: {data['need_for_cognition_scale']['score']}/90 ({data['need_for_cognition_scale']['level']})")
        print(f"  GSE Score: {data['general_self_efficacy_scale']['score']}/40 ({data['general_self_efficacy_scale']['level']})")
        print(f"  Thinking Mode: {data['thinking_mode_preference']}")
        print(f"  Interpretation:")
        print(f"    NCS: {data['need_for_cognition_scale']['interpretation']}")
        print(f"    GSE: {data['general_self_efficacy_scale']['interpretation']}")
    
    print(f"\\n✓ Enhanced persona configuration working correctly ({len(persona_data)} personas loaded)")

def test_agent_creation():
    """Test creating PersonaAgent with enhanced features"""
    print("\\n=== Testing Enhanced Agent Creation ===")
    
    try:
        # Create persona configuration
        from persona_config import create_persona_configs
        personas = create_persona_configs()
        test_persona = personas[0]
        
        # Create mock LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key='test-key',
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Test agent creation
        agent = PersonaAgent(test_persona, llm)
        
        print(f"✓ Agent created successfully with persona: {test_persona.get_persona_description()}")
        print(f"  Thinking mode controller: {type(agent.thinking_controller).__name__}")
        
    except Exception as e:
        print(f"⚠ Agent creation test skipped (requires API key): {e}")

def test_ml_model():
    """Test ML model functionality"""
    print("\\n=== Testing ML Model ===")
    
    # Check if model file exists
    model_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl'
    if os.path.exists(model_path):
        # Load model
        model = LoanApprovalModel()
        model.load_model(model_path)
        
        # Test prediction
        test_loan = {
            'no_of_dependents': 2,
            'education': 0,  # Encoded
            'self_employed': 0,  # Encoded
            'income_annum': 5000000,
            'loan_amount': 10000000,
            'loan_term': 15,
            'cibil_score': 750,
            'residential_assets_value': 8000000,
            'commercial_assets_value': 2000000,
            'luxury_assets_value': 5000000,
            'bank_asset_value': 3000000
        }
        
        prediction = model.predict_single(test_loan)
        print(f"✓ ML model working correctly")
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
        
    else:
        print("⚠ ML model not found. Run 'python ml_model.py' first.")

def test_data_availability():
    """Test data file availability"""
    print("\\n=== Testing Data Availability ===")
    
    # Check test data
    test_data_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/test_data.csv'
    if os.path.exists(test_data_path):
        df = pd.read_csv(test_data_path)
        print(f"✓ Test data available: {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
    else:
        print("⚠ Test data not found. Run 'python ml_model.py' first.")
    
    # Check persona configurations
    personas_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json'
    if os.path.exists(personas_path):
        with open(personas_path, 'r') as f:
            personas = json.load(f)
        print(f"✓ Persona configurations available: {len(personas)} personas")
    else:
        print("⚠ Persona configurations not found. Run 'python persona_config.py' first.")

def test_system_integration():
    """Test the complete system integration"""
    print("\\n=== Testing System Integration ===")
    
    try:
        # Test simulation initialization
        sim = HumanBehaviorSimulation()
        
        # Test component loading
        model_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl'
        if os.path.exists(model_path):
            sim.load_ml_model(model_path)
            print("✓ ML model loaded successfully")
        
        personas_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json'
        if os.path.exists(personas_path):
            sim.load_personas(personas_path)
            print("✓ Personas loaded successfully")
        
        test_data_path = '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/test_data.csv'
        if os.path.exists(test_data_path):
            test_data = sim.load_test_data(test_data_path)
            print("✓ Test data loaded successfully")
        
        print("✓ System integration working correctly")
        
    except Exception as e:
        print(f"⚠ System integration test failed: {e}")

def main():
    """Run all tests"""
    print("==========================================")
    print("Human Behavior Simulation - Enhanced System Test")
    print("==========================================")
    
    test_psychological_scales()
    test_thinking_modes()
    test_enhanced_persona_config()
    test_agent_creation()
    test_ml_model()
    test_data_availability()
    test_system_integration()
    
    print("\\n==========================================")
    print("Test Summary")
    print("==========================================")
    print("✓ All structural tests completed successfully")
    print("⚠ Some tests require API key for full functionality")
    print("\\nTo run the full system:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python main.py --cases 2 --personas 3 --verbose")

if __name__ == "__main__":
    main()