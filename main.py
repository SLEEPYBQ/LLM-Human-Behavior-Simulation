import os
import pandas as pd
import json
import time
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
import argparse
from datetime import datetime
from tqdm import tqdm

# Import our modules
from ml_model import LoanApprovalModel
from persona_config import *
from agents.persona_agent import PersonaAgent

class HumanBehaviorSimulation:
    """Main simulation class for running LLM agents with different personas"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.llm = ChatOpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key=self.openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        self.ml_model = LoanApprovalModel()
        self.personas = []
        self.results = []
        
    def load_ml_model(self, model_path: str):
        """Load the trained ML model"""
        self.ml_model.load_model(model_path)
        print(f"Loaded ML model from {model_path}")
    
    def load_personas(self, personas_config_path: str):
        """Load persona configurations"""
        with open(personas_config_path, 'r') as f:
            persona_data = json.load(f)
        
        self.personas = []
        for data in persona_data:
            persona = PersonaConfig(
                expertise_level=ExpertiseLevel(data['expertise_level']),
                ai_trust_level=AITrustLevel(data['ai_trust_level']),
                need_for_cognition=data['need_for_cognition'],
                knowledge_base=data['knowledge_base'],
                need_for_cognition_scale=data.get('need_for_cognition_scale'),
                general_self_efficacy_scale=data.get('general_self_efficacy_scale'),
                thinking_mode_preference=data.get('thinking_mode_preference', 'adaptive')
            )
            self.personas.append(persona)
        
        print(f"Loaded {len(self.personas)} persona configurations")
    
    def load_test_data(self, test_data_path: str) -> pd.DataFrame:
        """Load test dataset"""
        df = pd.read_csv(test_data_path)
        print(f"Loaded {len(df)} test cases")
        return df
    
    def run_simulation(self, 
                      test_data: pd.DataFrame, 
                      num_cases: int = 10, 
                      num_personas: int = 5,
                      verbose: bool = True) -> List[Dict[str, Any]]:
        """Run the simulation with multiple personas on multiple test cases"""
        
        # Select subset of test cases and personas
        selected_cases = test_data.head(num_cases)
        selected_personas = self.personas[:num_personas]
        
        results = []
        total_runs = len(selected_cases) * len(selected_personas)
        current_run = 0
        
        print(f"\\nStarting simulation: {num_cases} cases × {num_personas} personas = {total_runs} total runs\\n")
        
        # Create overall progress bar
        with tqdm(total=total_runs, desc="Overall Progress", unit="run") as pbar:
            for case_idx, case_row in selected_cases.iterrows():
                # Convert to LoanApplication object
                loan_app = LoanApplication(
                    loan_id=int(case_idx),  # Use index as loan_id since it's missing
                    no_of_dependents=int(case_row['no_of_dependents']),
                    education=case_row['education'],
                    self_employed=case_row['self_employed'],
                    income_annum=int(case_row['income_annum']),
                    loan_amount=int(case_row['loan_amount']),
                    loan_term=int(case_row['loan_term']),
                    cibil_score=int(case_row['cibil_score']),
                    residential_assets_value=int(case_row['residential_assets_value']),
                    commercial_assets_value=int(case_row['commercial_assets_value']),
                    luxury_assets_value=int(case_row['luxury_assets_value']),
                    bank_asset_value=int(case_row['bank_asset_value'])
                )
                
                # Get AI recommendation from ML model
                ai_recommendation = AIRecommendation(**self.ml_model.predict_single(loan_app.to_dict()))
                
                # Handle label mapping - the label encoder maps: 0=Approved, 1=Rejected (alphabetical order)
                # But our test data uses: 0=Rejected, 1=Approved (reverse)
                if 'true_loan_status' in case_row:
                    true_label_encoded = case_row['true_loan_status']
                    # Convert from test data encoding (0=Rejected, 1=Approved) to string
                    true_label_str = "Approved" if true_label_encoded == 1 else "Rejected"
                else:
                    # Fallback for old format
                    true_label_encoded = case_row.get('loan_status', 0)
                    true_label_str = "Approved" if true_label_encoded == 1 else "Rejected"
                
                if verbose:
                    tqdm.write(f"\\n--- Case {case_idx + 1}: Loan ID {loan_app.loan_id} ---")
                    tqdm.write(f"True Label: {true_label_str}")
                    tqdm.write(f"AI Prediction: {ai_recommendation.prediction} (confidence: {ai_recommendation.confidence:.2f})")
                
                for persona_idx, persona_config in enumerate(selected_personas):
                    current_run += 1
                    
                    # Update progress bar description with current case and persona info
                    pbar.set_description(f"Case {case_idx + 1}/{len(selected_cases)}, Persona {persona_idx + 1}/{len(selected_personas)}")
                    
                    if verbose:
                        tqdm.write(f"\\n  Persona {persona_idx + 1}/{len(selected_personas)} " + 
                              f"({persona_config.get_persona_description()}) [{current_run}/{total_runs}]")
                    
                    try:
                        # Create agent for this persona
                        agent = PersonaAgent(persona_config, self.llm)
                        
                        # Run the decision-making process
                        start_time = time.time()
                        
                        # Step 1: Initial decision
                        initial_cognition = agent.make_initial_decision(loan_app)
                        
                        # Step 2: Evaluate AI recommendation
                        utility_evaluation = agent.evaluate_ai_recommendation(
                            loan_app, ai_recommendation, initial_cognition
                        )
                        
                        # Step 3: Make final decision
                        final_decision = agent.make_final_decision(
                            loan_app, ai_recommendation, initial_cognition, utility_evaluation
                        )
                        
                        total_time = time.time() - start_time
                        
                        # Store results
                        result = {
                            'loan_id': loan_app.loan_id,
                            'case_index': case_idx,
                            'persona_index': persona_idx,
                            'expertise_level': persona_config.expertise_level.value,
                            'ai_trust_level': persona_config.ai_trust_level.value,
                            'need_for_cognition': persona_config.need_for_cognition,
                            'true_label': true_label_str,
                            'ai_prediction': ai_recommendation.prediction,
                            'ai_confidence': ai_recommendation.confidence,
                            'initial_decision': initial_cognition.initial_decision,
                            'initial_confidence': initial_cognition.confidence_level,
                            'final_decision_choice': final_decision.final_decision.value,
                            'final_confidence': final_decision.confidence,
                            'processing_time': total_time,
                            'initial_reaction': utility_evaluation.initial_reaction,
                            'comparison_with_judgment': utility_evaluation.comparison_with_judgment,
                            'ai_capability_assessment': utility_evaluation.ai_capability_assessment,
                            'risk_benefit_consideration': utility_evaluation.risk_benefit_consideration,
                            'inner_conflict': utility_evaluation.inner_conflict,
                            'full_reasoning': final_decision.reasoning,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results.append(result)
                        
                        if verbose:
                            tqdm.write(f"    Initial: {initial_cognition.initial_decision} ({initial_cognition.confidence_level:.1f}%)")
                            tqdm.write(f"    Final: {final_decision.final_decision.value} ({final_decision.confidence:.1f}%)")
                            tqdm.write(f"    Time: {total_time:.2f}s")
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Update progress bar postfix with stats
                        accept_count = sum(1 for r in results if r.get('final_decision_choice') == 'ACCEPT')
                        reject_count = sum(1 for r in results if r.get('final_decision_choice') == 'REJECT')
                        avg_time = sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
                        
                        pbar.set_postfix({
                            'Accept': accept_count,
                            'Reject': reject_count, 
                            'Avg Time': f'{avg_time:.1f}s'
                        })
                        
                    except Exception as e:
                        if verbose:
                            tqdm.write(f"    ERROR: {str(e)}")
                        # Still record the error case
                        results.append({
                            'loan_id': loan_app.loan_id,
                            'case_index': case_idx,
                            'persona_index': persona_idx,
                            'expertise_level': persona_config.expertise_level.value,
                            'ai_trust_level': persona_config.ai_trust_level.value,
                            'need_for_cognition': persona_config.need_for_cognition,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Still update progress bar even on error
                        pbar.update(1)
        
        self.results = results
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filepath: str):
        """Save simulation results to CSV"""
        df_results = pd.DataFrame(results)
        df_results.to_csv(filepath, index=False)
        print(f"\\nSaved {len(results)} simulation results to {filepath}")
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        df = pd.DataFrame(results)
        
        # Filter out error cases
        df_clean = df[~df['error'].notna()] if 'error' in df.columns else df
        
        if len(df_clean) == 0:
            print("No successful results to analyze")
            return {}
        
        analysis = {
            'total_cases': len(df_clean),
            'avg_processing_time': df_clean['processing_time'].mean(),
            'decision_distribution': df_clean['final_decision_choice'].value_counts().to_dict()
        }
        
        # Convert grouped results to simple dictionaries
        expertise_patterns = {}
        for (expertise, decision), count in df_clean.groupby('expertise_level')['final_decision_choice'].value_counts().items():
            key = f"{expertise}_{decision}"
            expertise_patterns[key] = count
        
        trust_patterns = {}
        for (trust, decision), count in df_clean.groupby('ai_trust_level')['final_decision_choice'].value_counts().items():
            key = f"trust_{trust}_{decision}"
            trust_patterns[key] = count
        
        cognition_patterns = {}
        for (cognition, decision), count in df_clean.groupby('need_for_cognition')['final_decision_choice'].value_counts().items():
            key = f"cognition_{cognition}_{decision}"
            cognition_patterns[key] = count
        
        analysis.update({
            'expertise_patterns': expertise_patterns,
            'trust_patterns': trust_patterns,
            'cognition_patterns': cognition_patterns
        })
        
        print("\\n=== SIMULATION ANALYSIS ===")
        print(f"Total successful runs: {analysis['total_cases']}")
        print(f"Average processing time: {analysis['avg_processing_time']:.2f}s")
        print(f"\\nDecision Distribution:")
        for decision, count in analysis['decision_distribution'].items():
            print(f"  {decision}: {count}")
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Run Human Behavior Simulation')
    parser.add_argument('--cases', type=int, default=5, help='Number of test cases to run')
    parser.add_argument('--personas', type=int, default=3, help='Number of personas to test')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    try:
        # Initialize simulation
        sim = HumanBehaviorSimulation(openai_api_key=args.api_key)
        
        # Load components
        sim.load_ml_model('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/loan_model.pkl')
        sim.load_personas('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json')
        test_data = sim.load_test_data('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/test_data.csv')
        
        # Run simulation
        results = sim.run_simulation(
            test_data=test_data,
            num_cases=args.cases,
            num_personas=args.personas,
            verbose=args.verbose
        )
        
        # Save and analyze results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/results/simulation_results_{timestamp}.csv'
        
        # Create results directory
        os.makedirs('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/results', exist_ok=True)
        
        sim.save_results(results, results_file)
        analysis = sim.analyze_results(results)
        
        # Save analysis
        analysis_file = f'/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/results/analysis_{timestamp}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {analysis_file}")
        
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())