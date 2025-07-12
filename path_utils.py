"""
Path utilities for the LLM Human Behavior Simulation project.
Provides consistent relative path handling across all modules.
"""

import os

def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    """Get path to data files."""
    return os.path.join(get_project_root(), 'data', filename)

def get_config_path(filename):
    """Get path to config files."""
    return os.path.join(get_project_root(), 'config', filename)

def get_results_path(filename):
    """Get path to results files."""
    results_dir = os.path.join(get_project_root(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)

def get_model_path(filename):
    """Get path to model files."""
    return os.path.join(get_project_root(), filename)

# Common file paths
LOAN_MODEL_PATH = get_model_path('loan_model.pkl')
PERSONAS_CONFIG_PATH = get_config_path('personas.json')
EXPERIMENT_DATA_PATH = get_model_path('experiment_cases_balanced.csv')
TRAINING_DATA_PATH = get_data_path('loan_approval_dataset 2.csv')

if __name__ == "__main__":
    print("Project Path Configuration:")
    print(f"Project Root: {get_project_root()}")
    print(f"Model Path: {LOAN_MODEL_PATH}")
    print(f"Config Path: {PERSONAS_CONFIG_PATH}")
    print(f"Experiment Data: {EXPERIMENT_DATA_PATH}")
    print(f"Training Data: {TRAINING_DATA_PATH}")