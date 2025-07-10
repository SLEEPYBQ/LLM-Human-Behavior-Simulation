# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Agents for Simulating Human Decision Making Behavior (LLM4SHB)** - A comprehensive LangChain-based system that simulates human decision-making processes in loan approval scenarios. The system models different personas with varying expertise levels, cognitive abilities, and AI trust levels to understand how humans make decisions when presented with AI recommendations.

## System Architecture & Design Philosophy

### Core Design Principles
1. **Multi-Agent Architecture**: Separate specialized agents for different cognitive processes
2. **Persona-Based Modeling**: Different expertise levels with corresponding knowledge bases
3. **Dual-Process Theory**: Fast vs Slow thinking implementation
4. **Trust-Based Decision Making**: AI trust levels influence final decisions
5. **Utility Evaluation**: Structured decision-making process with emotional and rational components

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Task          │    │   AI Trust      │    │   Advice        │
│   Proficiency   │    │   Level         │    │   Adoption      │
│                 │    │                 │    │                 │
│ • Persona       │    │ • Likert Scale  │    │ • Task          │
│ • Knowledge     │    │   (1-5)         │    │ • Self-decision │
│   Base          │    │                 │    │ • Utility Eval  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                ┌─────────────────────────────────┐
                │   Cognition & Thinking Ability  │
                │                                 │
                │ Fast Thinking    Slow Thinking  │
                │ • Memory         • Reflection   │
                │ • Task Simp.     • Reasoning    │
                │                  • Planning     │
                │                  • Chain of     │
                │                    Thought      │
                │                  • Theory of    │
                │                    Mind         │
                └─────────────────────────────────┘
```

## Implementation Details

### 1. Data Pipeline (`ml_model.py`)
**Purpose**: Train ML model to provide AI recommendations

**Key Components**:
- `LoanApprovalModel`: Random Forest classifier with preprocessing
- Feature encoding for categorical variables
- Train/test split with stratification
- Model persistence and prediction interface

**Design Decisions**:
- Used Random Forest for interpretability and robustness
- Achieved 98.36% accuracy on test set
- Separate test set for LLM agent evaluation (same data, different predictions)

### 2. Persona Configuration (`persona_config.py`)
**Purpose**: Define different human personas with varying characteristics

**Key Components**:
- `PersonaConfig`: Dataclass defining persona attributes
- `ExpertiseLevel`: Beginner (0%), Intermediate (30%), Expert (80%) knowledge
- `AITrustLevel`: -2 (Very Distrusting) to +2 (Very Trusting)
- `KnowledgeBase`: Expertise-dependent loan approval knowledge

**Design Decisions**:
- 27 different persona combinations (3 expertise × 3 trust × 3 cognition levels)
- Knowledge base content varies by expertise level
- Structured data classes for type safety and serialization

### 3. LangChain Agents (`agents/persona_agent.py`)
**Purpose**: Implement persona-based decision-making agents

**Key Components**:
- `PersonaAgent`: Main agent class embodying specific persona
- `create_loan_analysis_tool`: Knowledge-based loan analysis tool
- Prompt templates reflecting persona characteristics
- Multi-step decision process implementation

**Design Decisions**:
- Used LangChain's function calling for structured analysis
- Persona-specific prompts for authentic behavior
- Tool-based knowledge access (not direct prompt injection)
- Simplified memory model to avoid complexity

### 4. Decision Process Flow
**Step 1: Initial Decision**
```python
def make_initial_decision(loan_application) -> CognitionState:
    # Analyze loan using persona's knowledge base
    # Generate initial approve/reject decision
    # Extract confidence level and reasoning
```

**Step 2: AI Recommendation Evaluation**
```python
def evaluate_ai_recommendation(loan_app, ai_rec, initial_cognition) -> UtilityEvaluation:
    # Present AI recommendation to persona
    # Capture emotional reaction and comparison
    # Assess AI capability based on trust level
    # Consider risks and benefits
```

**Step 3: Final Decision**
```python
def make_final_decision(...) -> DecisionResult:
    # Combine initial decision with AI evaluation
    # Choose adoption level (A-E scale)
    # Provide final reasoning and confidence
```

### 5. Simulation Engine (`main.py`)
**Purpose**: Orchestrate the complete simulation workflow

**Key Components**:
- `HumanBehaviorSimulation`: Main simulation controller
- Batch processing of test cases across multiple personas
- Result collection and analysis
- Performance metrics and timing

**Design Decisions**:
- Configurable simulation parameters (cases, personas)
- Detailed logging and error handling
- Structured result output for analysis
- Real-time progress tracking

## Development Commands

```bash
# Setup
pip install langchain openai pandas scikit-learn joblib

# Train ML model (AI decision maker)
python ml_model.py

# Generate persona configurations
python persona_config.py

# Test API connection
python test_api.py

# Run simulation
export OPENAI_API_KEY='your-api-key'
python main.py --cases 5 --personas 3 --verbose

# Results will be saved in results/ directory
```

## Key Files & Responsibilities

### Core System Files
- `ml_model.py`: ML model training and AI prediction generation
- `persona_config.py`: Persona definitions and knowledge base management
- `agents/persona_agent.py`: LangChain agent implementation
- `main.py`: Simulation orchestration and execution
- `test_api.py`: API connection testing

### Data Files
- `data/loan_approval_dataset 2.csv`: Training dataset
- `test_data.csv`: Test cases for LLM evaluation
- `config/personas.json`: Generated persona configurations
- `loan_model.pkl`: Trained ML model

### Output Files
- `results/simulation_results_*.csv`: Detailed simulation results
- `results/analysis_*.json`: Aggregated analysis results

## Configuration

### API Configuration
- Uses OpenAI proxy: `https://api.openai-proxy.org/v1`
- Model: `gpt-3.5-turbo`
- Temperature: 0.7 for consistent but varied responses

### Persona Parameters
- **Expertise Levels**: 3 levels with different knowledge completeness
- **AI Trust Levels**: 5 levels from very distrusting to very trusting  
- **Cognition Levels**: 3 levels of need for cognition (2, 4, 6)

## Research Insights

### System Capabilities
1. **Persona Differentiation**: Different expertise levels show distinct decision patterns
2. **Trust Impact**: AI trust levels significantly influence adoption choices
3. **Cognitive Load**: Higher cognition need leads to more detailed analysis
4. **Utility Evaluation**: Captures human-like decision conflicts and emotions

### Observed Patterns
- Beginners rely more heavily on basic indicators (CIBIL score)
- Lower trust personas show more resistance to AI recommendations
- Higher cognition personas provide more detailed reasoning
- Decision conflicts emerge when AI and human judgments differ

## Future Enhancements

1. **Advanced Cognitive Models**: Implement more sophisticated dual-process theory
2. **Learning Mechanisms**: Allow personas to update based on experience
3. **Contextual Factors**: Add market conditions and regulatory constraints
4. **Batch Processing**: Optimize for large-scale experiments
5. **Visualization**: Add decision tree and pattern analysis tools

## Technical Notes

- System requires OpenAI API access for LLM functionality
- Processing time: ~8-12 seconds per persona per case
- Memory usage scales with number of concurrent personas
- Results are fully reproducible with same random seeds
- All intermediate steps are captured for detailed analysis