# LLM Agents for Simulating Human Decision-Making Behavior

A comprehensive LangChain-based system that simulates human decision-making processes in AI-assisted loan approval scenarios, modeling different personas with varying expertise levels, cognitive abilities, and AI trust levels.

## 🎯 Project Overview

This system implements sophisticated psychological theories to create realistic human behavior simulations:

- **81 different personas** (3 expertise × 3 trust × 3 cognition × 3 self-efficacy)
- **Dual Process Theory** implementation (Fast vs Slow thinking)
- **Psychological Scales** integration (NCS-18, GSE-10)
- **AI Trust Modeling** with authentic emotional responses
- **Complete decision pipeline** tracking

## ✨ Key Features

- 🧠 **Advanced Psychological Modeling**: Implements Need for Cognition Scale (NCS-18) and General Self-Efficacy Scale (GSE-10)
- 🎭 **Realistic Persona Simulation**: 81 unique psychological profiles with authentic decision-making patterns
- 🤖 **AI-Human Interaction**: Models how different personality types respond to AI recommendations
- 📊 **Comprehensive Analytics**: Detailed analysis of acceptance rates, decision switching patterns, and psychological insights
- 🔄 **Dual Process Implementation**: Separate fast (System 1) and slow (System 2) thinking modes
- 🎨 **Enhanced Prompt Engineering**: Psychologically-driven persona prompts for authentic behavior

## 🚀 Quick Start

### Prerequisites

```bash
pip install langchain langchain-openai openai pandas scikit-learn tqdm joblib numpy
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/[username]/LLM-Human-Behavior-Simulation.git
cd LLM-Human-Behavior-Simulation
```

2. **Set up API key**
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

3. **Train the ML model**
```bash
python ml_model.py
```

4. **Generate persona configurations**
```bash
python persona_config.py
```

5. **Run simulation**
```bash
python main.py --cases 5 --personas 9 --verbose
```

## 📊 Recent Results

**Latest Analysis (12 valid cases)**:
- **AI Acceptance Rate**: 16.7% (significant improvement from 0%)
- **Decision Switching Rate**: 41.7% (shows cognitive conflict)
- **Average Processing Time**: 10-12 seconds per persona
- **System Stability**: 100% success rate, zero JSON parsing errors

## 🧠 Psychological Insights

### Key Findings

1. **Trust Breakthrough**: Even "very distrusting" personas accept AI recommendations under certain conditions
2. **Cognitive Conflict**: 41.7% decision switching rate demonstrates complex human psychology
3. **Quality Impact**: High-quality cases (CIBIL 890) overcome initial AI distrust
4. **Self-Efficacy Effects**: Higher self-efficacy correlates with decision confidence

### Persona Behavior Patterns

- **Beginners**: More cautious, rely on basic indicators
- **Low Trust**: Actively resist AI recommendations but show internal conflict
- **High Self-Efficacy**: More confident in final decisions, less swayed by AI

## 🏗️ Architecture

```
HumanBehaviorSimulation (Main Controller)
├── LoanApprovalModel (ML Baseline)
├── PersonaAgent (Enhanced Psychological Prompting)
│   ├── ThinkingModeController
│   │   ├── FastThinkingAgent (System 1)
│   │   └── SlowThinkingAgent (System 2)
│   └── PersonaConfig (81 combinations)
│       ├── NeedForCognitionScale (NCS-18)
│       ├── GeneralSelfEfficacyScale (GSE-10)
│       └── Dynamic Psychological Adaptation
└── Analysis & Reporting
    ├── Real-time Progress Monitoring
    ├── Psychological Pattern Analysis
    └── Decision Behavior Statistics
```

## 📁 Project Structure

```
├── agents/
│   └── persona_agent.py          # Enhanced psychological persona agents
├── config/
│   └── personas.json            # Generated persona configurations
├── data/
│   ├── loan_approval_dataset 2.csv  # Training dataset
│   └── test_data.csv           # Test cases for evaluation
├── results/
│   ├── simulation_results_*.csv # Detailed simulation results
│   └── analysis_*.json         # Aggregated analysis results
├── main.py                     # Main simulation controller
├── ml_model.py                 # Random Forest loan approval model
├── persona_config.py           # Persona configuration and psychology
├── psychological_scales.py     # NCS-18 and GSE-10 implementation
├── thinking_modes.py           # Dual process theory implementation
└── analyze_persona_decisions.py # Comprehensive analysis tool
```

## 🔬 Psychological Theory Implementation

### Dual Process Theory
- **Fast Thinking (System 1)**: Intuitive, emotion-based decisions
- **Slow Thinking (System 2)**: Deliberate, analytical reasoning
- **Mode Selection**: Based on persona characteristics and task complexity

### Psychological Scales
- **NCS-18**: 18-item Need for Cognition Scale measuring thinking preference
- **GSE-10**: 10-item General Self-Efficacy Scale measuring confidence
- **Reverse Scoring**: Proper implementation of psychological measurement

### AI Trust Modeling
- **5-point scale**: From very distrusting (-2) to very trusting (+2)
- **Emotional reactions**: Authentic responses to AI recommendations
- **Behavioral tendencies**: Consistent with psychological profiles

## 📈 Analysis Tools

### Automated Analysis
```bash
python analyze_persona_decisions.py results/simulation_results_latest.csv
```

**Generates**:
- AI acceptance rates by persona type
- Decision switching patterns
- Psychological factor correlations
- Detailed JSON reports

### Key Metrics
- **Acceptance Rate**: Percentage accepting AI recommendations
- **Switching Rate**: Percentage changing initial decisions
- **Processing Time**: Average time per decision
- **Confidence Levels**: Before/after AI recommendation

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### Model Configuration
```python
self.llm = ChatOpenAI(
    base_url='https://api.openai-proxy.org/v1',
    model="gpt-3.5-turbo",
    temperature=0.7
)
```

### Persona Generation
- **3 expertise levels**: Beginner, Intermediate, Expert
- **3 AI trust levels**: Very Distrusting, Neutral, Very Trusting  
- **3 cognition levels**: Low, Medium, High NCS
- **3 self-efficacy levels**: Low, Medium, High GSE

## 🎯 Research Applications

### Academic Research
- Human-AI interaction studies
- Decision-making psychology
- Technology acceptance modeling
- Cognitive bias research

### Industry Applications
- AI interface design optimization
- Personalized training systems
- Risk assessment improvement
- User experience research

## 🔮 Future Enhancements

### Technical
- [ ] Parallel processing for large-scale experiments
- [ ] Real-time decision support interface
- [ ] Multi-modal input support
- [ ] Advanced visualization dashboard

### Psychological
- [ ] Additional personality scales (Big Five, Risk Preference)
- [ ] Dynamic learning and adaptation
- [ ] Contextual factors (market conditions, stress)
- [ ] Longitudinal behavior tracking

## 📊 Performance Metrics

- **ML Model Accuracy**: 98.36% (Random Forest)
- **System Reliability**: 100% success rate
- **Average Processing**: 10-12 seconds per persona
- **Memory Efficiency**: Linear scaling with persona count
- **Reproducibility**: Consistent results with same random seeds

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style requirements
- Testing procedures
- Documentation standards
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citations

If you use this work in your research, please cite:

```bibtex
@software{llm_human_behavior_simulation,
  title={LLM Agents for Simulating Human Decision-Making Behavior},
  author={[Author Names]},
  year={2025},
  url={https://github.com/[username]/LLM-Human-Behavior-Simulation}
}
```

## 🙏 Acknowledgments

- LangChain framework for agent orchestration
- OpenAI for language model capabilities
- Psychological research community for theoretical foundations
- Open source community for tools and libraries

## 📞 Contact

For questions, suggestions, or collaborations:
- Create an issue in this repository
- Contact: [contact information]

---

**Project Status**: ✅ Production Ready  
**Last Updated**: July 2025  
**Version**: 2.0 (Enhanced Psychological Modeling)