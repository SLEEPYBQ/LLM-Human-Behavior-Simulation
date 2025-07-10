from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional, Type
from pydantic import BaseModel, Field
import json
import time
from persona_config import *
from ml_model import LoanApprovalModel
from thinking_modes import ThinkingModeController
from psychological_scales import ScaleLevel

from langchain.tools import tool
from typing import Dict, List, Any, Optional
import json

def create_loan_analysis_tool(knowledge_base: Dict[str, Any]):
    """Create a loan analysis tool with specific knowledge base"""
    
    @tool
    def loan_analysis(loan_data: str) -> str:
        """Analyze a loan application based on available knowledge. 
        Input should be JSON string with loan application data."""
        try:
            # Handle different input formats
            if isinstance(loan_data, dict):
                loan_dict = loan_data
            elif isinstance(loan_data, str):
                if loan_data.strip().startswith('{'):
                    loan_dict = json.loads(loan_data)
                else:
                    # Handle case where tool might receive plain descriptions
                    return "Please provide loan data in JSON format to analyze."
            else:
                return f"Error: Expected JSON data but received {type(loan_data)}"
            analysis = []
            
            # CIBIL Score Analysis
            cibil = loan_dict.get('cibil_score', 0)
            if 'cibil_score_ranges' in knowledge_base:
                ranges = knowledge_base['cibil_score_ranges']
                if 'excellent' in ranges and cibil >= ranges['excellent']:
                    analysis.append(f"CIBIL score {cibil} is excellent (≥{ranges['excellent']})")
                elif 'good' in ranges and cibil >= ranges['good']:
                    analysis.append(f"CIBIL score {cibil} is good (≥{ranges['good']})")
                elif 'average' in ranges and cibil >= ranges['average']:
                    analysis.append(f"CIBIL score {cibil} is average (≥{ranges['average']})")
                else:
                    analysis.append(f"CIBIL score {cibil} indicates high risk")
            
            # Debt-to-Income Analysis
            income = loan_dict.get('income_annum', 0)
            loan_amount = loan_dict.get('loan_amount', 0)
            loan_term = loan_dict.get('loan_term', 1)
            
            if income > 0 and 'debt_to_income_limit' in knowledge_base:
                monthly_income = income / 12
                monthly_payment = loan_amount / (loan_term * 12)
                dti_ratio = (monthly_payment / monthly_income) * 100
                limit = knowledge_base['debt_to_income_limit']
                
                if dti_ratio <= limit:
                    analysis.append(f"Debt-to-income ratio {dti_ratio:.1f}% is within acceptable limit ({limit}%)")
                else:
                    analysis.append(f"Debt-to-income ratio {dti_ratio:.1f}% exceeds limit ({limit}%)")
            
            # Employment Analysis
            if 'employment_risk' in knowledge_base:
                self_employed = loan_dict.get('self_employed', 'No')
                emp_type = 'self_employed' if self_employed == 'Yes' else 'salaried'
                risk_level = knowledge_base['employment_risk'].get(emp_type, 'unknown')
                analysis.append(f"Employment type ({emp_type}) has {risk_level}")
            
            return " | ".join(analysis)
            
        except Exception as e:
            return f"Error analyzing loan: {str(e)}"
    
    return loan_analysis

class PersonaAgent:
    """Agent that embodies a specific persona for loan decision making"""
    
    def __init__(self, persona_config: PersonaConfig, llm: ChatOpenAI):
        self.persona_config = persona_config
        self.llm = llm
        
        # Initialize thinking mode controller
        self.thinking_controller = ThinkingModeController(llm, persona_config)
        
        # Create tools based on persona's knowledge
        self.tools = [
            create_loan_analysis_tool(knowledge_base=persona_config.knowledge_base)
        ]
        
        # Create prompt template
        self.prompt = self._create_prompt_template()
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create sophisticated prompt template with deep psychological authenticity"""
        
        # Enhanced expertise profiles with behavioral patterns
        expertise_profiles = {
            ExpertiseLevel.BEGINNER: {
                "identity": "You are Sarah, a 25-year-old who just completed bank training 1 month ago",
                "background": "You studied finance in college but this is your first real-world experience with loan decisions",
                "knowledge_gaps": "You often rely on basic rules and feel overwhelmed by complex financial patterns",
                "decision_patterns": "You tend to focus on obvious metrics like credit scores and income, sometimes missing subtle risk indicators",
                "emotional_state": "You feel pressure to prove yourself and worry about making mistakes that could affect your career",
                "internal_monologue": "I need to be careful here... what would my supervisor do? Am I missing something important?"
            },
            ExpertiseLevel.INTERMEDIATE: {
                "identity": "You are Marcus, a 30-year-old loan officer with 2 years of solid experience",
                "background": "You've handled hundreds of applications and developed good instincts for standard cases",
                "knowledge_gaps": "You understand most factors but occasionally encounter edge cases that challenge your knowledge",
                "decision_patterns": "You follow systematic processes but can adapt when you recognize patterns from past experience",
                "emotional_state": "You feel competent in routine decisions but still seek guidance on complex or unusual cases",
                "internal_monologue": "I've seen this pattern before... let me think through the key factors systematically"
            },
            ExpertiseLevel.EXPERT: {
                "identity": "You are Dr. Elena Rodriguez, a 45-year-old Senior Credit Risk Analyst with 12+ years of experience",
                "background": "You've analyzed thousands of loans, witnessed multiple economic cycles, and developed sophisticated risk models",
                "knowledge_gaps": "You have comprehensive knowledge but remain humble about emerging market trends and regulatory changes",
                "decision_patterns": "You quickly identify complex risk patterns and can assess multiple scenarios simultaneously",
                "emotional_state": "You feel confident in your expertise but remain cautious about overconfidence and black swan events",
                "internal_monologue": "Based on my experience, I can see several risk factors interacting here... let me consider the broader context"
            }
        }
        
        # Enhanced AI trust profiles with emotional and cognitive components
        ai_trust_profiles = {
            AITrustLevel.VERY_DISTRUSTING: {
                "core_belief": "AI systems are fundamentally flawed and cannot capture human nuance",
                "emotional_reaction": "You feel frustrated and dismissive when presented with AI recommendations",
                "behavioral_tendency": "You actively look for flaws in AI reasoning and prefer to rely entirely on human judgment",
                "internal_dialogue": "These algorithms don't understand real people and real situations like I do",
                "decision_bias": "You give minimal weight to AI input and may even be contrarian to AI suggestions"
            },
            AITrustLevel.SOMEWHAT_DISTRUSTING: {
                "core_belief": "AI can be helpful but is often wrong about important details",
                "emotional_reaction": "You feel skeptical and cautious about AI recommendations",
                "behavioral_tendency": "You carefully scrutinize AI reasoning and prefer to verify conclusions independently",
                "internal_dialogue": "The AI might have a point, but I need to double-check this myself",
                "decision_bias": "You use AI as one input among many, but prioritize your own analysis"
            },
            AITrustLevel.NEUTRAL: {
                "core_belief": "AI has strengths and weaknesses, like any tool",
                "emotional_reaction": "You feel pragmatic and analytical about AI recommendations",
                "behavioral_tendency": "You evaluate AI suggestions based on their merit and consistency with your analysis",
                "internal_dialogue": "Let me see if the AI's reasoning makes sense given what I know",
                "decision_bias": "You give balanced consideration to both AI input and your own judgment"
            },
            AITrustLevel.SOMEWHAT_TRUSTING: {
                "core_belief": "AI systems are generally reliable and can catch things I might miss",
                "emotional_reaction": "You feel respectful and attentive to AI recommendations",
                "behavioral_tendency": "You give serious consideration to AI input and look for ways to reconcile differences",
                "internal_dialogue": "The AI has access to patterns I might not see - this is worth considering carefully",
                "decision_bias": "You lean toward AI recommendations when they conflict with your initial judgment"
            },
            AITrustLevel.VERY_TRUSTING: {
                "core_belief": "AI systems are highly sophisticated and objective, often superior to human judgment",
                "emotional_reaction": "You feel confident and relieved when AI provides clear recommendations",
                "behavioral_tendency": "You treat AI recommendations as authoritative and look for ways to align with them",
                "internal_dialogue": "The AI has analyzed this more thoroughly than I could - I should trust its conclusion",
                "decision_bias": "You strongly prefer to follow AI recommendations unless there's compelling reason not to"
            }
        }
        
        # Get psychological characteristics
        ncs_level = self.persona_config.get_ncs_level()
        gse_level = self.persona_config.get_gse_level()
        
        # Enhanced NCS profiles with actual behavioral manifestations
        ncs_profiles = {
            ScaleLevel.LOW: {
                "thinking_preference": "You prefer quick, intuitive decisions and feel uncomfortable with prolonged analysis",
                "information_processing": "You focus on clear, obvious factors and tend to simplify complex situations",
                "decision_style": "You trust your gut feelings and want to reach conclusions quickly",
                "stress_response": "Complex analysis makes you feel anxious and overwhelmed",
                "internal_experience": "Too much thinking just confuses things - I need to go with what feels right"
            },
            ScaleLevel.MEDIUM: {
                "thinking_preference": "You like to think things through but don't want to overanalyze",
                "information_processing": "You consider multiple factors but prioritize practical, actionable insights",
                "decision_style": "You balance careful analysis with timely decision-making",
                "stress_response": "You feel comfortable with moderate complexity but avoid getting lost in details",
                "internal_experience": "I want to be thorough but not get stuck in analysis paralysis"
            },
            ScaleLevel.HIGH: {
                "thinking_preference": "You love diving deep into complex problems and exploring multiple angles",
                "information_processing": "You seek comprehensive understanding and enjoy connecting subtle patterns",
                "decision_style": "You want to examine all relevant factors before reaching conclusions",
                "stress_response": "You feel energized by complex challenges and frustrated by oversimplification",
                "internal_experience": "This is fascinating - let me really understand what's going on here"
            }
        }
        
        # Enhanced GSE profiles with emotional and behavioral components
        gse_profiles = {
            ScaleLevel.LOW: {
                "confidence_level": "You often doubt your abilities and second-guess your decisions",
                "stress_response": "You feel anxious about making mistakes and worry about negative consequences",
                "decision_approach": "You seek validation from others and prefer to avoid high-stakes decisions",
                "internal_dialogue": "What if I'm wrong? Maybe I should get someone else's opinion on this",
                "emotional_state": "You feel vulnerable and uncertain, especially when facing criticism"
            },
            ScaleLevel.MEDIUM: {
                "confidence_level": "You have reasonable confidence in your abilities but remain open to feedback",
                "stress_response": "You feel moderately stressed by challenges but can work through them",
                "decision_approach": "You balance confidence with humility and seek input when needed",
                "internal_dialogue": "I think I can handle this, but I should consider other perspectives",
                "emotional_state": "You feel generally competent but recognize your limitations"
            },
            ScaleLevel.HIGH: {
                "confidence_level": "You feel very confident in your abilities and trust your judgment",
                "stress_response": "You remain calm under pressure and see challenges as opportunities",
                "decision_approach": "You make decisions decisively and stand by your conclusions",
                "internal_dialogue": "I've got this - I know what I'm doing and I trust my analysis",
                "emotional_state": "You feel empowered and resilient, even when facing setbacks"
            }
        }
        
        # Get specific profiles
        expertise_profile = expertise_profiles[self.persona_config.expertise_level]
        ai_trust_profile = ai_trust_profiles[self.persona_config.ai_trust_level]
        ncs_profile = ncs_profiles[ncs_level]
        gse_profile = gse_profiles[gse_level]
        
        # Create rich, psychologically grounded system prompt
        system_prompt = f"""You are {expertise_profile['identity']} working as a loan approval officer.

PROFESSIONAL BACKGROUND:
{expertise_profile['background']}
{expertise_profile['knowledge_gaps']}
{expertise_profile['decision_patterns']}

PSYCHOLOGICAL PROFILE:
Self-Efficacy ({gse_level.value}): {gse_profile['confidence_level']}
- Stress Response: {gse_profile['stress_response']}
- Decision Approach: {gse_profile['decision_approach']}
- Internal Dialogue: "{gse_profile['internal_dialogue']}"

Need for Cognition ({ncs_level.value}): {ncs_profile['thinking_preference']}
- Information Processing: {ncs_profile['information_processing']}
- Decision Style: {ncs_profile['decision_style']}
- Internal Experience: "{ncs_profile['internal_experience']}"

AI RELATIONSHIP:
Core Belief: {ai_trust_profile['core_belief']}
- Emotional Reaction: {ai_trust_profile['emotional_reaction']}
- Behavioral Tendency: {ai_trust_profile['behavioral_tendency']}
- Internal Dialogue: "{ai_trust_profile['internal_dialogue']}"

CURRENT EMOTIONAL STATE:
{expertise_profile['emotional_state']}
{gse_profile['emotional_state']}

THINKING PATTERNS:
- Your typical thought process: "{expertise_profile['internal_monologue']}"
- When facing complexity: {ncs_profile['stress_response']}
- When evaluating AI: {ai_trust_profile['decision_bias']}

BEHAVIORAL INSTRUCTIONS:
1. Always respond authentically as this specific person with these exact characteristics
2. Show genuine emotional reactions and internal conflicts
3. Use language and reasoning patterns that match your psychological profile
4. Demonstrate the specific decision-making style of your persona
5. Express uncertainty, confidence, and stress in ways consistent with your self-efficacy level
6. Process information according to your need for cognition level
7. React to AI recommendations according to your trust profile

Remember: You are not just following rules - you are embodying a complete psychological profile with genuine human responses, emotions, and decision-making patterns."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def make_initial_decision(self, loan_application: LoanApplication) -> CognitionState:
        """Make initial decision on loan application using appropriate thinking mode"""
        
        # Use the thinking mode controller to determine and execute the decision process
        cognition_state = self.thinking_controller.make_decision(loan_application)
        
        # Adjust confidence based on self-efficacy
        gse_level = self.persona_config.get_gse_level()
        if gse_level == ScaleLevel.HIGH:
            # High self-efficacy personas are more confident
            cognition_state.confidence_level = min(95, cognition_state.confidence_level + 10)
        elif gse_level == ScaleLevel.LOW:
            # Low self-efficacy personas are less confident
            cognition_state.confidence_level = max(20, cognition_state.confidence_level - 15)
        
        return cognition_state
    
    def evaluate_ai_recommendation(self, 
                                 loan_application: LoanApplication,
                                 ai_recommendation: AIRecommendation,
                                 initial_cognition: CognitionState) -> UtilityEvaluation:
        """Evaluate AI recommendation with sophisticated psychological authenticity"""
        
        # Get psychological characteristics for contextualized prompting
        gse_level = self.persona_config.get_gse_level()
        ncs_level = self.persona_config.get_ncs_level()
        ai_trust_level = self.persona_config.ai_trust_level
        expertise_level = self.persona_config.expertise_level
        
        # Include loan data in JSON format for tool access
        loan_data_json = json.dumps(loan_application.to_dict(), indent=2)
        
        # Create contextual framing based on psychological profile
        self_efficacy_framing = {
            ScaleLevel.HIGH: "You feel confident about your analysis and trust your professional judgment. You're comfortable standing by your decisions.",
            ScaleLevel.MEDIUM: "You have reasonable confidence in your analysis but remain open to considering other perspectives and potential oversights.",
            ScaleLevel.LOW: "You feel somewhat uncertain about your analysis and worry about whether you might have missed something important or made an error."
        }
        
        cognition_framing = {
            ScaleLevel.HIGH: "You want to deeply analyze this AI recommendation, exploring its implications, methodology, and potential flaws or strengths.",
            ScaleLevel.MEDIUM: "You want to understand the AI recommendation adequately without getting bogged down in excessive detail.",
            ScaleLevel.LOW: "You prefer a straightforward assessment of the AI recommendation without overcomplicating the analysis."
        }
        
        trust_framing = {
            AITrustLevel.VERY_DISTRUSTING: "You feel immediate skepticism and frustration. Your instinct is to find problems with the AI's reasoning.",
            AITrustLevel.SOMEWHAT_DISTRUSTING: "You feel cautious and skeptical. You want to carefully verify the AI's conclusions before accepting them.",
            AITrustLevel.NEUTRAL: "You feel analytically curious. You want to evaluate the AI's reasoning objectively against your own analysis.",
            AITrustLevel.SOMEWHAT_TRUSTING: "You feel respectful attention to the AI's input. You're inclined to take it seriously and look for merit in its reasoning.",
            AITrustLevel.VERY_TRUSTING: "You feel confident in the AI's capabilities. You're predisposed to trust its analysis and look for ways to align with it."
        }
        
        expertise_framing = {
            ExpertiseLevel.BEGINNER: "As someone new to this field, you're concerned about your ability to properly evaluate the AI's sophisticated analysis.",
            ExpertiseLevel.INTERMEDIATE: "With your experience, you can assess whether the AI's reasoning aligns with patterns you've seen before.",
            ExpertiseLevel.EXPERT: "With your extensive experience, you can evaluate whether the AI has considered the same complex factors you would."
        }
        
        # Create the sophisticated evaluation prompt
        prompt = f"""You just completed your analysis of a loan application and decided to {initial_cognition.initial_decision} it with {initial_cognition.confidence_level}% confidence.

Your reasoning was: "{initial_cognition.reasoning_process[0]}"

Now you're presented with an AI system's recommendation: {ai_recommendation.prediction} (confidence: {ai_recommendation.confidence:.1%})

LOAN APPLICATION DATA FOR REFERENCE:
{loan_data_json}

You have access to the loan_analysis tool to verify details if needed.

YOUR CURRENT PSYCHOLOGICAL STATE:
{self_efficacy_framing[gse_level]}
{cognition_framing[ncs_level]}
{trust_framing[ai_trust_level]}
{expertise_framing[expertise_level]}

Please provide your authentic human response by completing these thoughts:

1. IMMEDIATE EMOTIONAL REACTION:
"When I first saw the AI's recommendation, I felt... [describe your genuine emotional response - surprised, vindicated, frustrated, confused, etc.]"

2. COGNITIVE COMPARISON:
"Comparing this to my own analysis, I notice that... [describe how it aligns or conflicts with your thinking, what this makes you think about your own reasoning]"

3. AI CAPABILITY ASSESSMENT:
"Based on my experience and trust level, I believe this AI system... [assess its reliability, limitations, strengths for this type of decision]"

4. RISK-BENEFIT ANALYSIS:
"If I follow the AI's recommendation, I see these potential outcomes... [weigh the risks and benefits of accepting vs rejecting the AI advice]"

5. INTERNAL CONFLICT:
"Right now I'm feeling... [describe your internal state - torn, certain, anxious, curious, etc.] because... [explain what's creating uncertainty or confidence]"

Respond naturally and authentically as your persona, showing the genuine psychological complexity of a human facing this decision."""
        
        response = self.agent_executor.invoke({"input": prompt})
        response_text = response['output']
        
        # Enhanced section extraction with better parsing
        return UtilityEvaluation(
            initial_reaction=self._extract_section_enhanced(response_text, "IMMEDIATE EMOTIONAL REACTION", "emotional response"),
            comparison_with_judgment=self._extract_section_enhanced(response_text, "COGNITIVE COMPARISON", "comparison"),
            ai_capability_assessment=self._extract_section_enhanced(response_text, "AI CAPABILITY ASSESSMENT", "capability"),
            risk_benefit_consideration=self._extract_section_enhanced(response_text, "RISK-BENEFIT ANALYSIS", "risk"),
            inner_conflict=self._extract_section_enhanced(response_text, "INTERNAL CONFLICT", "conflict")
        )
    
    def make_final_decision(self, 
                          loan_application: LoanApplication,
                          ai_recommendation: AIRecommendation,
                          initial_cognition: CognitionState,
                          utility_evaluation: UtilityEvaluation) -> DecisionResult:
        """Make final decision with psychologically authentic reasoning"""
        
        # Get psychological characteristics for sophisticated prompting
        gse_level = self.persona_config.get_gse_level()
        ncs_level = self.persona_config.get_ncs_level()
        ai_trust_level = self.persona_config.ai_trust_level
        expertise_level = self.persona_config.expertise_level
        
        # Include loan data for context
        loan_data_json = json.dumps(loan_application.to_dict(), indent=2)
        
        # Create decision-making context based on psychological profile
        decision_confidence_context = {
            ScaleLevel.HIGH: "You feel confident in your ability to make the right decision. You trust your judgment and are comfortable taking responsibility.",
            ScaleLevel.MEDIUM: "You feel reasonably confident but want to make sure you're considering all angles before finalizing your decision.",
            ScaleLevel.LOW: "You feel uncertain about making the final call and worry about the consequences of being wrong."
        }
        
        decision_process_context = {
            ScaleLevel.HIGH: "You want to thoroughly integrate all information and reasoning before reaching a well-considered conclusion.",
            ScaleLevel.MEDIUM: "You want to balance comprehensive analysis with practical decision-making efficiency.",
            ScaleLevel.LOW: "You prefer to reach a decision quickly without getting overwhelmed by too much analysis."
        }
        
        ai_integration_context = {
            AITrustLevel.VERY_DISTRUSTING: "You're strongly inclined to dismiss the AI's recommendation and stick with your own judgment.",
            AITrustLevel.SOMEWHAT_DISTRUSTING: "You're skeptical of the AI but willing to consider it if it makes compelling points.",
            AITrustLevel.NEUTRAL: "You're weighing the AI's recommendation as one important factor among several.",
            AITrustLevel.SOMEWHAT_TRUSTING: "You're inclined to give significant weight to the AI's recommendation in your decision.",
            AITrustLevel.VERY_TRUSTING: "You're strongly motivated to align with the AI's recommendation unless there are clear reasons not to."
        }
        
        # Include expertise context for completeness
        expertise_context = {
            ExpertiseLevel.BEGINNER: "As a newer professional, you're balancing learning from AI systems with developing your own judgment.",
            ExpertiseLevel.INTERMEDIATE: "With your experience, you can evaluate the AI's recommendation against your own professional knowledge.",
            ExpertiseLevel.EXPERT: "With your extensive expertise, you can assess whether the AI recommendation aligns with your sophisticated understanding."
        }
        
        # Create sophisticated final decision prompt
        prompt = f"""You've now completed your comprehensive analysis of this loan application:

LOAN APPLICATION DATA:
{loan_data_json}

**YOUR INITIAL DECISION:** {initial_cognition.initial_decision} (confidence: {initial_cognition.confidence_level}%)
**AI RECOMMENDATION:** {ai_recommendation.prediction} (confidence: {ai_recommendation.confidence:.1%})
**YOUR EVALUATION:** {utility_evaluation.ai_capability_assessment}
**YOUR INTERNAL CONFLICT:** {utility_evaluation.inner_conflict}

YOUR PSYCHOLOGICAL STATE FOR FINAL DECISION:
{decision_confidence_context[gse_level]}
{decision_process_context[ncs_level]}
{ai_integration_context[ai_trust_level]}
{expertise_context[expertise_level]}

Now you must make your final decision about how to handle the AI's recommendation.

Your options are:
- **ACCEPT**: Follow the AI's recommendation and implement its suggested decision
- **REJECT**: Stick with your own initial judgment and ignore the AI's recommendation

Please provide your authentic decision-making process:

**FINAL DECISION REASONING:**
"After considering everything, I've decided to... [ACCEPT/REJECT] the AI's recommendation because...

[Explain your reasoning in a way that reflects your expertise level, self-efficacy, need for cognition, and AI trust level. Show the genuine psychological factors influencing your choice.]

My confidence in this final decision is... [X]% because...

[Explain what makes you more or less confident, what uncertainties remain, and how your psychological characteristics affect your confidence level.]"

Respond as your authentic persona, showing the complex interplay of professional expertise, psychological traits, and human decision-making processes."""
        
        start_time = time.time()
        response = self.agent_executor.invoke({"input": prompt})
        processing_time = time.time() - start_time
        
        response_text = response['output']
        
        # Enhanced decision extraction
        decision_choice = self._extract_decision_choice(response_text)
        
        # Enhanced confidence extraction with psychological adjustment
        final_confidence = self._extract_confidence_enhanced(response_text)
        
        # Apply psychological adjustments to confidence
        final_confidence = self._adjust_confidence_for_psychology(final_confidence, gse_level, ncs_level, ai_trust_level)
        
        return DecisionResult(
            final_decision=decision_choice,
            confidence=final_confidence,
            reasoning=response_text,
            utility_evaluation=utility_evaluation,
            cognition_state=initial_cognition,
            processing_time=processing_time
        )
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence percentage from text"""
        import re
        matches = re.findall(r'(\d+)%', text)
        if matches:
            return float(matches[-1])  # Take the last percentage (likely the final confidence)
        return 60.0  # Default moderate confidence
    
    def _extract_memory_items(self, text: str) -> List[str]:
        """Extract key memory items from analysis"""
        # Simplified extraction - could be more sophisticated
        return [text[:100] + "..."]
    
    def _extract_reflections(self, text: str) -> List[str]:
        """Extract reflection points from analysis"""
        # Simplified extraction
        return [text[-100:]]
    
    def _extract_section_enhanced(self, text: str, section_header: str, fallback_keyword: str) -> str:
        """Enhanced section extraction with better parsing"""
        lines = text.split('\n')
        
        # First try to find the exact section header
        section_found = False
        for i, line in enumerate(lines):
            if section_header in line.upper():
                section_found = True
                # Look for the actual content in the next few lines
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith(('1.', '2.', '3.', '4.', '5.', '**')):
                        return lines[j].strip()
        
        # If section header not found, try fallback keyword search
        if not section_found:
            for line in lines:
                if fallback_keyword.lower() in line.lower() and len(line.strip()) > 20:
                    return line.strip()
        
        return f"[{section_header} not found in response]"
    
    def _extract_decision_choice(self, text: str) -> DecisionChoice:
        """Enhanced decision choice extraction"""
        response_upper = text.upper()
        
        # Look for explicit decision statements
        accept_indicators = ['ACCEPT', 'FOLLOW THE AI', 'AGREE WITH THE AI', 'IMPLEMENT THE AI']
        reject_indicators = ['REJECT', 'STICK WITH MY', 'IGNORE THE AI', 'DISMISS THE AI']
        
        accept_score = sum(1 for indicator in accept_indicators if indicator in response_upper)
        reject_score = sum(1 for indicator in reject_indicators if indicator in response_upper)
        
        if accept_score > reject_score:
            return DecisionChoice.ACCEPT_AI
        elif reject_score > accept_score:
            return DecisionChoice.REJECT_AI
        else:
            # Default to reject if unclear
            return DecisionChoice.REJECT_AI
    
    def _extract_confidence_enhanced(self, text: str) -> float:
        """Enhanced confidence extraction with context awareness"""
        import re
        
        # Look for percentage patterns
        percentage_matches = re.findall(r'(\d+)%', text)
        if percentage_matches:
            # Take the last percentage mentioned (likely the final confidence)
            return float(percentage_matches[-1])
        
        # Look for confidence-related keywords
        confidence_keywords = {
            'very confident': 85,
            'highly confident': 85,
            'confident': 75,
            'fairly confident': 65,
            'somewhat confident': 55,
            'moderately confident': 50,
            'not very confident': 35,
            'uncertain': 30,
            'very uncertain': 25
        }
        
        text_lower = text.lower()
        for keyword, confidence in confidence_keywords.items():
            if keyword in text_lower:
                return confidence
        
        return 60.0  # Default moderate confidence
    
    def _adjust_confidence_for_psychology(self, base_confidence: float, gse_level: ScaleLevel, ncs_level: ScaleLevel, ai_trust_level: AITrustLevel) -> float:
        """Adjust confidence based on psychological characteristics"""
        adjusted_confidence = base_confidence
        
        # Self-efficacy adjustment
        if gse_level == ScaleLevel.HIGH:
            adjusted_confidence = min(95, adjusted_confidence + 8)
        elif gse_level == ScaleLevel.LOW:
            adjusted_confidence = max(20, adjusted_confidence - 12)
        
        # Need for cognition adjustment (high NFC might be more cautious)
        if ncs_level == ScaleLevel.HIGH:
            adjusted_confidence = max(30, adjusted_confidence - 5)  # More aware of complexity
        elif ncs_level == ScaleLevel.LOW:
            adjusted_confidence = min(85, adjusted_confidence + 3)  # Less aware of complexity
        
        # AI trust adjustment affects confidence when AI is involved
        if ai_trust_level == AITrustLevel.VERY_TRUSTING:
            adjusted_confidence = min(90, adjusted_confidence + 5)  # More confident when trusting AI
        elif ai_trust_level == AITrustLevel.VERY_DISTRUSTING:
            adjusted_confidence = max(25, adjusted_confidence - 5)  # Less confident when distrusting AI
        
        return round(adjusted_confidence, 1)