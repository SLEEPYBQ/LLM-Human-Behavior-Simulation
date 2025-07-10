from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from psychological_scales import ScaleManager, ScaleLevel

class ExpertiseLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

class AITrustLevel(Enum):
    VERY_DISTRUSTING = -2
    SOMEWHAT_DISTRUSTING = -1
    NEUTRAL = 0
    SOMEWHAT_TRUSTING = 1
    VERY_TRUSTING = 2

class DecisionChoice(Enum):
    ACCEPT_AI = "ACCEPT"
    REJECT_AI = "REJECT"

@dataclass
class LoanApplication:
    loan_id: int
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int
    
    def to_dict(self):
        # Convert numpy types to native Python types for JSON serialization
        result = {}
        for key, value in asdict(self).items():
            if hasattr(value, 'item'):  # numpy type
                result[key] = value.item()
            else:
                result[key] = value
        return result
    
    def get_debt_to_income_ratio(self):
        """Calculate monthly debt-to-income ratio"""
        monthly_income = self.income_annum / 12
        monthly_payment = self.loan_amount / (self.loan_term * 12)  # Simplified calculation
        return (monthly_payment / monthly_income) * 100 if monthly_income > 0 else 0

@dataclass
class AIRecommendation:
    prediction: str  # "Approved" or "Rejected"
    confidence: float
    probability_approved: float

@dataclass
class PersonaConfig:
    expertise_level: ExpertiseLevel
    ai_trust_level: AITrustLevel
    need_for_cognition: int  # 1-7 scale (for backward compatibility)
    knowledge_base: Dict[str, Any]
    
    # New psychological scales
    need_for_cognition_scale: Dict[str, Any] = None
    general_self_efficacy_scale: Dict[str, Any] = None
    thinking_mode_preference: str = "adaptive"  # "fast", "slow", "adaptive"
    
    def get_knowledge_completeness(self) -> float:
        """Return knowledge completeness based on expertise level"""
        if self.expertise_level == ExpertiseLevel.BEGINNER:
            return 0.0
        elif self.expertise_level == ExpertiseLevel.INTERMEDIATE:
            return 0.3
        else:  # EXPERT
            return 0.8
    
    def get_ncs_level(self) -> ScaleLevel:
        """Get Need for Cognition Scale level"""
        if self.need_for_cognition_scale:
            level_str = self.need_for_cognition_scale.get('level', 'medium')
            return ScaleLevel(level_str)
        else:
            # Fallback to old system
            if self.need_for_cognition <= 2:
                return ScaleLevel.LOW
            elif self.need_for_cognition <= 4:
                return ScaleLevel.MEDIUM
            else:
                return ScaleLevel.HIGH
    
    def get_gse_level(self) -> ScaleLevel:
        """Get General Self-Efficacy Scale level"""
        if self.general_self_efficacy_scale:
            level_str = self.general_self_efficacy_scale.get('level', 'medium')
            return ScaleLevel(level_str)
        else:
            # Default to medium if not set
            return ScaleLevel.MEDIUM
    
    def get_persona_description(self) -> str:
        """Get comprehensive persona description"""
        expertise_desc = {
            ExpertiseLevel.BEGINNER: "Beginner with 1 month experience",
            ExpertiseLevel.INTERMEDIATE: "Intermediate with 2 years experience",
            ExpertiseLevel.EXPERT: "Expert with 10+ years experience"
        }
        
        trust_desc = {
            AITrustLevel.VERY_DISTRUSTING: "Very distrusting of AI",
            AITrustLevel.SOMEWHAT_DISTRUSTING: "Somewhat distrusting of AI",
            AITrustLevel.NEUTRAL: "Neutral towards AI",
            AITrustLevel.SOMEWHAT_TRUSTING: "Somewhat trusting of AI",
            AITrustLevel.VERY_TRUSTING: "Very trusting of AI"
        }
        
        ncs_level = self.get_ncs_level()
        gse_level = self.get_gse_level()
        
        return f"{expertise_desc[self.expertise_level]}, {trust_desc[self.ai_trust_level]}, {ncs_level.value} need for cognition, {gse_level.value} self-efficacy"

@dataclass
class CognitionState:
    initial_decision: str
    confidence_level: float
    reasoning_process: List[str]
    memory_items: List[str]
    reflection_points: List[str]

@dataclass
class UtilityEvaluation:
    initial_reaction: str
    comparison_with_judgment: str
    ai_capability_assessment: str
    risk_benefit_consideration: str
    inner_conflict: str

@dataclass
class DecisionResult:
    final_decision: DecisionChoice
    confidence: float
    reasoning: str
    utility_evaluation: UtilityEvaluation
    cognition_state: CognitionState
    processing_time: float

class KnowledgeBase:
    """Knowledge base for loan approval criteria"""
    
    @staticmethod
    def get_base_knowledge() -> Dict[str, Any]:
        return {
            "cibil_score_ranges": {
                "excellent": 750,
                "good": 650,
                "average": 550,
                "high_risk": 550
            },
            "debt_to_income_limit": 40,
            "asset_preferences": {
                "residential": "most_stable",
                "commercial": "volatile", 
                "luxury": "high_risk"
            },
            "employment_risk": {
                "salaried": "low_risk",
                "self_employed": "high_risk"
            },
            "loan_term_impact": "long_term_increases_default_risk",
            "dependents_impact": "affects_repayment_capacity"
        }
    
    @staticmethod
    def get_expertise_knowledge(level: ExpertiseLevel) -> Dict[str, Any]:
        base = KnowledgeBase.get_base_knowledge()
        
        if level == ExpertiseLevel.BEGINNER:
            # Limited knowledge - only basic concepts
            return {
                "cibil_score_ranges": {
                    "good": 650,
                    "bad": 550
                },
                "basic_concepts": "income_and_employment_important"
            }
        elif level == ExpertiseLevel.INTERMEDIATE:
            # Partial knowledge
            filtered_base = base.copy()
            filtered_base.pop("loan_term_impact", None)  # Missing some advanced concepts
            return filtered_base
        else:  # EXPERT
            # Complete knowledge with additional insights
            base.update({
                "advanced_risk_factors": {
                    "debt_consolidation_patterns": "indicator_of_financial_stress",
                    "asset_diversification": "reduces_overall_risk",
                    "industry_specific_risks": "consider_economic_cycles"
                },
                "regulatory_compliance": "follow_fair_lending_practices"
            })
            return base

def create_persona_configs() -> List[PersonaConfig]:
    """Create different persona configurations for testing"""
    personas = []
    scale_manager = ScaleManager()
    
    # Expertise levels
    expertise_levels = [ExpertiseLevel.BEGINNER, ExpertiseLevel.INTERMEDIATE, ExpertiseLevel.EXPERT]
    
    # AI trust levels
    trust_levels = [AITrustLevel.VERY_DISTRUSTING, AITrustLevel.NEUTRAL, AITrustLevel.VERY_TRUSTING]
    
    # Scale levels for NCS and GSE
    scale_levels = [ScaleLevel.LOW, ScaleLevel.MEDIUM, ScaleLevel.HIGH]
    
    # Need for cognition levels (1-7 scale) - for backward compatibility
    cognition_levels = [2, 4, 6]  # Low, medium, high
    
    for expertise in expertise_levels:
        for trust in trust_levels:
            for i, ncs_level in enumerate(scale_levels):
                for gse_level in scale_levels:
                    knowledge_base = KnowledgeBase.get_expertise_knowledge(expertise)
                    
                    # Generate psychological scales
                    scales = scale_manager.generate_persona_scales(ncs_level, gse_level)
                    
                    # Determine thinking mode preference based on scales
                    if ncs_level == ScaleLevel.HIGH and gse_level == ScaleLevel.HIGH:
                        thinking_mode = "slow"  # Prefers deep thinking
                    elif ncs_level == ScaleLevel.LOW and gse_level == ScaleLevel.LOW:
                        thinking_mode = "fast"  # Prefers quick decisions
                    else:
                        thinking_mode = "adaptive"  # Adapts based on situation
                    
                    persona = PersonaConfig(
                        expertise_level=expertise,
                        ai_trust_level=trust,
                        need_for_cognition=cognition_levels[i],  # Backward compatibility
                        knowledge_base=knowledge_base,
                        need_for_cognition_scale=scales['need_for_cognition'],
                        general_self_efficacy_scale=scales['general_self_efficacy'],
                        thinking_mode_preference=thinking_mode
                    )
                    personas.append(persona)
    
    return personas

def save_persona_configs(personas: List[PersonaConfig], filepath: str):
    """Save persona configurations to JSON file"""
    # Convert to serializable format
    serializable_personas = []
    for persona in personas:
        persona_dict = {
            'expertise_level': persona.expertise_level.value,
            'ai_trust_level': persona.ai_trust_level.value,
            'need_for_cognition': persona.need_for_cognition,
            'knowledge_base': persona.knowledge_base,
            'need_for_cognition_scale': persona.need_for_cognition_scale,
            'general_self_efficacy_scale': persona.general_self_efficacy_scale,
            'thinking_mode_preference': persona.thinking_mode_preference
        }
        serializable_personas.append(persona_dict)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_personas, f, indent=2)
    
    print(f"Saved {len(personas)} persona configurations to {filepath}")

if __name__ == "__main__":
    # Create and save persona configurations
    personas = create_persona_configs()
    save_persona_configs(personas, '/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP/config/personas.json')
    
    print(f"Created {len(personas)} different persona combinations:")
    for i, persona in enumerate(personas[:5]):  # Show first 5
        print(f"{i+1}. {persona.get_persona_description()}")
        print(f"   NCS Score: {persona.need_for_cognition_scale['score']}/90")
        print(f"   GSE Score: {persona.general_self_efficacy_scale['score']}/40")
        print(f"   Thinking Mode: {persona.thinking_mode_preference}")
        print()