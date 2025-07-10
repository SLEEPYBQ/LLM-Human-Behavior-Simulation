from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import random

class ScaleLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class ScaleItem:
    """Individual item in a psychological scale"""
    item_id: int
    text: str
    reverse_scored: bool = False

@dataclass
class ScaleResponse:
    """Response to a scale item"""
    item_id: int
    response: int  # 1-5 for most scales
    
@dataclass
class ScaleResult:
    """Result of a completed scale"""
    scale_name: str
    total_score: int
    max_score: int
    min_score: int
    level: ScaleLevel
    interpretation: str

class NeedForCognitionScale:
    """
    Need for Cognition Scale (NCS-18)
    
    Measures the tendency to engage in and enjoy thinking.
    Items are rated on a 5-point scale (1 = Extremely Uncharacteristic, 5 = Extremely Characteristic)
    """
    
    def __init__(self):
        self.items = [
            ScaleItem(1, "I would prefer complex to simple problems.", False),
            ScaleItem(2, "I like to have the responsibility of handling a situation that requires a lot of thinking.", False),
            ScaleItem(3, "Thinking is not my idea of fun.", True),
            ScaleItem(4, "I would rather do something that requires little thought than something that is sure to challenge my thinking abilities.", True),
            ScaleItem(5, "I try to anticipate and avoid situations where there is likely a chance I will have to think in depth about something.", True),
            ScaleItem(6, "I find satisfaction in deliberating hard and for long hours.", False),
            ScaleItem(7, "I only think as hard as I have to.", True),
            ScaleItem(8, "I prefer to think about small, daily projects to long-term ones.", True),
            ScaleItem(9, "I like tasks that require little thought once I've learned them.", True),
            ScaleItem(10, "The idea of relying on thought to make my way to the top appeals to me.", False),
            ScaleItem(11, "I really enjoy a task that involves coming up with new solutions to problems.", False),
            ScaleItem(12, "Learning new ways to think doesn't excite me very much.", True),
            ScaleItem(13, "I prefer my life to be filled with puzzles that I must solve.", False),
            ScaleItem(14, "The notion of thinking abstractly is appealing to me.", False),
            ScaleItem(15, "I would prefer a task that is intellectual, difficult, and important to one that is somewhat important but does not require much thought.", False),
            ScaleItem(16, "I feel relief rather than satisfaction after completing a task that required a lot of mental effort.", True),
            ScaleItem(17, "It's enough for me that something gets the job done; I don't care how or why it works.", True),
            ScaleItem(18, "I usually end up deliberating about issues even when they do not affect me personally.", False)
        ]
        
        self.reverse_scored_items = [3, 4, 5, 7, 8, 9, 12, 16, 17]
    
    def calculate_score(self, responses: List[ScaleResponse]) -> ScaleResult:
        """Calculate NCS score from responses"""
        
        total_score = 0
        for response in responses:
            item = next((item for item in self.items if item.item_id == response.item_id), None)
            if item:
                if item.reverse_scored:
                    # Reverse score: 1->5, 2->4, 3->3, 4->2, 5->1
                    score = 6 - response.response
                else:
                    score = response.response
                total_score += score
        
        # Determine level based on research norms
        if total_score >= 75:
            level = ScaleLevel.HIGH
            interpretation = "High need for cognition - enjoys thinking and seeks complex problems"
        elif total_score >= 60:
            level = ScaleLevel.MEDIUM
            interpretation = "Medium need for cognition - moderate enjoyment of thinking"
        else:
            level = ScaleLevel.LOW
            interpretation = "Low need for cognition - prefers simple tasks and avoids complex thinking"
        
        return ScaleResult(
            scale_name="Need for Cognition Scale (NCS-18)",
            total_score=total_score,
            max_score=90,
            min_score=18,
            level=level,
            interpretation=interpretation
        )
    
    def generate_realistic_responses(self, target_level: ScaleLevel) -> List[ScaleResponse]:
        """Generate realistic responses for a given target level"""
        
        responses = []
        
        # Define target score ranges
        if target_level == ScaleLevel.HIGH:
            target_range = (75, 85)
        elif target_level == ScaleLevel.MEDIUM:
            target_range = (60, 74)
        else:  # LOW
            target_range = (35, 59)
        
        target_score = random.randint(*target_range)
        avg_score_per_item = target_score / 18
        
        for item in self.items:
            # Generate response around the target average with some variation
            base_response = max(1, min(5, round(avg_score_per_item + random.uniform(-1, 1))))
            
            # Apply reverse scoring logic during generation
            if item.reverse_scored:
                # For reverse items, we want the opposite response pattern
                if target_level == ScaleLevel.HIGH:
                    # High NFC should disagree with reverse items
                    response = random.choice([1, 2, 3])
                elif target_level == ScaleLevel.LOW:
                    # Low NFC should agree with reverse items
                    response = random.choice([3, 4, 5])
                else:
                    # Medium NFC should be neutral
                    response = random.choice([2, 3, 4])
            else:
                # For normal items, response aligns with target level
                if target_level == ScaleLevel.HIGH:
                    response = random.choice([3, 4, 5])
                elif target_level == ScaleLevel.LOW:
                    response = random.choice([1, 2, 3])
                else:
                    response = random.choice([2, 3, 4])
            
            responses.append(ScaleResponse(item_id=item.item_id, response=response))
        
        return responses

class GeneralSelfEfficacyScale:
    """
    General Self-Efficacy Scale (GSE-10)
    
    Measures belief in one's ability to cope with difficult situations.
    Items are rated on a 4-point scale (1 = Not at all true, 4 = Exactly true)
    """
    
    def __init__(self):
        self.items = [
            ScaleItem(1, "I can always manage to solve difficult problems if I try hard enough.", False),
            ScaleItem(2, "If someone opposes me, I can find the means and ways to get what I want.", False),
            ScaleItem(3, "It is easy for me to stick to my aims and accomplish my goals.", False),
            ScaleItem(4, "I am confident that I could deal efficiently with unexpected events.", False),
            ScaleItem(5, "Thanks to my resourcefulness, I know how to handle unforeseen situations.", False),
            ScaleItem(6, "I can solve most problems if I invest the necessary effort.", False),
            ScaleItem(7, "I can remain calm when facing difficulties because I can rely on my coping abilities.", False),
            ScaleItem(8, "When I am confronted with a problem, I can usually find several solutions.", False),
            ScaleItem(9, "If I am in trouble, I can usually think of a solution.", False),
            ScaleItem(10, "I can usually handle whatever comes my way.", False)
        ]
    
    def calculate_score(self, responses: List[ScaleResponse]) -> ScaleResult:
        """Calculate GSE score from responses"""
        
        total_score = sum(response.response for response in responses)
        
        # Determine level based on research norms
        if total_score >= 35:
            level = ScaleLevel.HIGH
            interpretation = "High self-efficacy - strong belief in ability to handle challenges"
        elif total_score >= 30:
            level = ScaleLevel.MEDIUM
            interpretation = "Medium self-efficacy - moderate confidence in abilities"
        else:
            level = ScaleLevel.LOW
            interpretation = "Low self-efficacy - limited confidence in ability to handle challenges"
        
        return ScaleResult(
            scale_name="General Self-Efficacy Scale (GSE-10)",
            total_score=total_score,
            max_score=40,
            min_score=10,
            level=level,
            interpretation=interpretation
        )
    
    def generate_realistic_responses(self, target_level: ScaleLevel) -> List[ScaleResponse]:
        """Generate realistic responses for a given target level"""
        
        responses = []
        
        # Define target score ranges
        if target_level == ScaleLevel.HIGH:
            target_range = (35, 40)
        elif target_level == ScaleLevel.MEDIUM:
            target_range = (30, 34)
        else:  # LOW
            target_range = (20, 29)
        
        target_score = random.randint(*target_range)
        avg_score_per_item = target_score / 10
        
        for item in self.items:
            # Generate response around the target average with some variation
            base_response = max(1, min(4, round(avg_score_per_item + random.uniform(-0.5, 0.5))))
            
            # Add some realistic variation
            if target_level == ScaleLevel.HIGH:
                response = random.choice([3, 4, 4])  # Mostly high responses
            elif target_level == ScaleLevel.LOW:
                response = random.choice([1, 2, 2])  # Mostly low responses
            else:
                response = random.choice([2, 3, 3])  # Mostly medium responses
            
            responses.append(ScaleResponse(item_id=item.item_id, response=response))
        
        return responses

class ScaleManager:
    """Manager for psychological scales"""
    
    def __init__(self):
        self.ncs = NeedForCognitionScale()
        self.gse = GeneralSelfEfficacyScale()
    
    def generate_persona_scales(self, ncs_level: ScaleLevel, gse_level: ScaleLevel) -> Dict[str, Any]:
        """Generate scale scores for a persona"""
        
        # Generate responses
        ncs_responses = self.ncs.generate_realistic_responses(ncs_level)
        gse_responses = self.gse.generate_realistic_responses(gse_level)
        
        # Calculate scores
        ncs_result = self.ncs.calculate_score(ncs_responses)
        gse_result = self.gse.calculate_score(gse_responses)
        
        return {
            'need_for_cognition': {
                'score': ncs_result.total_score,
                'level': ncs_result.level.value,
                'interpretation': ncs_result.interpretation,
                'responses': [{'item_id': r.item_id, 'response': r.response} for r in ncs_responses]
            },
            'general_self_efficacy': {
                'score': gse_result.total_score,
                'level': gse_result.level.value,
                'interpretation': gse_result.interpretation,
                'responses': [{'item_id': r.item_id, 'response': r.response} for r in gse_responses]
            }
        }
    
    def get_scale_levels_for_cognition(self, cognition_number: int) -> ScaleLevel:
        """Convert old cognition number to scale level"""
        if cognition_number <= 2:
            return ScaleLevel.LOW
        elif cognition_number <= 4:
            return ScaleLevel.MEDIUM
        else:
            return ScaleLevel.HIGH

# Example usage and testing
if __name__ == "__main__":
    # Test the scales
    scale_manager = ScaleManager()
    
    # Test all combinations
    levels = [ScaleLevel.LOW, ScaleLevel.MEDIUM, ScaleLevel.HIGH]
    
    for ncs_level in levels:
        for gse_level in levels:
            print(f"\n=== Testing NCS: {ncs_level.value.upper()}, GSE: {gse_level.value.upper()} ===")
            
            scales = scale_manager.generate_persona_scales(ncs_level, gse_level)
            
            ncs_data = scales['need_for_cognition']
            gse_data = scales['general_self_efficacy']
            
            print(f"NCS Score: {ncs_data['score']}/90 ({ncs_data['level']})")
            print(f"GSE Score: {gse_data['score']}/40 ({gse_data['level']})")
            print(f"NCS: {ncs_data['interpretation']}")
            print(f"GSE: {gse_data['interpretation']}")