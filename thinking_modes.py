from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from persona_config import LoanApplication, CognitionState, ExpertiseLevel, AITrustLevel
from psychological_scales import ScaleLevel

class ThinkingMode(Enum):
    FAST = "fast"
    SLOW = "slow"

@dataclass
class TaskSimplification:
    """Result of task simplification for fast thinking"""
    simplified_task: str
    key_factors: List[str]
    simplification_reasoning: str

@dataclass
class SlowThinkingResult:
    """Result of slow thinking multi-agent process"""
    memory_analysis: str
    reflection_analysis: str
    reasoning_analysis: str
    planning_analysis: str
    theory_of_mind_analysis: str
    final_synthesis: str

class FastThinkingAgent:
    """Agent that implements fast thinking (System 1) through task simplification"""
    
    def __init__(self, llm: ChatOpenAI, persona_config):
        self.llm = llm
        self.persona_config = persona_config
        
    def simplify_task(self, loan_application: LoanApplication) -> TaskSimplification:
        """Simplify the loan approval task for fast thinking with psychological authenticity"""
        
        loan_data = json.dumps(loan_application.to_dict(), indent=2)
        
        # Get persona characteristics for contextual fast thinking
        gse_level = self.persona_config.get_gse_level()
        ncs_level = self.persona_config.get_ncs_level()
        
        # Create fast thinking context
        fast_thinking_context = {
            ScaleLevel.HIGH: "You're confident in your ability to quickly identify key factors, but you want to avoid overthinking.",
            ScaleLevel.MEDIUM: "You feel reasonably confident about making quick assessments when needed.",
            ScaleLevel.LOW: "You feel some pressure to make quick decisions and worry about missing important details."
        }
        
        cognition_context = {
            ScaleLevel.HIGH: "Even though you're thinking fast, you can't help but notice multiple factors - but you force yourself to focus on the most obvious ones.",
            ScaleLevel.MEDIUM: "You naturally balance simplicity with adequate analysis when thinking quickly.",
            ScaleLevel.LOW: "You prefer this quick, simplified approach and feel comfortable focusing on just the basics."
        }
        
        prompt = f"""
You're a loan officer who needs to make a QUICK decision without overthinking. Time pressure is forcing you to simplify.

Loan Application Data:
{loan_data}

YOUR PSYCHOLOGICAL STATE:
{fast_thinking_context[gse_level]}
{cognition_context[ncs_level]}

Your task: Rapidly simplify this loan approval into the most ESSENTIAL factors that would allow for an immediate gut-reaction decision.

Think like a human under time pressure - what would you instinctively focus on first?

Provide:
1. SIMPLIFIED TASK: One clear sentence describing what you need to decide
2. KEY FACTORS: The 2-3 most obvious indicators you'd check first
3. SIMPLIFICATION REASONING: Why you chose these factors over others

Be authentic to your psychological profile - show how your confidence level and thinking style affect your rapid decision-making."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        
        # Enhanced parsing with better structure recognition
        return self._parse_task_simplification(response_text)
    
    def make_fast_decision(self, task_simplification: TaskSimplification, loan_application: LoanApplication) -> CognitionState:
        """Make a quick decision with psychological authenticity"""
        
        # Get persona characteristics
        gse_level = self.persona_config.get_gse_level()
        ncs_level = self.persona_config.get_ncs_level()
        
        # Create fast decision context
        confidence_context = {
            ScaleLevel.HIGH: "You trust your instincts and feel confident making quick decisions.",
            ScaleLevel.MEDIUM: "You feel reasonably confident about quick decisions when the factors are clear.",
            ScaleLevel.LOW: "You feel a bit anxious about making quick decisions but recognize the need to move forward."
        }
        
        thinking_context = {
            ScaleLevel.HIGH: "You fight the urge to dig deeper and force yourself to stick with your gut reaction.",
            ScaleLevel.MEDIUM: "You feel comfortable making decisions with this level of analysis.",
            ScaleLevel.LOW: "You feel relieved to keep the analysis simple and straightforward."
        }
        
        prompt = f"""
You need to make a QUICK decision based on your simplified analysis. No overthinking allowed!

SIMPLIFIED TASK: {task_simplification.simplified_task}
KEY FACTORS: {', '.join(task_simplification.key_factors)}
REASONING: {task_simplification.simplification_reasoning}

YOUR PSYCHOLOGICAL STATE:
{confidence_context[gse_level]}
{thinking_context[ncs_level]}

Give your immediate gut reaction:

DECISION: [Approve/Reject]
CONFIDENCE: [X]%
QUICK REASONING: [One sentence explaining your gut reaction]

Be authentic to your psychological profile - show how your confidence and thinking style affect this rapid decision."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        
        # Enhanced extraction
        decision = self._extract_decision_from_text(response_text)
        confidence = self._extract_confidence(response_text)
        
        return CognitionState(
            initial_decision=decision,
            confidence_level=confidence,
            reasoning_process=[f"[FAST THINKING] {response_text}"],
            memory_items=task_simplification.key_factors,
            reflection_points=["Fast decision - minimal reflection due to time pressure"]
        )
    
    def _parse_task_simplification(self, response_text: str) -> TaskSimplification:
        """Enhanced parsing of task simplification response"""
        lines = response_text.split('\n')
        simplified_task = ""
        key_factors = []
        reasoning = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Enhanced section recognition
            if any(keyword in line.lower() for keyword in ["simplified task", "task:", "1."]):
                current_section = "task"
                # Extract task from same line if present
                if ":" in line:
                    simplified_task = line.split(":", 1)[1].strip()
            elif any(keyword in line.lower() for keyword in ["key factors", "factors:", "2."]):
                current_section = "factors"
            elif any(keyword in line.lower() for keyword in ["reasoning", "simplification", "3."]):
                current_section = "reasoning"
            elif current_section == "task" and not simplified_task:
                simplified_task = line
            elif current_section == "factors" and line.startswith(('-', '•', '1.', '2.', '3.', '*')):
                key_factors.append(line)
            elif current_section == "reasoning" and not reasoning:
                reasoning = line
        
        return TaskSimplification(
            simplified_task=simplified_task or "Quick loan approval based on key indicators",
            key_factors=key_factors[:3],  # Max 3 factors
            simplification_reasoning=reasoning or "Focused on most obvious risk indicators"
        )
    
    def _extract_decision_from_text(self, text: str) -> str:
        """Extract decision from text with enhanced recognition"""
        text_lower = text.lower()
        
        # Look for explicit decision keywords
        if any(word in text_lower for word in ["approve", "accept", "grant"]):
            return "Approve"
        elif any(word in text_lower for word in ["reject", "deny", "decline"]):
            return "Reject"
        else:
            return "Reject"  # Default to conservative decision
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence percentage from text"""
        import re
        matches = re.findall(r'(\d+)%', text)
        if matches:
            return float(matches[0])
        return 60.0  # Default confidence for fast thinking

class SlowThinkingAgent:
    """Agent that implements slow thinking (System 2) through multi-agent deliberation"""
    
    def __init__(self, llm: ChatOpenAI, persona_config):
        self.llm = llm
        self.persona_config = persona_config
        
    def engage_slow_thinking(self, loan_application: LoanApplication) -> SlowThinkingResult:
        """Engage in slow, deliberate thinking using multiple cognitive processes"""
        
        loan_data = json.dumps(loan_application.to_dict(), indent=2)
        
        # Step 1: Memory Analysis
        memory_analysis = self._memory_analysis(loan_data)
        
        # Step 2: Reflection
        reflection_analysis = self._reflection_analysis(loan_data, memory_analysis)
        
        # Step 3: Reasoning (Chain of Thought)
        reasoning_analysis = self._reasoning_analysis(loan_data, memory_analysis, reflection_analysis)
        
        # Step 4: Planning
        planning_analysis = self._planning_analysis(loan_data, reasoning_analysis)
        
        # Step 5: Theory of Mind (understanding other perspectives)
        theory_of_mind_analysis = self._theory_of_mind_analysis(loan_data, reasoning_analysis)
        
        # Step 6: Final Synthesis
        final_synthesis = self._final_synthesis(loan_data, memory_analysis, reflection_analysis, 
                                               reasoning_analysis, planning_analysis, theory_of_mind_analysis)
        
        return SlowThinkingResult(
            memory_analysis=memory_analysis,
            reflection_analysis=reflection_analysis,
            reasoning_analysis=reasoning_analysis,
            planning_analysis=planning_analysis,
            theory_of_mind_analysis=theory_of_mind_analysis,
            final_synthesis=final_synthesis
        )
    
    def _memory_analysis(self, loan_data: str) -> str:
        """Analyze based on past experience and knowledge"""
        prompt = f"""
        As a loan officer, recall your past experiences with similar loan applications.
        
        Loan Application:
        {loan_data}
        
        Based on your experience and knowledge, what patterns do you remember that are relevant to this case?
        What similar cases have you seen before?
        What outcomes did they have?
        
        Provide a detailed memory-based analysis.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _reflection_analysis(self, loan_data: str, memory_analysis: str) -> str:
        """Reflect on the analysis and potential biases"""
        # Note: loan_data parameter kept for consistency but reflection focuses on memory_analysis
        prompt = f"""
        Now reflect on your initial memory-based analysis.
        
        Your memory analysis: {memory_analysis}
        
        Questions to consider:
        1. What assumptions am I making?
        2. What biases might be influencing my judgment?
        3. What additional information would be helpful?
        4. What are the potential consequences of being wrong?
        
        Provide a thoughtful reflection that challenges your initial analysis.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _reasoning_analysis(self, loan_data: str, memory_analysis: str, reflection_analysis: str) -> str:
        """Deep chain-of-thought reasoning"""
        prompt = f"""
        Now engage in step-by-step logical reasoning about this loan application.
        
        Loan Data: {loan_data}
        Memory Analysis: {memory_analysis}
        Reflection: {reflection_analysis}
        
        Apply chain-of-thought reasoning:
        1. What are the key risk factors?
        2. How do these factors interact with each other?
        3. What is the probability of default based on each factor?
        4. How do these factors combine to create overall risk?
        5. What evidence supports approval vs rejection?
        
        Work through this systematically, step by step.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _planning_analysis(self, loan_data: str, reasoning_analysis: str) -> str:
        """Plan the decision-making approach"""
        # Note: loan_data parameter kept for consistency but planning focuses on reasoning_analysis
        prompt = f"""
        Based on your reasoning, plan your decision-making approach.
        
        Reasoning Analysis: {reasoning_analysis}
        
        Consider:
        1. What decision criteria should I prioritize?
        2. What additional checks or conditions might be needed?
        3. How should I weigh different factors?
        4. What would be my decision-making framework?
        
        Create a structured plan for making this decision.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _theory_of_mind_analysis(self, loan_data: str, reasoning_analysis: str) -> str:
        """Consider other perspectives and stakeholder viewpoints"""
        # Note: loan_data parameter kept for consistency but analysis focuses on reasoning_analysis
        prompt = f"""
        Now consider this loan application from multiple perspectives.
        
        Reasoning: {reasoning_analysis}
        
        Consider the viewpoints of:
        1. The loan applicant - what might they be thinking/feeling?
        2. The bank's management - what are their concerns?
        3. Regulatory authorities - what compliance issues might arise?
        4. Other loan officers - how might they view this case?
        
        How do these different perspectives influence the decision?
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _final_synthesis(self, loan_data: str, memory_analysis: str, reflection_analysis: str, 
                        reasoning_analysis: str, planning_analysis: str, theory_of_mind_analysis: str) -> str:
        """Synthesize all analyses into a final decision"""
        # Note: loan_data parameter kept for consistency but synthesis focuses on combining all analyses
        prompt = f"""
        Now synthesize all your analyses into a comprehensive decision.
        
        Memory Analysis: {memory_analysis}
        Reflection: {reflection_analysis}
        Reasoning: {reasoning_analysis}
        Planning: {planning_analysis}
        Theory of Mind: {theory_of_mind_analysis}
        
        Based on all this deep thinking, provide:
        1. Your final decision (Approve/Reject)
        2. Your confidence level (0-100%)
        3. A comprehensive reasoning that integrates all your analyses
        4. Key factors that influenced your decision
        5. Any remaining concerns or uncertainties
        
        This should be a thorough, well-reasoned decision.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def make_slow_decision(self, slow_thinking_result: SlowThinkingResult) -> CognitionState:
        """Convert slow thinking result to CognitionState"""
        
        final_text = slow_thinking_result.final_synthesis
        
        # Extract decision and confidence
        decision = "Approve" if "approve" in final_text.lower() else "Reject"
        confidence = self._extract_confidence(final_text)
        
        return CognitionState(
            initial_decision=decision,
            confidence_level=confidence,
            reasoning_process=[
                slow_thinking_result.memory_analysis,
                slow_thinking_result.reflection_analysis,
                slow_thinking_result.reasoning_analysis,
                slow_thinking_result.planning_analysis,
                slow_thinking_result.theory_of_mind_analysis,
                slow_thinking_result.final_synthesis
            ],
            memory_items=self._extract_memory_items(slow_thinking_result.memory_analysis),
            reflection_points=self._extract_reflection_points(slow_thinking_result.reflection_analysis)
        )
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence percentage from text"""
        import re
        matches = re.findall(r'(\d+)%', text)
        if matches:
            return float(matches[0])
        return 75.0  # Default confidence for slow thinking
    
    def _extract_memory_items(self, memory_text: str) -> List[str]:
        """Extract key memory items from memory analysis"""
        lines = memory_text.split('\n')
        memory_items = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:  # Substantial content
                memory_items.append(line[:100] + "..." if len(line) > 100 else line)
        return memory_items[:5]  # Top 5 memory items
    
    def _extract_reflection_points(self, reflection_text: str) -> List[str]:
        """Extract key reflection points from reflection analysis"""
        lines = reflection_text.split('\n')
        reflection_points = []
        for line in lines:
            line = line.strip()
            if line and ('?' in line or 'consider' in line.lower() or 'might' in line.lower()):
                reflection_points.append(line[:100] + "..." if len(line) > 100 else line)
        return reflection_points[:5]  # Top 5 reflection points

class ThinkingModeController:
    """Controller that decides between fast and slow thinking based on persona characteristics"""
    
    def __init__(self, llm: ChatOpenAI, persona_config):
        self.llm = llm
        self.persona_config = persona_config
        self.fast_thinking_agent = FastThinkingAgent(llm, persona_config)
        self.slow_thinking_agent = SlowThinkingAgent(llm, persona_config)
    
    def determine_thinking_mode(self, loan_application: LoanApplication) -> ThinkingMode:
        """Determine whether to use fast or slow thinking based on persona characteristics"""
        
        # Use the new psychological scale system instead of old simple numeric score
        ncs_level = self.persona_config.get_ncs_level()
        
        # Determine thinking mode based on Need for Cognition level
        if ncs_level.value == "high":
            return ThinkingMode.SLOW  # High NFC prefers complex thinking
        elif ncs_level.value == "low":
            return ThinkingMode.FAST  # Low NFC prefers simple thinking
        else:  # medium
            # For medium NFC, use the thinking mode preference set during persona creation
            if hasattr(self.persona_config, 'thinking_mode_preference'):
                if self.persona_config.thinking_mode_preference == "slow":
                    return ThinkingMode.SLOW
                elif self.persona_config.thinking_mode_preference == "fast":
                    return ThinkingMode.FAST
                else:  # adaptive
                    # For adaptive mode, decide based on loan complexity
                    # Simple heuristic: high loan amounts or low CIBIL scores require slow thinking
                    if (loan_application.loan_amount > 20000000 or 
                        loan_application.cibil_score < 600):
                        return ThinkingMode.SLOW
                    else:
                        return ThinkingMode.FAST
            else:
                # Fallback to adaptive behavior for medium NFC
                return ThinkingMode.FAST if loan_application.cibil_score > 700 else ThinkingMode.SLOW
    
    def make_decision(self, loan_application: LoanApplication) -> CognitionState:
        """Make a decision using the appropriate thinking mode"""
        
        thinking_mode = self.determine_thinking_mode(loan_application)
        
        # Note: start_time for timing would be used if we needed to track thinking mode performance
        # start_time = time.time()
        
        if thinking_mode == ThinkingMode.FAST:
            # Fast thinking process
            task_simplification = self.fast_thinking_agent.simplify_task(loan_application)
            cognition_state = self.fast_thinking_agent.make_fast_decision(task_simplification, loan_application)
            
            # Add thinking mode info
            cognition_state.reasoning_process.insert(0, f"[FAST THINKING] {task_simplification.simplified_task}")
            
        else:
            # Slow thinking process
            slow_thinking_result = self.slow_thinking_agent.engage_slow_thinking(loan_application)
            cognition_state = self.slow_thinking_agent.make_slow_decision(slow_thinking_result)
            
            # Add thinking mode info
            cognition_state.reasoning_process.insert(0, "[SLOW THINKING] Deep multi-agent analysis")
        
        # Note: processing_time is calculated but not used - could be added to CognitionState if needed
        # processing_time = time.time() - start_time
        
        return cognition_state