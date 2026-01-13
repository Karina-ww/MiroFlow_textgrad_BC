"""
Gradient Handler for MiroFlow TextGrad
=======================================
Agent-specific feedback aggregation and routing.

Adapted from Over-TextGrad's agent_specific_aggregation.py for MiroFlow.
"""

import re
from typing import List, Dict
from collections import defaultdict

import textgrad as tg
from textgrad import logger
from textgrad.variable import Variable
from textgrad.engine import EngineLM
from textgrad.autograd.function import Function, BackwardContext


AGENT_FEEDBACK_SUMMARY_PROMPT = """You are tasked with summarizing multiple feedback items for a specific agent to identify high-level, recurring error patterns.

You will receive multiple feedback items from different problem instances. Your goal is to:
1. Identify GENERAL, HIGH-LEVEL error patterns (NOT case-specific details)
2. Categorize errors by type (e.g., task decomposition, tool selection, information integration, etc.)
3. Extract actionable improvement guidelines that apply broadly

Focus on:
- Recurring themes across multiple instances
- Systematic weaknesses in the agent's approach
- General improvements to the agent's problem-solving strategy

Avoid:
- Case-specific details or example-specific fixes
- Mentioning specific numbers, problem contexts, or individual examples
- Lengthy descriptions (be concise)

Output a concise summary (max 300 words) with:
- High-level error categories
- General improvement directions
- Systematic patterns observed

Feedback items:
{feedback_items}

Provide the high-level summary:"""


def parse_agent_feedback(feedback_text: str) -> Dict[str, str]:
    """
    Parse structured feedback to extract agent-specific sections.
    
    Expected format in feedback_text:
    <agent-by-agent>
        <main_agent>...</main_agent>
        <agent-worker>...</agent-worker>
        ...
    </agent-by-agent>
    
    Returns a dict mapping agent_name -> feedback_content
    """
    agent_feedback = {}
    
    # Try to find agent-by-agent section
    agent_section_match = re.search(
        r'<agent-by-agent>(.*?)</agent-by-agent>',
        feedback_text,
        re.DOTALL | re.IGNORECASE
    )
    
    if not agent_section_match:
        logger.info("No structured agent-by-agent feedback found", extra={"feedback": feedback_text[:200]})
        return agent_feedback
    
    agent_section = agent_section_match.group(1)
    
    # Extract individual agent feedback
    # MiroFlow common agent names
    agent_names = [
        "main_agent",
        "sub_agent",  # Updated to match actual sub-agent name
        "agent-worker",  # Keep for backward compatibility
        # "agent-browser",
        # "agent-coder" # 好像没有这两个
    ]
    
    for agent_name in agent_names:
        # Match both exact names and variations
        pattern = rf'<({agent_name}[^>]*)>(.*?)</\1>'
        match = re.search(pattern, agent_section, re.DOTALL | re.IGNORECASE)
        if match:
            feedback_content = match.group(2).strip()
            # Skip if it's just "Correct Reasoning" or "Not Used" or empty
            if feedback_content and feedback_content.lower() not in ["correct reasoning", "not used"]:
                agent_feedback[agent_name] = feedback_content
    
    return agent_feedback


def summarize_agent_feedbacks(agent_name: str, feedback_list: List[str], backward_engine: EngineLM) -> str:
    """
    Summarize multiple feedback items for a specific agent.
    Focus on high-level patterns, not case-specific errors.
    """
    if not feedback_list:
        return ""
    
    # If only one feedback, still summarize to extract high-level patterns
    feedback_items_str = "\n\n---\n\n".join([f"Instance {i+1}:\n{fb}" for i, fb in enumerate(feedback_list)])
    
    prompt = AGENT_FEEDBACK_SUMMARY_PROMPT.format(feedback_items=feedback_items_str)
    
    try:
        summary = backward_engine(prompt)
        logger.info(f"Summarized feedback for {agent_name}", extra={"summary_length": len(summary)})
        return summary
    except Exception as e:
        logger.error(f"Failed to summarize feedback for {agent_name}: {e}")
        # Fallback: concatenate with deduplication hints
        return f"Multiple feedback instances for {agent_name}. Key issues:\n" + "\n".join(feedback_list[:3])


class AgentSpecificAggregate(Function):
    """
    Aggregates losses with agent-specific feedback clustering.
    
    In the forward pass: concatenates loss values (like standard aggregation)
    In the backward pass: 
        - Parses structured feedback from each loss
        - Groups feedback by agent
        - Summarizes per-agent feedback (high-level patterns)
        - Backpropagates only relevant summary to each agent's parameter
    """
    
    def forward(self, variables: List[Variable]) -> Variable:
        """
        Forward pass: concatenate loss feedback values.
        """
        concat_values = "\n\n=====\n\n".join([v.get_value() for v in variables])
        role_descriptions = set([v.get_role_description() for v in variables])
        role_descriptions = ", ".join(role_descriptions)
        
        aggregated_variable = Variable(
            value=concat_values,
            role_description=f"aggregated feedback from {len(variables)} instances",
            predecessors=variables,
            requires_grad=any([v.requires_grad for v in variables])
        )
        
        aggregated_variable.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                aggregated_variable=aggregated_variable
            )
        )
        
        return aggregated_variable
    
    def backward(self, aggregated_variable: Variable, backward_engine: EngineLM):
        """
        Backward pass with agent-specific feedback clustering.
        
        1. Collect all loss feedbacks
        2. Parse and group by agent
        3. Summarize per-agent 
        4. Send only relevant summary to each parameter
        """
        children_variables = aggregated_variable.predecessors
        
        # Step 1: Collect all feedback texts (these are the loss outputs)
        all_feedbacks = []
        for var in children_variables:
            feedback_text = var.get_value()
            all_feedbacks.append(feedback_text)
        
        logger.info(f"Aggregating {len(all_feedbacks)} loss feedbacks")
        
        # Step 2: Parse and group by agent
        agent_feedback_groups: Dict[str, List[str]] = defaultdict(list)
        
        for feedback_text in all_feedbacks:
            parsed = parse_agent_feedback(feedback_text)
            for agent_name, agent_fb in parsed.items():
                agent_feedback_groups[agent_name].append(agent_fb)
        
        # Log feedback counts
        for agent_name, fb_list in agent_feedback_groups.items():
            logger.info(f"{agent_name}: {len(fb_list)} feedback items")
        
        # Step 3: Summarize per-agent feedback
        agent_summaries: Dict[str, str] = {}
        for agent_name, feedback_list in agent_feedback_groups.items():
            if feedback_list:
                summary = summarize_agent_feedbacks(agent_name, feedback_list, backward_engine)
                agent_summaries[agent_name] = summary
        
        # Step 4: Backpropagate agent-specific summaries
        # We need to find which parameters belong to which agents
        # The parameters are the prompts that were used in the forward pass
        # We'll traverse the computation graph to find the prompt variables
        
        visited = set()
        prompt_variables = []
        
        def find_prompt_variables(var: Variable):
            if var in visited:
                return
            visited.add(var)
            
            # Check if this is a prompt variable (has "system prompt" in role description)
            role_desc = var.get_role_description().lower()
            if "system prompt" in role_desc or "prompt" in role_desc:
                prompt_variables.append(var)
            
            # Traverse predecessors
            if hasattr(var, 'predecessors') and var.predecessors:
                for pred in var.predecessors:
                    find_prompt_variables(pred)
        
        # Find all prompt variables in the computation graph
        for child_var in children_variables:
            find_prompt_variables(child_var)
        
        # Now backpropagate agent-specific summaries to corresponding prompts
        for prompt_var in prompt_variables:
            role_desc = prompt_var.get_role_description()
            
            # Determine which agent this prompt belongs to
            agent_name = None
            for name in ["main_agent", "agent-worker", "agent-browser", "agent-coder"]:
                if name in role_desc:
                    agent_name = name
                    break
            
            if agent_name and agent_name in agent_summaries:
                summary = agent_summaries[agent_name]
                if summary:
                    gradient_var = Variable(
                        value=f"High-level feedback summary for {agent_name}:\n{summary}",
                        role_description=f"gradient for {agent_name} prompt"
                    )
                    prompt_var.gradients.add(gradient_var)
                    logger.info(
                        f"Backpropagated to {agent_name}",
                        extra={"summary_length": len(summary)}
                    )


def agent_specific_aggregate(variables: List[Variable]) -> Variable:
    """
    Convenience function to create agent-specific aggregation.
    Use this instead of tg.sum() or tg.aggregate() for agent-specific feedback.
    """
    agg = AgentSpecificAggregate()
    return agg.forward(variables)
