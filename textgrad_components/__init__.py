"""
TextGrad Components for MiroFlow
=================================
Integrates TextGrad optimization into MiroFlow multi-agent system.
"""

from .textgrad_orchestrator import TextGradOrchestrator
from .prompt_wrapper import PromptVariableManager
from .miroflow_loss import MiroFlowTextualFeedbackLoss
from .gradient_handler import agent_specific_aggregate
from .my_openai import ChatOpenAI
__all__ = [
    "TextGradOrchestrator",
    "PromptVariableManager",
    "MiroFlowTextualFeedbackLoss",
    "agent_specific_aggregate",
    "ChatOpenAI"
]
