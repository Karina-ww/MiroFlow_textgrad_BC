"""
Prompt Variable Manager
=======================
Manages trainable prompt variables for MiroFlow agents.

Wraps BaseAgentPrompt outputs as tg.Variables for TextGrad optimization.
"""

import os
from typing import Dict, List, Any
import importlib

import textgrad as tg
from omegaconf import DictConfig

from config.agent_prompts.base_agent_prompt import BaseAgentPrompt
from src.logging.logger import bootstrap_logger

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


class PromptVariableManager:
    """
    Manages prompt variables for main and sub agents.
    
    Responsibilities:
    1. Load prompt classes from config
    2. Generate initial prompts
    3. Wrap prompts as trainable tg.Variables
    4. Provide access to trainable parameters
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # Load prompt instances
        self.main_agent_prompt_instance = self._load_prompt_class(
            cfg.main_agent.prompt_class
        )
        
        self.sub_agent_prompt_instances: Dict[str, BaseAgentPrompt] = {}
        if cfg.sub_agents is not None and cfg.sub_agents:
            for agent_name, agent_cfg in cfg.sub_agents.items():
                prompt_instance = self._load_prompt_class(agent_cfg.prompt_class)
                self.sub_agent_prompt_instances[agent_name] = prompt_instance
        
        # Generate initial prompt texts (with dummy tool definitions for now)
        chinese_context = cfg.main_agent.get("chinese_context", "false").lower() == "true"
        
        # Main agent prompt
        main_prompt_text = self.main_agent_prompt_instance.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],  # Empty for initial template, will be filled at runtime
            chinese_context=chinese_context
        )
        
        # Create trainable variable for main agent
        self.main_agent_prompt_var = tg.Variable(
            value=main_prompt_text,
            role_description="main_agent system prompt",
            requires_grad=True
        )
        
        logger.info(f"Initialized main agent prompt ({len(main_prompt_text)} chars)")
        
        # Sub agent prompts
        self.sub_agent_prompt_vars: Dict[str, tg.Variable] = {}
        for agent_name, prompt_instance in self.sub_agent_prompt_instances.items():
            sub_prompt_text = prompt_instance.generate_system_prompt_with_mcp_tools(
                mcp_servers=[],
                chinese_context=chinese_context
            )
            
            sub_prompt_var = tg.Variable(
                value=sub_prompt_text,
                role_description=f"{agent_name} system prompt",
                requires_grad=True
            )
            
            self.sub_agent_prompt_vars[agent_name] = sub_prompt_var
            logger.info(f"Initialized {agent_name} prompt ({len(sub_prompt_text)} chars)")
    
    def _load_prompt_class(self, prompt_class_name: str) -> BaseAgentPrompt:
        """Load prompt class dynamically from config.agent_prompts module"""
        if not isinstance(prompt_class_name, str) or not prompt_class_name.isidentifier():
            raise ValueError(f"Invalid prompt class name: {prompt_class_name}")
        
        try:
            agent_prompts_module = importlib.import_module("config.agent_prompts")
            PromptClass = getattr(agent_prompts_module, prompt_class_name)
            return PromptClass()
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(
                f"Could not import class '{prompt_class_name}' from 'config.agent_prompts': {e}"
            )
    
    def get_main_agent_prompt_variable(self) -> tg.Variable:
        """Get the trainable main agent prompt variable"""
        return self.main_agent_prompt_var
    
    def get_sub_agent_prompt_variables(self) -> Dict[str, tg.Variable]:
        """Get all trainable sub-agent prompt variables"""
        return self.sub_agent_prompt_vars.copy()
    
    def trainable_parameters(self) -> List[tg.Variable]:
        """Get all trainable prompt variables as a list"""
        params = [self.main_agent_prompt_var]
        params.extend(self.sub_agent_prompt_vars.values())
        return params
    
    def get_current_prompt_text(self, agent_name: str) -> str:
        """Get current prompt text for a specific agent"""
        if agent_name == "main_agent":
            return self.main_agent_prompt_var.get_value()
        elif agent_name in self.sub_agent_prompt_vars:
            return self.sub_agent_prompt_vars[agent_name].get_value()
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")
    
    def update_prompt(self, agent_name: str, new_prompt: str):
        """
        Update prompt text for a specific agent.
        This is called by TextGrad optimizer during backward pass.
        """
        if agent_name == "main_agent":
            self.main_agent_prompt_var.set_value(new_prompt)
            logger.info(f"Updated main agent prompt")
        elif agent_name in self.sub_agent_prompt_vars:
            self.sub_agent_prompt_vars[agent_name].set_value(new_prompt)
            logger.info(f"Updated {agent_name} prompt")
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")
