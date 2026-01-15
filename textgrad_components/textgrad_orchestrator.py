"""
TextGrad-wrapped Orchestrator
==============================
Wraps MiroFlow's Orchestrator to make it compatible with TextGrad optimization.

Key features:
- Wraps agent prompts as trainable tg.Variables
- Tracks agent outputs for gradient computation
- Extracts final turn responses (ignores multi-turn history)
- Returns structured outputs for loss computation
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path

import textgrad as tg
from omegaconf import DictConfig

from src.core.orchestrator import Orchestrator
from src.llm.client import LLMClient
from src.llm.provider_client_base import LLMProviderClientBase
from src.logging.logger import bootstrap_logger
from src.logging.task_tracer import TaskTracer
from src.tool.manager import ToolManager
from src.utils.io_utils import OutputFormatter

from .prompt_wrapper import PromptVariableManager

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


class TextGradOrchestrator:
    """
    TextGrad-wrapped version of MiroFlow's Orchestrator.
    
    This class wraps the original Orchestrator and adds TextGrad functionality:
    1. Manages trainable prompt variables
    2. Tracks agent outputs as tg.Variables
    3. Extracts only final turn responses for loss computation
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        main_agent_tool_manager: ToolManager,
        sub_agent_tool_managers: Dict[str, ToolManager],
        output_formatter: OutputFormatter,
        prompt_manager: PromptVariableManager,
    ):
        self.cfg = cfg
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.output_formatter = output_formatter
        self.prompt_manager = prompt_manager
        
        # Initialize LLM clients
        self.main_llm_client = None
        self.sub_llm_client = None
        
        # Initialize evaluator engine for TextGrad
        self.evaluator_engine = None
        self._init_evaluator_engine()
        
        # Storage for agent outputs (per task)
        self.current_agent_outputs: Dict[str, tg.Variable] = {}
        
    def _init_evaluator_engine(self):
        """Initialize evaluator LLM engine for TextGrad backward pass"""
        # Use the same LLM config as main agent for evaluation
        if hasattr(self.cfg.main_agent, "llm") and self.cfg.main_agent.llm is not None:
            from src.llm.providers.gpt5_openai_client import GPT5OpenAIClient
            from src.llm.providers.claude_openrouter_client import ClaudeOpenRouterClient
            
            provider_class = self.cfg.main_agent.llm.provider_class
            
            # Create evaluator client (use GPT-5 for optimizer)
            from .my_openai import ChatOpenAI
            self.evaluator_engine = ChatOpenAI(
                model_string="gpt-5",
            )
            
            logger.info(f"Initialized evaluator engine: {self.evaluator_engine}")
        else:
            raise ValueError("No LLM configuration found for evaluator engine")
    
    async def run_main_agent(
        self,
        task_description: str,
        task_file_name: Optional[str] = None,
        task_id: str = "default_task",
        is_training: bool = True,
        ground_truth: Optional[str] = None,
        epoch: int = 0,
        token_counter = None,
        log_subdir: str = "task_logs"
    ) -> Dict[str, Any]:
        """
        Run the main agent with TextGrad tracking.
        
        Args:
            task_description: Task question
            task_file_name: Optional file path
            task_id: Unique task identifier
            is_training: Whether in training mode
            ground_truth: Ground truth answer (optional)
            token_counter: Optional TokenCounter instance for tracking token usage
            log_subdir: Subdirectory for task logs (default: "task_logs")
        
        Returns:
            Dict with:
                - final_answer: Full text response
                - final_boxed_answer: Extracted answer
                - agent_outputs: Dict[agent_name, tg.Variable] with final responses
                - task_log_path: Path to task log
        """
        logger.info(f"\n{'='*60}\nRunning task: {task_id}\n{'='*60}")
        
        # Reset agent outputs for this task
        self.current_agent_outputs = {}
        
        # Create task log with epoch in filename, using custom log_subdir
        if epoch >= 0:
            log_path = Path(self.cfg.output_dir) / log_subdir / f"task_{task_id}_epoch{epoch}.json"
        else:
            # For evaluation (epoch=-1), use "eval" prefix
            log_path = Path(self.cfg.output_dir) / log_subdir / f"task_{task_id}_eval.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        task_log = TaskTracer(
            log_path=log_path,
            task_name=f"textgrad_{task_id}",
            task_id=task_id,
            task_file_name=task_file_name,
            ground_truth=ground_truth,
            epoch=epoch,  # Store epoch in TaskTracer
            input={"task_description": task_description, "task_file_name": task_file_name}
        )
        
        # Initialize LLM clients for this task
        self.main_llm_client = LLMClient(
            task_id=task_id,
            llm_config=self.cfg.main_agent.llm
        )
        
        # Attach token_counter to main LLM client if provided
        if token_counter:
            self.main_llm_client.token_counter = token_counter
        
        if self.cfg.sub_agents is not None and self.cfg.sub_agents:
            first_sub_agent = next(iter(self.cfg.sub_agents.values()))
            if hasattr(first_sub_agent, "llm") and first_sub_agent.llm is not None:
                self.sub_llm_client = LLMClient(
                    task_id=f"{task_id}_sub",
                    llm_config=first_sub_agent.llm
                )
                # Attach token_counter to sub LLM client if provided
                if token_counter:
                    self.sub_llm_client.token_counter = token_counter
        
        # Get current prompts from prompt manager (as strings for now, will wrap later)
        main_prompt_var = self.prompt_manager.get_main_agent_prompt_variable()
        sub_prompt_vars = self.prompt_manager.get_sub_agent_prompt_variables()
        
        # Create standard Orchestrator with current prompt values
        # IMPORTANT: Pass prompt_manager so Orchestrator can use optimized prompts
        orchestrator = Orchestrator(
            main_agent_tool_manager=self.main_agent_tool_manager,
            sub_agent_tool_managers=self.sub_agent_tool_managers,
            llm_client=self.main_llm_client,
            sub_agent_llm_client=self.sub_llm_client,
            output_formatter=self.output_formatter,
            cfg=self.cfg,
            task_log=task_log,
            prompt_manager=self.prompt_manager  # Pass PromptVariableManager for optimized prompts
        )
        
        try:
            # Run the orchestrator
            final_answer, final_boxed_answer = await orchestrator.run_main_agent(
                task_description=task_description,
                task_file_name=task_file_name,
                task_id=task_id
            )
            
            # Extract agent outputs from task log
            # We only care about the FINAL turn responses, not multi-turn history
            self._extract_agent_outputs_from_log(task_log, is_training)
            
            # IMPORTANT: Save final_boxed_answer to task_log so it gets persisted
            task_log.final_boxed_answer = final_boxed_answer
            task_log.status = "completed"
            
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}", exc_info=True)
            final_answer = f"Error: {e}"
            final_boxed_answer = ""
            task_log.final_boxed_answer = final_boxed_answer
            task_log.status = "failed"
            task_log.error = str(e)
        
        finally:
            task_log.save()
            if self.main_llm_client:
                self.main_llm_client.close()
            if self.sub_llm_client and self.sub_llm_client != self.main_llm_client:
                self.sub_llm_client.close()
        
        return {
            "final_answer": final_answer,
            "final_boxed_answer": final_boxed_answer,
            "agent_outputs": self.current_agent_outputs.copy(),
            "task_log_path": log_path
        }
    
    def _extract_agent_outputs_from_log(self, task_log: TaskTracer, is_training: bool):
        """
        Extract final turn responses from task log and wrap as tg.Variables.
        
        For MiroFlow's multi-turn dialogue:
        - Main agent: extract final summary response
        - Sub agents: extract their final summary responses
        
        We ignore intermediate tool calls and only keep the final "answer" turn.
        """
        # Main agent final response - use main_agent_message_history
        if task_log.main_agent_message_history:
            # Get the last assistant message from history
            main_final_response = None
            message_history = task_log.main_agent_message_history.get("message_history", [])
            for message in reversed(message_history):
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        main_final_response = content
                    elif isinstance(content, list) and len(content) > 0:
                        # Extract text from content list
                        for item in content:
                            if item.get("type") == "text":
                                main_final_response = item.get("text", "")
                                break
                    if main_final_response:
                        break
            
            if main_final_response:
                # Link to prompt variable as predecessor only in training mode
                main_prompt_var = self.prompt_manager.get_main_agent_prompt_variable()
                
                # During evaluation (is_training=False), don't link to trainable predecessors
                # to avoid "variable does not require grad but predecessor does" error
                predecessors = [main_prompt_var] if is_training else []
                
                main_var = tg.Variable(
                    value=main_final_response,
                    role_description="main_agent final response",
                    predecessors=predecessors,
                    requires_grad=is_training
                )
                
                self.current_agent_outputs["main_agent"] = main_var
                logger.debug(f"Extracted main agent final response: {main_final_response[:200]}...")
        
        # Sub agent final responses - use sub_agent_message_history_sessions
        if task_log.sub_agent_message_history_sessions:
            for session_id, session_data in task_log.sub_agent_message_history_sessions.items():
                # Extract agent name from session_id (format: agent_name_number)
                agent_name = "_".join(session_id.split("_")[:-1]) if "_" in session_id else "unknown_sub_agent"
                
                # Find last assistant response in this sub-agent session
                sub_final_response = None
                message_history = session_data.get("message_history", [])
                for message in reversed(message_history):
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        if isinstance(content, str):
                            sub_final_response = content
                        elif isinstance(content, list) and len(content) > 0:
                            for item in content:
                                if item.get("type") == "text":
                                    sub_final_response = item.get("text", "")
                                    break
                        if sub_final_response:
                            break
                
                if sub_final_response:
                    # Link to sub-agent prompt variable as predecessor only in training mode
                    sub_prompt_vars = self.prompt_manager.get_sub_agent_prompt_variables()
                    
                    # During evaluation (is_training=False), don't link to trainable predecessors
                    if is_training and agent_name in sub_prompt_vars:
                        predecessors = [sub_prompt_vars[agent_name]]
                    else:
                        predecessors = []
                    
                    sub_var = tg.Variable(
                        value=sub_final_response,
                        role_description=f"{agent_name} final response",
                        predecessors=predecessors,
                        requires_grad=is_training
                    )
                    
                    self.current_agent_outputs[agent_name] = sub_var
                    logger.debug(f"Extracted {agent_name} final response: {sub_final_response[:200]}...")
        
        logger.info(f"Extracted {len(self.current_agent_outputs)} agent final responses")
    
    def get_agent_output_variables(self) -> Dict[str, tg.Variable]:
        """Get the current agent output variables for loss computation"""
        return self.current_agent_outputs.copy()
    
    def trainable_parameters(self):
        """Get all trainable prompt variables"""
        return self.prompt_manager.trainable_parameters()
