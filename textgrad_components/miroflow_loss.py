"""
MiroFlow TextGrad Loss Module
==============================
Evaluates final answers and generates agent-specific feedback for MiroFlow.

Adapted from Over-TextGrad's TextualFeedbackLoss for MiroFlow's multi-agent structure.
"""

import re
from typing import Dict, Any, Optional

import textgrad as tg
from textgrad.autograd.string_based_ops import BackwardContext, StringBasedFunction
from textgrad.loss import MultiFieldEvaluation
from textgrad.engine import EngineLM

from src.logging.logger import bootstrap_logger

logger = bootstrap_logger()


# Evaluation prompt adapted for MiroFlow
MIROFLOW_EVAL_PROMPT = """You are an evaluator that produces "textual gradients" to improve agent performance in a multi-agent system.

You will receive:
- **Problem**: The original question or task
- **Gold Answer**: The correct answer
- **Model Output**: Responses from multiple agents in the system
  - Main Agent: Orchestrates the task and produces final answer
  - Sub Agents (if any): Perform specialized subtasks (e.g., web search, code execution)

Your job is provide structured feedback:
   
   a) **Final Answer Deviation**: Briefly state how the final answer differs from the gold answer.
      - Is it numerical error? Unit mismatch? Formatting issue? Logical error?
   
   b) **Agent-by-Agent Analysis**:
      - **Main Agent**: Did it orchestrate the task well? Did it select the right tools/sub-agents? Did it integrate information correctly?
      - **Sub Agents** (if any): Did they complete their subtasks correctly? Did they provide accurate information?
      
   c) **Fix Guidance** (HIGH-LEVEL only):
      - Provide general, actionable guidance for each agent
      - Focus on systematic improvements, NOT case-specific details
      - Examples: "Improve numerical calculation accuracy", "Better interpret tool outputs", "More thorough web search strategy"

**IMPORTANT**:
- If an agent performed correctly, return **"Correct Reasoning"** for that agent
- Be specific but concise
- Focus on high-level patterns, not example-specific fixes
- If a sub-agent wasn't used, you can skip it or mark as "Not Used"

**Output Format** (when there is deviation):
```
<deviation_summary>
Brief summary of the final answer error
</deviation_summary>

<agent-by-agent>
    <main_agent>
        Feedback for main agent, or "Correct Reasoning"
    </main_agent>
    <agent-worker>
        Feedback for sub-agent if used, or "Correct Reasoning" / "Not Used"
    </agent-worker>
</agent-by-agent>
```

Now perform the evaluation.
"""


class MiroFlowTextualFeedbackLoss(StringBasedFunction):
    """
    TextGrad loss module for MiroFlow multi-agent system.
    
    Evaluates the final answer and generates agent-specific feedback.
    """
    
    def __init__(
        self,
        engine: EngineLM,
        evaluation_prompt: str = MIROFLOW_EVAL_PROMPT,
        logger=None,
        memory_manager=None
    ):
        if engine is None:
            raise ValueError("An evaluation engine must be provided for MiroFlowTextualFeedbackLoss.")
        
        self.engine = engine
        self.logger = logger
        self.memory_manager = memory_manager  # Optional memory manager for feedback enhancement
        self.evaluation_instruction = tg.Variable(
            evaluation_prompt,
            requires_grad=False,
            role_description="loss evaluation instruction"
        )
        
        self.loss_fn = MultiFieldEvaluation(
            self.evaluation_instruction,
            ["Problem", "Model Output", "Gold Answer"],
            engine
        )
        
        super().__init__(fn=self.forward, function_purpose="textual feedback loss for MiroFlow")
    
    def forward(
        self,
        prediction_text: str,
        problem_text: str,
        gold_answer: str,
        agent_outputs: Dict[str, tg.Variable]
    ) -> tg.Variable:
        """
        Forward pass: evaluate prediction and generate feedback.
        
        Args:
            prediction_text: Final answer from main agent
            problem_text: Original problem/question
            gold_answer: Ground truth answer
            agent_outputs: Dict mapping agent names to their output Variables
        
        Returns:
            Loss variable containing structured feedback
        """
        
        # Build model output that includes all agent responses
        model_output_parts = []
        
        # Main agent
        if "main_agent" in agent_outputs:
            main_output = agent_outputs["main_agent"].get_value()
            model_output_parts.append(f"**Main Agent Response**:\n{main_output}\n")
        
        # Sub agents
        for agent_name, agent_var in agent_outputs.items():
            if agent_name != "main_agent":
                sub_output = agent_var.get_value()
                model_output_parts.append(f"**{agent_name} Response**:\n{sub_output}\n")
        
        # Add final answer
        model_output_parts.append(f"\n**Final Answer**: {prediction_text}")
        
        model_output_text = "\n".join(model_output_parts)
        
        # Create combined variable with all agent outputs as predecessors
        predecessors = list(agent_outputs.values())
        
        model_output_var = tg.Variable(
            model_output_text,
            requires_grad=True,
            role_description="Model Output",
            predecessors=predecessors
        )
        
        # Set custom backward to route feedback to agents
        def custom_backward(backward_engine: EngineLM):
            """
            自定义反向传播函数：解析反馈并将梯度信息路由到对应的代理变量
            
            该函数负责：
            1. 从模型输出变量中获取梯度文本（反馈信息）
            2. 解析结构化的代理反馈，提取针对各个代理的特定反馈
            3. 将反馈路由到相应的代理变量，以便后续的梯度更新
            
            Args:
                backward_engine: 反向传播引擎实例，用于反向传播计算
                
            处理逻辑：
            - 如果反馈文本为空，则直接返回
            - 使用_parse_agent_feedback方法解析出代理名称到反馈内容的映射
            - 对于每个代理，如果其名称出现在反馈中且不是"Correct Reasoning"或"Not Used"，
              则创建梯度变量并添加到对应代理的梯度集合中
            """
            feedback_text = model_output_var.get_gradient_text()
            
            if not feedback_text:
                return
            
            # Parse agent-specific feedback
            agent_feedback = self._parse_agent_feedback(feedback_text)
            
            # Route feedback to each agent
            for agent_name, agent_var in agent_outputs.items():
                if agent_name in agent_feedback:
                    feedback_content = agent_feedback[agent_name]
                    
                    # Skip if "Correct Reasoning" or "Not Used"
                    if feedback_content.strip().lower() in ["correct reasoning", "not used"]:
                        continue
                    
                    # Create gradient variable
                    gradient_var = tg.Variable(
                        value=feedback_content,
                        role_description=f"feedback to {agent_name}"
                    )
                    agent_var.gradients.add(gradient_var)
                    
                    
                    print(f"Routed feedback to {agent_name}: {feedback_content[:200]}...")
        
        model_output_var.set_grad_fn(custom_backward)
        
        # Create question and gold answer variables
        question_var = tg.Variable(
            problem_text,
            requires_grad=False,
            role_description="Problem"
        )
        
        gold_var = tg.Variable(
            gold_answer,
            requires_grad=False,
            role_description="Gold Answer"
        )
        
        # Log inputs
        
        print(f"Loss forward - Problem: {problem_text[:100]}...")
        print(f"Loss forward - Predicted: {prediction_text}")
        print(f"Loss forward - Gold: {gold_answer}")
        
        # Generate feedback
        feedback_variable = self.loss_fn([question_var, model_output_var, gold_var])
        
        # Enhance feedback using memory (if available and enabled)
        if self.memory_manager:
            try:
                current_feedback = feedback_variable.get_value()
                
                # Parse agent-specific feedback
                parsed = self._parse_agent_feedback(current_feedback)
                
                # Enhance each agent's feedback using memory
                enhanced_parts = []
                enhanced_parts.append("<agent-by-agent>")
                
                for agent_name, agent_fb in parsed.items():
                    # Check if agent has correct reasoning (case-insensitive)
                    is_correct = agent_fb and agent_fb.strip().lower() == "correct reasoning"
                    
                    if agent_fb and agent_fb.strip() and not is_correct:
                        # Enhance using memory manager
                        enhanced_fb = self.memory_manager.enhance_feedback(
                            current_feedback=agent_fb,
                            agent_name=agent_name,
                            llm_engine=self.engine
                        )
                        enhanced_parts.append(f"    <{agent_name}>\n{enhanced_fb}\n    </{agent_name}>")
                    elif agent_fb:
                        # Keep correct reasoning as is
                        enhanced_parts.append(f"    <{agent_name}>\n{agent_fb}\n    </{agent_name}>")
                
                enhanced_parts.append("</agent-by-agent>")
                
                if len(enhanced_parts) > 2:  # Has content beyond opening and closing tags
                    # Reconstruct enhanced feedback
                    enhanced_feedback = "\n".join(enhanced_parts)
                    
                    # Preserve deviation summary if present
                    deviation_match = re.search(
                        r'<deviation_summary>(.*?)</deviation_summary>',
                        current_feedback,
                        re.DOTALL | re.IGNORECASE
                    )
                    if deviation_match:
                        enhanced_feedback = f"<deviation_summary>\n{deviation_match.group(1)}\n</deviation_summary>\n\n{enhanced_feedback}"
                    
                    feedback_variable.set_value(enhanced_feedback)
                    
                    print("=================== Memory-Enhanced Feedback ===================")
                    print(f"Enhanced feedback with memory patterns: {enhanced_feedback[:500]}...")
                    print("================================================================")
                    
            except Exception as e:
                # If enhancement fails, continue with original feedback
                logger.warning(f"Memory enhancement failed: {e}")
                print(f"Warning: Memory enhancement failed, using original feedback: {e}")
        
        print("=================== Loss feedback begin ===================================")
        print(f"Loss feedback: {feedback_variable.get_value()[:500]}...")
        print("=================== Loss feedback end =====================================")

        return feedback_variable
    
    def _parse_agent_feedback(self, feedback_text: str) -> Dict[str, str]:
        """
        Parse structured feedback to extract agent-specific sections.
        
        Expected format:
        <agent-by-agent>
            <main_agent>...</main_agent>
            <agent-worker>...</agent-worker>
            ...
        </agent-by-agent>
        
        Returns:
            Dict mapping agent_name -> feedback_content
        """
        agent_feedback = {}
        
        # Find agent-by-agent section
        agent_section_match = re.search(
            r'<agent-by-agent>(.*?)</agent-by-agent>',
            feedback_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if not agent_section_match:
            logger.warning("No structured agent-by-agent feedback found")
            return agent_feedback
        
        agent_section = agent_section_match.group(1)
        
        # Extract individual agent feedback
        # Common agent names in MiroFlow
        agent_names = ["main_agent", "agent-worker", "agent-browser", "agent-coder"]
        
        for agent_name in agent_names:
            pattern = rf'<({agent_name}[^>]*)>(.*?)</\1>'
            match = re.search(pattern, agent_section, re.DOTALL | re.IGNORECASE)
            if match:
                feedback_content = match.group(2).strip()
                # Store if not empty and not just "Correct Reasoning"
                if feedback_content and feedback_content.lower() != "correct reasoning":
                    agent_feedback[agent_name] = feedback_content
        
        print(f"Parsed feedback for {len(agent_feedback)} agents")
        return agent_feedback


def __call__(
    self,
    prediction_text: str,
    problem_text: str,
    gold_answer: str,
    agent_outputs: Dict[str, tg.Variable]
) -> tg.Variable:
    """Alias for forward()"""
    return self.forward(prediction_text, problem_text, gold_answer, agent_outputs)