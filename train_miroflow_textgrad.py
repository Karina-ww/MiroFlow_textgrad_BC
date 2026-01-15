#!/usr/bin/env python3
"""
MiroFlow TextGrad Training Script
==================================
Integrates Over-TextGrad optimization into MiroFlow multi-agent system.

Features:
- Train/test split on GAIA dataset
- Extract final turn responses only (ignore multi-turn history)
- Agent-specific feedback and gradient routing
- Memory-enhanced loss generation
"""

import argparse
import asyncio
import json
import os
import random
import re
import string
import warnings
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import textgrad as tg
from omegaconf import DictConfig, OmegaConf
import hydra

from src.logging.logger import bootstrap_logger, task_logging_context, init_logging_for_benchmark_evaluation
from src.core.pipeline import create_pipeline_components
from textgrad_components.textgrad_orchestrator import TextGradOrchestrator
from textgrad_components.prompt_wrapper import PromptVariableManager
from textgrad_components.miroflow_loss import MiroFlowTextualFeedbackLoss
from textgrad_components.gradient_handler import agent_specific_aggregate

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)

# Import memory manager from textgrad_components
try:
    from textgrad_components.memory_mechanisms import MemoryManager
    logger.info("Memory mechanisms imported successfully from textgrad_components")
except ImportError as e:
    MemoryManager = None
    logger.warning(f"Failed to import memory mechanisms: {e}")


class TokenCounter:
    """记录反向传播过程中的 token 使用情况"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.backward_prompt_tokens = 0
        self.backward_completion_tokens = 0
        self.backward_total_tokens = 0
        self.backward_calls = 0
        self.api_call_prompt_tokens = 0
        self.api_call_completion_tokens = 0
        self.api_call_calls = 0
    
    def record(self, prompt_tokens=0, completion_tokens=0, call_type="backward"):
        """
        记录 token 使用
        
        Args:
            prompt_tokens: 提示词 tokens 数量
            completion_tokens: 完成响应 tokens 数量
            call_type: 调用类型，"backward" (梯度计算) 或 "api_call" (API调用)
        """
        if call_type == "backward":
            self.backward_prompt_tokens += prompt_tokens
            self.backward_completion_tokens += completion_tokens
            self.backward_total_tokens = self.backward_prompt_tokens + self.backward_completion_tokens
            self.backward_calls += 1
        elif call_type == "api_call":
            self.api_call_prompt_tokens += prompt_tokens
            self.api_call_completion_tokens += completion_tokens
            self.api_call_calls += 1
    
    def get_summary(self):
        return {
            "backward": {
                "calls": self.backward_calls,
                "prompt_tokens": self.backward_prompt_tokens,
                "completion_tokens": self.backward_completion_tokens,
                "total_tokens": self.backward_total_tokens
            },
            "api_calls": {
                "calls": self.api_call_calls,
                "prompt_tokens": self.api_call_prompt_tokens,
                "completion_tokens": self.api_call_completion_tokens,
                "total_tokens": self.api_call_prompt_tokens + self.api_call_completion_tokens
            },
            "total_tokens": self.backward_total_tokens + self.api_call_prompt_tokens + self.api_call_completion_tokens
        }


class TimingTracker:
    """记录训练过程中的时间消耗"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.training_start = None
        self.training_end = None
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.forward_calls = 0
        self.backward_calls = 0
    
    def start_training(self):
        self.training_start = time.time()
    
    def end_training(self):
        self.training_end = time.time()
    
    def record_forward(self, duration):
        self.forward_time += duration
        self.forward_calls += 1
    
    def record_backward(self, duration):
        self.backward_time += duration
        self.backward_calls += 1
    
    def get_summary(self):
        total_training_time = (self.training_end - self.training_start) if (self.training_start and self.training_end) else 0
        return {
            "forward": {
                "calls": self.forward_calls,
                "total_time": self.forward_time,
                "avg_time": self.forward_time / self.forward_calls if self.forward_calls > 0 else 0
            },
            "backward": {
                "calls": self.backward_calls,
                "total_time": self.backward_time,
                "avg_time": self.backward_time / self.backward_calls if self.backward_calls > 0 else 0
            },
            "total_training_time": total_training_time
        }


def select_trajectory_by_strategy(
    failure_trajectories: List[Tuple[Any, Dict, Dict[str, Any]]], 
    strategy: str = "max_feedback_length",
    verbose: bool = True
) -> Tuple[Any, Dict]:
    """
    Select a failure trajectory based on different strategies.
    
    This function implements multiple strategies for selecting which failure case to use
    for gradient computation. The selection strategy can significantly impact optimization.
    
    Args:
        failure_trajectories: List of (loss, agent_outputs, metadata) tuples where metadata contains
                             'predicted', 'gold_answer', 'task_id', etc.
        strategy: Selection strategy. Options:
            - "random": Random selection (original behavior)
            - "max_feedback_length": Select the case with longest/most detailed feedback
            - "weighted_feedback": Weighted random sampling by feedback length
            - "diverse_errors": Select case with most diverse agent errors
        verbose: If True, print selection metrics
    
    Returns:
        Tuple of (selected_loss, selected_agent_outputs) without metadata
    """
    if not failure_trajectories:
        raise ValueError("failure_trajectories cannot be empty")
    
    if strategy == "random":
        loss, outputs, metadata = random.choice(failure_trajectories)
        if verbose:
            print(f"  → Random selection: task={metadata.get('task_id')}, "
                       f"predicted={metadata.get('predicted')[:50]}..., gold={metadata.get('gold_answer')[:50]}...")
        return loss, outputs
    
    elif strategy == "max_feedback_length":
        # Select the trajectory with the longest/most detailed feedback
        max_length = -1
        selected = None
        selected_metadata = None
        
        for loss, outputs, metadata in failure_trajectories:
            feedback = loss.get_value() if hasattr(loss, 'get_value') else str(loss.value)
            feedback_length = len(feedback)
            
            if feedback_length > max_length:
                max_length = feedback_length
                selected = (loss, outputs)
                selected_metadata = metadata
        
        if verbose and selected_metadata:
            print(f"  → Max feedback length selection: length={max_length} chars, "
                       f"task={selected_metadata.get('task_id')}, "
                       f"predicted={selected_metadata.get('predicted')[:50]}...")
        
        return selected if selected else (failure_trajectories[0][0], failure_trajectories[0][1])
    
    elif strategy == "weighted_feedback":
        # Weighted random sampling by feedback length
        weights = []
        
        for loss, outputs, metadata in failure_trajectories:
            try:
                feedback = loss.get_value() if hasattr(loss, 'get_value') else str(loss.value)
                weight = len(feedback)
                weights.append(weight)
            except:
                weights.append(1)  # Default weight
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1] * len(failure_trajectories)
            total_weight = len(failure_trajectories)
        
        probabilities = [w / total_weight for w in weights]
        
        # Random selection with weights
        selected_idx = random.choices(range(len(failure_trajectories)), weights=probabilities, k=1)[0]
        loss, outputs, metadata = failure_trajectories[selected_idx]
        
        if verbose:
            print(f"  → Weighted selection: weight={weights[selected_idx]}, "
                       f"task={metadata.get('task_id')}, "
                       f"predicted={metadata.get('predicted')[:50]}...")
        
        return loss, outputs
    
    elif strategy == "diverse_errors":
        # Select the case with the most diverse agent errors
        max_agents_with_errors = -1
        selected = None
        selected_metadata = None
        selected_agent_list = []
        
        for loss, outputs, metadata in failure_trajectories:
            feedback = loss.get_value() if hasattr(loss, 'get_value') else str(loss.value)
            
            # Count how many different agents have errors
            agents_with_errors = 0
            error_agents = []
            # Check for common agent names in MiroFlow
            for agent_keyword in ["main_agent", "sub_agent", "search", "browse", "code", "calculator"]:
                # Check if this agent is mentioned with error/issue keywords
                if agent_keyword in feedback.lower() and any(keyword in feedback.lower() for keyword in 
                                                  ["error", "incorrect", "wrong", "mistake", "issue", "problem", "failed"]):
                    agents_with_errors += 1
                    error_agents.append(agent_keyword)
            
            if agents_with_errors > max_agents_with_errors:
                max_agents_with_errors = agents_with_errors
                selected = (loss, outputs)
                selected_metadata = metadata
                selected_agent_list = error_agents
        
        if verbose and selected_metadata:
            print(f"  → Diverse errors selection: {max_agents_with_errors} agents with errors {selected_agent_list}, "
                       f"task={selected_metadata.get('task_id')}")
        
        return selected if selected else (failure_trajectories[0][0], failure_trajectories[0][1])
    
    else:
        print(f"Unknown strategy '{strategy}', falling back to random selection")
        loss, outputs, _ = random.choice(failure_trajectories)
        return loss, outputs


def verify_answer_gaia(target: str, predicted_answer: str) -> bool:
    """
    Use GAIA-style judge to verify if the predicted answer is correct.
    Returns True if correct, False otherwise.
    """

    def normalize_number_str(number_str: str) -> float:
        # we replace these common units and commas to allow conversion to float
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            logger.debug(f"String {number_str} cannot be normalized to number str.")
            return float("inf")

    def split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def normalize_str(input_str, remove_punct=True) -> str:
        """
        Normalize a string by:
        - Removing all white spaces
        - Optionally removing punctuation (if remove_punct is True)
        - Converting to lowercase
        """
        # Remove all white spaces. Required e.g for seagull vs. sea gull
        no_spaces = re.sub(r"\s", "", input_str)

        # Remove punctuation, if specified.
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()

    def is_float(element: Any) -> bool:
        try:
            _ = float(element)
            return True
        except ValueError:
            return False

    def question_scorer(model_answer: str, ground_truth: str) -> bool:
        if model_answer is None:
            model_answer = "None"

        # if gt is a number
        if is_float(ground_truth):
            logger.debug(f"Evaluating {model_answer} as a number.")
            normalized_answer = normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        # if gt is a list
        elif any(char in ground_truth for char in [",", ";"]):
            logger.debug(f"Evaluating {model_answer} as a comma separated list.")
            gt_elems = split_string(ground_truth)
            ma_elems = split_string(model_answer)

            # check length is the same
            if len(gt_elems) != len(ma_elems):
                warnings.warn(
                    "Answer lists have different lengths, returning False.", UserWarning
                )
                return False

            # compare each element as float or str
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    # we do not remove punct since comparisons can include punct
                    comparisons.append(
                        normalize_str(ma_elem, remove_punct=False)
                        == normalize_str(gt_elem, remove_punct=False)
                    )
            return all(comparisons)

        # if gt is a str
        else:
            logger.debug(f"Evaluating {model_answer} as a string.")
            return normalize_str(model_answer) == normalize_str(ground_truth)

    # Use the question_scorer to evaluate the answer
    try:
        is_correct = question_scorer(predicted_answer, target)
        return is_correct
    except Exception as e:
        print(f"GAIA evaluation failed: {e}")
        return False


class MiroFlowTask:
    """Wrapper for a single GAIA task"""
    def __init__(self, task_id: str, question: str, ground_truth: str, 
                 file_path: Optional[str] = None, metadata: Dict = None):
        self.task_id = task_id
        self.question = question
        self.ground_truth = ground_truth
        self.file_path = file_path
        self.metadata = metadata or {}


def load_gaia_dataset(data_dir: Path, metadata_file: str = "metadata.jsonl") -> List[MiroFlowTask]:
    """Load GAIA dataset from JSONL file"""
    metadata_path = data_dir / metadata_file
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    tasks = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Store the data_dir in metadata so we can construct correct path later
            metadata = data.get("metadata", {})
            metadata["data_dir"] = str(data_dir)
            
            task = MiroFlowTask(
                task_id=data["task_id"],
                question=data["task_question"],
                ground_truth=data["ground_truth"],
                file_path=data.get("file_path"),
                metadata=metadata
            )
            tasks.append(task)
    
    print(f"Loaded {len(tasks)} tasks from {metadata_path}")
    return tasks


def load_train_test_datasets(
    train_data_dir: Path,
    test_data_dir: Path,
    metadata_file: str = "standardized_data.jsonl",
    seed: int = 42
) -> Tuple[List[MiroFlowTask], List[MiroFlowTask]]:
    """
    Load datasets into train/test.
    
    Strategy:
    - Train: Load all data from train_data_dir (e.g., data/browsecomp-train)
    - Test: Load from test_data_dir (e.g., data/browsecomp-test with GT for evaluation)
    
    Args:
        train_data_dir: Path to training data with ground truth
        test_data_dir: Path to test data with GT (for evaluation)
        metadata_file: Name of the metadata file (default: standardized_data.jsonl)
        seed: Random seed for shuffling (default: 42)
    
    Returns:
        Tuple of (train_tasks, test_tasks)
    """
    # Load training dataset (has ground truth)
    print(f"\n{'='*60}")
    print(f"Loading training dataset from {train_data_dir}")
    print(f"{'='*60}")
    train_tasks = load_gaia_dataset(train_data_dir, metadata_file)
    print(f"Loaded {len(train_tasks)} training tasks (with GT)")
    
    # Shuffle training data
    random.seed(seed)
    random.shuffle(train_tasks)
    
    # Load test dataset (has ground truth, for evaluation)
    print(f"\n{'='*60}")
    print(f"Loading test dataset from {test_data_dir}")
    print(f"{'='*60}")
    test_tasks = load_gaia_dataset(test_data_dir, metadata_file)
    print(f"Loaded {len(test_tasks)} test tasks (with GT, for evaluation)")
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Train: {len(train_tasks)} (with GT, for training)")
    print(f"  Test:  {len(test_tasks)} (with GT, for evaluation)")
    print(f"{'='*60}\n")
    
    return train_tasks, test_tasks


async def run_single_inference(
    orchestrator: TextGradOrchestrator,
    task: MiroFlowTask,
    cfg: DictConfig,
    is_training: bool = True,
    logs_dir: Optional[Path] = None,
    epoch: int = 0,
    max_retries: int = 2,
    token_counter = None,
    log_subdir: str = "task_logs"
) -> Dict[str, Any]:
    """
    Run inference on a single task with retry mechanism.
    Retries when final_boxed_answer is missing (like common_benchmark.py).
    
    Args:
        max_retries: Maximum retry attempts when no final_boxed_answer (default: 2)
        log_subdir: Subdirectory for task logs (default: "task_logs")
    
    Returns:
        Dict with keys:
            - final_answer: str
            - final_boxed_answer: str (empty if failed)
            - agent_outputs: Dict mapping agent names to their final responses
            - task_log_path: Path
            - error: str (if exception occurred)
            - retry_count: int
    """
    # Ensure task_logging_context is set if logs_dir provided
    if logs_dir:
        from src.logging.logger import TASK_CONTEXT_VAR
        token = TASK_CONTEXT_VAR.set(task.task_id)
        logger.debug(f"[run_single_inference] Set task context: {task.task_id}")
    else:
        token = None
    
    retry_count = 0
    
    for attempt in range(max_retries + 1):  # Initial attempt + retries
        try:
            # Prepare task file path
            task_file = None
            if task.file_path:
                if Path(task.file_path).is_absolute():
                    task_file = task.file_path
                else:
                    # Use the data_dir stored in task metadata (from load_gaia_dataset)
                    task_data_dir = task.metadata.get("data_dir", cfg.benchmark.data.data_dir)
                    task_file = str(Path(task_data_dir) / task.file_path)
            
            # Run orchestrator (wrapped with TextGrad tracking)
            result = await orchestrator.run_main_agent(
                task_description=task.question,
                task_file_name=task_file,
                task_id=task.task_id,
                is_training=is_training,
                ground_truth=task.ground_truth,
                epoch=epoch,
                token_counter=token_counter,
                log_subdir=log_subdir
            )
            
            # Check if we have final_boxed_answer (like common_benchmark.py)
            has_answer = bool(result.get("final_boxed_answer"))
            
            if has_answer:
                # Success - has answer
                result["retry_count"] = retry_count
                if retry_count > 0:
                    print(f"Task {task.task_id} succeeded after {retry_count} retries")
                return result
            else:
                # No final_boxed_answer - retry if attempts remain
                if attempt < max_retries:
                    retry_count += 1
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(
                        f"Task {task.task_id} completed but no final_boxed_answer "
                        f"(attempt {attempt+1}/{max_retries+1}). Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # No more retries - return without answer
                    print(
                        f"Task {task.task_id} failed to produce final_boxed_answer "
                        f"after {retry_count} retries"
                    )
                    result["retry_count"] = retry_count
                    return result
            
        except Exception as e:
            # Exception occurred - don't retry, just return error (keep original behavior)
            print(f"Error running task {task.task_id}: {e}", exc_info=True)
            return {
                "final_answer": "",
                "final_boxed_answer": "",
                "agent_outputs": {},
                "error": str(e),
                "retry_count": retry_count
            }


async def train_epoch(
    cfg: DictConfig,
    train_tasks: List[MiroFlowTask],
    test_tasks: List[MiroFlowTask],
    orchestrator: TextGradOrchestrator,
    prompt_manager: PromptVariableManager,
    loss_module: MiroFlowTextualFeedbackLoss,
    optimizer: tg.TGD,
    epoch: int,
    batch_size: int = 10,
    max_concurrent: int = 2,
    selection_strategy: str = "max_feedback_length",
    best_accuracy: float = 0.0,
    best_prompts: Dict[str, str] = None,
    memory_manager = None,
    token_counter: Optional[TokenCounter] = None,
    timing_tracker: Optional[TimingTracker] = None
) -> Dict[str, Any]:
    """
    Train for one epoch with parallel task execution
    
    Args:
        max_concurrent: Maximum number of concurrent tasks (default: 2)
        selection_strategy: Strategy for selecting failure trajectories (default: "max_feedback_length")
            Options: "random", "max_feedback_length", "weighted_feedback", "diverse_errors"
        best_accuracy: Current best test accuracy
        best_prompts: Current best prompts
    
    Returns:
        Dict with training statistics including updated best_accuracy and best_prompts
    """
    print(f"\n{'='*60}\nEpoch {epoch+1}\n{'='*60}")
    print(f"Running training with max_concurrent={max_concurrent}")
    
    if best_prompts is None:
        best_prompts = {}
    
    # Shuffle training data
    random.shuffle(train_tasks)
    
    losses = []
    failure_trajectories = []
    correct_count = 0
    total_count = 0
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    logs_dir = Path(cfg.output_dir)
    print("✅ -- Logs Dir -- ", logs_dir)
    
    # Track failures
    exception_failures = []  # Tasks that raised exceptions
    no_answer_failures = []  # Tasks that completed but have no answer
    
    async def process_single_task(i: int, task: MiroFlowTask):
        """Process a single training task with task-specific logging"""
        async with semaphore:
            with task_logging_context(task.task_id, logs_dir):
                print(f"\n--- Training Task {i+1}/{len(train_tasks)}: {task.task_id} ---")
                
                # Record forward pass start time
                forward_start = time.time() if timing_tracker else None
                
                # Run inference with retry mechanism
                result = await run_single_inference(
                    orchestrator, task, cfg,
                    is_training=True,
                    logs_dir=logs_dir,
                    epoch=epoch,
                    max_retries=cfg.train.get("max_retries_per_task", 2),
                    token_counter=token_counter
                )
                
                # Record forward pass duration
                if timing_tracker and forward_start:
                    forward_duration = time.time() - forward_start
                    timing_tracker.record_forward(forward_duration)
                
                retry_count = result.get("retry_count", 0)
                
                # Check for exception
                if "error" in result:
                    print(f"Task {task.task_id} exception: {result['error']}")
                    exception_failures.append({
                        "task_id": task.task_id,
                        "error": result["error"]
                    })
                    return None  # Skip exception tasks
                
                # Check if we have final_boxed_answer
                has_answer = bool(result.get("final_boxed_answer"))
                
                if not has_answer:
                    # No answer after retries
                    print(f"Task {task.task_id} no final_boxed_answer after {retry_count} retries")
                    no_answer_failures.append({
                        "task_id": task.task_id,
                        "retry_count": retry_count
                    })
                    # Return with empty answer to generate feedback
                    return {
                        "task_id": task.task_id,
                        "is_correct": False,
                        "predicted_answer": "",
                        "question": task.question,
                        "ground_truth": task.ground_truth,
                        "no_answer_failure": True
                    }
                
                # Has answer - check correctness
                predicted_answer = result["final_boxed_answer"]
                
                if retry_count > 0:
                    print(f"✓ Task succeeded after {retry_count} retries")
                
                # Check correctness using GAIA-style verification
                is_correct = verify_answer_gaia(task.ground_truth, predicted_answer)
                
                print(f"Predicted: {predicted_answer} | Ground Truth: {task.ground_truth} | Correct: {is_correct}")
                
                return {
                    "task_id": task.task_id,
                    "is_correct": is_correct,
                    "predicted_answer": predicted_answer,
                    "question": task.question,
                    "ground_truth": task.ground_truth,
                    "retry_count": retry_count
                }
    
    # Process tasks in batches with batch-level optimization
    for batch_start in range(0, len(train_tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(train_tasks))
        batch_tasks = train_tasks[batch_start:batch_end]
        
        print(f"\n{'='*50}")
        print(f"Processing Batch {batch_start//batch_size + 1}: Tasks {batch_start+1}-{batch_end}/{len(train_tasks)}")
        print(f"{'='*50}")
        
        # Run batch tasks in parallel
        batch_results = await asyncio.gather(
            *[process_single_task(batch_start + i, task) for i, task in enumerate(batch_tasks)],
            return_exceptions=True
        )
        
        # Process batch results
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Exception in training task: {result}")
                continue
            
            if result is None:  # Exception failure, already logged
                continue
            
            total_count += 1
            
            if result["is_correct"]:
                correct_count += 1
            else:
                # Compute loss for incorrect predictions (including no-answer failures)
                try:
                    # Convert agent outputs to textgrad Variables
                    agent_vars = orchestrator.get_agent_output_variables()
                    
                    # For no-answer failures, add explicit context
                    prediction_text = result["predicted_answer"]
                    if result.get("no_answer_failure"):
                        prediction_text = "[NO ANSWER] Agent failed to produce a final_boxed_answer after multiple attempts. This may indicate unclear instructions, insufficient reasoning steps, or inadequate tool usage."
                    
                    loss = loss_module(
                        prediction_text=prediction_text,
                        problem_text=result["question"],
                        gold_answer=result["ground_truth"],
                        agent_outputs=agent_vars
                    )
                    
                    losses.append(loss)
                    
                    # Store failure trajectory with metadata
                    metadata = {
                        "task_id": result["task_id"],
                        "predicted": prediction_text,
                        "gold_answer": result["ground_truth"],
                        "no_answer_failure": result.get("no_answer_failure", False)
                    }
                    failure_trajectories.append((loss, agent_vars, metadata))
                    
                    # Store feedback in memory manager if available
                    if memory_manager:
                        try:
                            feedback_text = loss.get_value() if hasattr(loss, 'get_value') else str(loss.value)
                            from textgrad_components.gradient_handler import parse_agent_feedback
                            parsed_feedback = parse_agent_feedback(feedback_text)
                            
                            # Store feedback for each agent
                            for agent_name, agent_fb in parsed_feedback.items():
                                if agent_fb and agent_fb.strip().lower() not in ["correct reasoning", "not used", ""]:
                                    memory_manager.store_feedback(
                                        agent_name=agent_name,
                                        feedback_text=agent_fb,
                                        metadata={
                                            "iteration": batch_start // batch_size + 1,
                                            "epoch": epoch,
                                            "task_id": result["task_id"],
                                            "predicted": prediction_text[:200],  # Truncate for storage
                                            "gold_answer": result["ground_truth"]
                                        }
                                    )
                        except Exception as mem_error:
                            logger.warning(f"Failed to store feedback in memory: {mem_error}")
                    
                except Exception as e:
                    logger.error(f"Loss computation failed for task {result['task_id']}: {e}", exc_info=True)
        
        # Print batch progress (like Over-TextGrad-AC style)
        batch_num = batch_start // batch_size + 1
        train_acc = correct_count / total_count if total_count > 0 else 0.0
        print(f"[Epoch: {epoch}] - [batch {batch_num}] train acc={train_acc:.4f} ({correct_count}/{total_count}); best test acc={best_accuracy:.4f}")
        
        # Optimization step after batch processed (if enough failures)
        if len(losses) >= 1:
            print(f"\n*** Batch Optimization Step: {len(losses)} losses accumulated ***")
            print(f"Using selection strategy: {selection_strategy}")
            
            # Record backward start time
            backward_start = time.time() if timing_tracker else None
            
            try:
                if len(failure_trajectories) > 0:
                    print(f"Selecting from {len(failure_trajectories)} failure trajectories...")
                    
                    # Step 1: Collect all feedback texts (no backward yet)
                    all_feedbacks = [loss.get_value() for loss in losses]
                    print(f"Collected {len(all_feedbacks)} feedback texts")
                    
                    # Step 2: Parse and group feedback by agent
                    from textgrad_components.gradient_handler import parse_agent_feedback, summarize_agent_feedbacks
                    
                    agent_feedback_groups = {
                        "main_agent": [],
                        "agent-worker": []
                    }
                    
                    for feedback_text in all_feedbacks:
                        parsed = parse_agent_feedback(feedback_text)
                        for agent_name, agent_fb in parsed.items():
                            if agent_name in agent_feedback_groups:
                                agent_feedback_groups[agent_name].append(agent_fb)
                    
                    # Step 3: Summarize per-agent feedback
                    agent_summaries = {}
                    for agent_name, feedback_list in agent_feedback_groups.items():
                        if feedback_list:
                            summary = summarize_agent_feedbacks(
                                agent_name, 
                                feedback_list, 
                                orchestrator.evaluator_engine
                            )
                            agent_summaries[agent_name] = summary
                            print(f"  {agent_name}: {len(feedback_list)} feedback items summarized")
                    
                    # Step 4: Select ONE failure trajectory
                    selected_loss, selected_agent_outputs = select_trajectory_by_strategy(
                        failure_trajectories,
                        strategy=selection_strategy,
                        verbose=True
                    )
                    print(f"Selected 1 trajectory out of {len(failure_trajectories)} for gradient injection")
                    
                    # Step 5: Manually inject summarized feedback as gradients
                    visited = set()
                    prompt_variables = []
                    
                    def find_prompt_variables(var):
                        if var in visited:
                            return
                        visited.add(var)
                        
                        # Check if this is a prompt variable
                        role_desc = var.get_role_description() if hasattr(var, 'get_role_description') else ""
                        if "prompt" in role_desc.lower():
                            prompt_variables.append(var)
                        
                        if hasattr(var, 'predecessors') and var.predecessors:
                            for pred in var.predecessors:
                                find_prompt_variables(pred)
                    
                    find_prompt_variables(selected_loss)
                    
                    # Inject agent-specific summaries as gradients
                    for prompt_var in prompt_variables:
                        role_desc = prompt_var.get_role_description()
                        
                        # Determine which agent this prompt belongs to
                        agent_name = None
                        for name in ["main_agent", "agent-worker"]:
                            if name in role_desc:
                                agent_name = name
                                break
                        
                        if agent_name and agent_name in agent_summaries:
                            summary = agent_summaries[agent_name]
                            if summary:
                                gradient_var = tg.Variable(
                                    value=f"High-level feedback summary for {agent_name}:\n{summary}",
                                    role_description=f"gradient for {agent_name} prompt"
                                )
                                prompt_var.gradients.add(gradient_var)
                                print(f"  Injected gradient to {agent_name}")
                    
                    # Step 6: Apply optimizer step (no backward, only use injected gradients)
                    optimizer.step()
                    optimizer.zero_grad() ### TODO: WWW 验证是否需要
                    
                    # Record backward end time
                    backward_duration = 0
                    if timing_tracker and backward_start:
                        backward_duration = time.time() - backward_start
                        timing_tracker.record_backward(backward_duration)
                        print(f"Backward pass completed in {backward_duration:.2f}s")
                    
                    # Save step-level statistics to JSON file
                    step_stats = {
                        "epoch": epoch + 1,
                        "batch": batch_start // batch_size + 1,
                        "step_range": f"{batch_start}-{batch_end}",
                        "num_failures": len(losses),
                        "backward_time_seconds": backward_duration,
                        "timestamp": time.time()
                    }
                    
                    if token_counter:
                        step_stats["token_stats"] = token_counter.get_summary()
                    
                    if timing_tracker:
                        step_stats["timing_stats"] = timing_tracker.get_summary()
                    
                    # Save to step-level JSON file
                    step_stats_file = logs_dir / f"step_stats_epoch{epoch+1}_batch{batch_start//batch_size+1}.json"
                    with open(step_stats_file, "w", encoding="utf-8") as f:
                        json.dump(step_stats, f, indent=2, ensure_ascii=False)
                    
                    print(f"Step statistics saved to {step_stats_file}")
                    
                    # Log updated prompts
                    print("\n=== Updated Prompts ===")
                    for param in prompt_manager.trainable_parameters():
                        print(f"[{param.role_description}]:\n{param.get_value()[:500]}...\n")
                    
                    # Clear for next batch
                    optimizer.zero_grad()
                    losses.clear()
                    failure_trajectories.clear()
                    
                    # Evaluate immediately after prompt update using two-stage evaluation
                    print(f"\n*** Evaluating after batch optimization ***")
                    
                    # Calculate small set size (10% of test data)
                    small_set_size = max(1, int(len(test_tasks) * 0.1))
                    
                    # Stage 1: Evaluate small set (10%)
                    print(f"\n{'='*60}")
                    print(f"评估前{small_set_size}条测试数据 (10%)...")
                    print(f"{'='*60}\n")
                    
                    small_acc = await evaluate(
                        cfg=cfg,
                        test_tasks=test_tasks[:small_set_size],
                        orchestrator=orchestrator,
                        max_eval_tasks=small_set_size,
                        max_concurrent=max_concurrent
                    )
                    
                    print(f"\n前{small_set_size}条数据准确率: {small_acc:.3f}")
                    
                    # Stage 2: If small set improved, evaluate full test set
                    if small_acc > best_accuracy:
                        print(f"\n小数据集性能提升! 评估全部{len(test_tasks)}条数据...")
                        print(f"{'='*60}\n")
                        
                        curr_accuracy = await evaluate(
                            cfg=cfg,
                            test_tasks=test_tasks,
                            orchestrator=orchestrator,
                            max_eval_tasks=len(test_tasks),
                            max_concurrent=max_concurrent
                        )
                        
                        # Update best prompts if improved
                        if curr_accuracy >= best_accuracy:
                            print(f"\n🎉 New best accuracy: {curr_accuracy:.2%} (previous: {best_accuracy:.2%})")
                            best_accuracy = curr_accuracy
                            best_prompts = {param.role_description: param.get_value() for param in prompt_manager.trainable_parameters()}
                        else:
                            print(f"\n⚠️ Accuracy decreased: {curr_accuracy:.2%} < {best_accuracy:.2%}. Reverting to best prompts.")
                            # Revert to best prompts
                            for param in prompt_manager.trainable_parameters():
                                if param.role_description in best_prompts:
                                    param.set_value(best_prompts[param.role_description])
                                    print(f"Reverted {param.role_description}")
                    else:
                        print(f"\n小数据集性能未提升 ({small_acc:.3f} <= {best_accuracy:.3f}), 跳过全量评估")
                        print(f"保持最佳准确率: {best_accuracy:.2%}")
                        # Revert to best prompts since small set didn't improve
                        for param in prompt_manager.trainable_parameters():
                            if param.role_description in best_prompts:
                                param.set_value(best_prompts[param.role_description])
                                print(f"Reverted {param.role_description}")
                else:
                    print("No failure trajectories available for optimization")
                
            except Exception as e:
                print(f"Batch optimization step failed: {e}", exc_info=True)
                # Record backward time even on exception
                if timing_tracker and backward_start:
                    backward_duration = time.time() - backward_start
                    timing_tracker.record_backward(backward_duration)
                optimizer.zero_grad()
                losses.clear()
                failure_trajectories.clear()
    
    # Force clear any remaining state at epoch end
    optimizer.zero_grad()
    losses.clear()
    failure_trajectories.clear()
    
    train_accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Print failure summary
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Failure Summary:")
    print(f"  Exception failures (skipped): {len(exception_failures)}")
    if exception_failures:
        for failure in exception_failures[:3]:  # Show first 3
            print(f"    - {failure['task_id']}: {failure['error'][:100]}...")
        if len(exception_failures) > 3:
            print(f"    ... and {len(exception_failures) - 3} more")
    
    print(f"  No-answer failures (used for training): {len(no_answer_failures)}")
    if no_answer_failures:
        for failure in no_answer_failures[:3]:  # Show first 3
            print(f"    - {failure['task_id']}: no answer after {failure['retry_count']} retries")
        if len(no_answer_failures) > 3:
            print(f"    ... and {len(no_answer_failures) - 3} more")
    print(f"{'='*60}\n")
    
    # Save failure details to file
    failure_summary_path = logs_dir / f"failures_epoch{epoch+1}.json"
    with open(failure_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "epoch": epoch + 1,
            "exception_failures": exception_failures,
            "no_answer_failures": no_answer_failures,
            "total_tasks": len(train_tasks),
            "successful_tasks": total_count,
            "success_rate": total_count / len(train_tasks) if train_tasks else 0.0
        }, f, indent=2, ensure_ascii=False)
    print(f"Failure details saved to {failure_summary_path}")
    
    stats = {
        "epoch": epoch + 1,
        "train_accuracy": train_accuracy,
        "correct": correct_count,
        "total": total_count,
        "exception_failures": len(exception_failures),
        "no_answer_failures": len(no_answer_failures),
        "best_accuracy": best_accuracy,
        "best_prompts": best_prompts
    }
    
    # Add token and timing statistics if available
    if token_counter:
        stats["token_stats"] = token_counter.get_summary()
        print(f"\n{'='*60}")
        print("Token Usage Statistics (Epoch {})".format(epoch + 1))
        print(f"{'='*60}")
        token_stats = token_counter.get_summary()
        print(f"Backward Pass:")
        print(f"  Calls: {token_stats['backward']['calls']}")
        print(f"  Prompt tokens: {token_stats['backward']['prompt_tokens']}")
        print(f"  Completion tokens: {token_stats['backward']['completion_tokens']}")
        print(f"  Total tokens: {token_stats['backward']['total_tokens']}")
        print(f"API Calls:")
        print(f"  Calls: {token_stats['api_calls']['calls']}")
        print(f"  Prompt tokens: {token_stats['api_calls']['prompt_tokens']}")
        print(f"  Completion tokens: {token_stats['api_calls']['completion_tokens']}")
        print(f"  Total tokens: {token_stats['api_calls']['total_tokens']}")
        print(f"Total tokens used: {token_stats['total_tokens']}")
        print(f"{'='*60}\n")
    
    if timing_tracker:
        stats["timing_stats"] = timing_tracker.get_summary()
        print(f"\n{'='*60}")
        print("Timing Statistics (Epoch {})".format(epoch + 1))
        print(f"{'='*60}")
        timing_stats = timing_tracker.get_summary()
        print(f"Forward Pass:")
        print(f"  Calls: {timing_stats['forward']['calls']}")
        print(f"  Total time: {timing_stats['forward']['total_time']:.2f}s")
        print(f"  Avg time per call: {timing_stats['forward']['avg_time']:.2f}s")
        print(f"Backward Pass:")
        print(f"  Calls: {timing_stats['backward']['calls']}")
        print(f"  Total time: {timing_stats['backward']['total_time']:.2f}s")
        print(f"  Avg time per call: {timing_stats['backward']['avg_time']:.2f}s")
        print(f"Total training time: {timing_stats['total_training_time']:.2f}s")
        print(f"{'='*60}\n")
    
    print(f"\nEpoch {epoch+1} Stats: {stats}")
    return stats


async def evaluate(
    cfg: DictConfig,
    test_tasks: List[MiroFlowTask],
    orchestrator: TextGradOrchestrator,
    max_eval_tasks: int = 50,
    max_concurrent: int = 2
) -> float:
    """
    Evaluate on validation set with parallel task execution (tasks must have GT)
    
    Args:
        max_concurrent: Maximum number of concurrent tasks (default: 2)
    
    Returns:
        Validation accuracy
    """
    eval_tasks = test_tasks[:max_eval_tasks]
    print(f"\n{'='*60}\nEvaluation on {len(eval_tasks)} validation tasks\n{'='*60}")
    print(f"Running evaluation with max_concurrent={max_concurrent}")
    
    correct_count = 0
    total_count = 0
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    logs_dir = Path(cfg.output_dir)
    
    async def process_single_eval_task(i: int, task: MiroFlowTask):
        """Process a single evaluation task with task-specific logging"""
        async with semaphore:
            with task_logging_context(task.task_id, logs_dir):
                print(f"\n--- Val Task {i+1}/{len(eval_tasks)}: {task.task_id} ---")
                
                result = await run_single_inference(orchestrator, task, cfg, is_training=False, logs_dir=logs_dir, epoch=-1, log_subdir="val_inference_logs")
                
                if "error" in result:
                    print(f"Evaluation task {task.task_id} failed: {result['error']}")
                    return None
                
                predicted_answer = result["final_boxed_answer"]
                
                # Check correctness using GAIA-style verification
                is_correct = verify_answer_gaia(task.ground_truth, predicted_answer)
                
                print(f"Predicted: {predicted_answer} | Ground Truth: {task.ground_truth} | Correct: {is_correct}")
                
                return {
                    "task_id": task.task_id,
                    "is_correct": is_correct
                }
    
    # Run all evaluation tasks in parallel
    results = await asyncio.gather(
        *[process_single_eval_task(i, task) for i, task in enumerate(eval_tasks)],
        return_exceptions=True
    )
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception in evaluation task: {result}")
            continue
        
        if result is None:
            continue
        
        total_count += 1
        if result["is_correct"]:
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"\nValidation Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    return accuracy


async def evaluate_dataset_two_stage(
    cfg: DictConfig,
    test_tasks: List[MiroFlowTask],
    orchestrator: TextGradOrchestrator,
    dataset_name: str = "test",
    max_concurrent: int = 8,
    small_set_ratio: float = 0.1
) -> Tuple[float, float]:
    """
    Two-stage evaluation on test set:
    1. First evaluate on small_set_ratio (e.g., 10%) of data
    2. If accuracy improves, evaluate on full dataset
    
    Args:
        cfg: Configuration
        test_tasks: List of test tasks with ground truth
        orchestrator: TextGrad orchestrator
        dataset_name: Name of dataset for logging
        max_concurrent: Maximum concurrent tasks (default: 8)
        small_set_ratio: Ratio of data for initial evaluation (default: 0.1)
    
    Returns:
        Tuple of (small_set_accuracy, full_accuracy)
    """
    # Calculate small set size (10% of data)
    small_set_size = max(1, int(len(test_tasks) * small_set_ratio))
    
    # Stage 1: Evaluate on small set (first 10%)
    print(f"\n{'='*60}")
    print(f"初始评估：评估前{small_set_size}条数据...")
    print(f"{'='*60}\n")
    
    small_correct = 0
    small_total = 0
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    logs_dir = Path(cfg.output_dir)
    
    async def process_single_task(i: int, task: MiroFlowTask):
        """Process a single test task"""
        async with semaphore:
            with task_logging_context(task.task_id, logs_dir):
                print(f"\n--- Test Task {i+1}/{len(test_tasks)}: {task.task_id} ---")
                
                result = await run_single_inference(
                    orchestrator, task, cfg,
                    is_training=False,
                    logs_dir=logs_dir,
                    epoch=-1,
                    max_retries=cfg.train.get("max_retries_per_task", 2),
                    log_subdir="test_evaluation_logs"
                )
                
                if "error" in result:
                    print(f"Test task {task.task_id} failed: {result['error']}")
                    return None
                
                predicted_answer = result.get("final_boxed_answer", "")
                
                # Check correctness using GAIA-style verification
                is_correct = verify_answer_gaia(task.ground_truth, predicted_answer)
                
                print(f"Predicted: {predicted_answer} | Ground Truth: {task.ground_truth} | Correct: {is_correct}")
                
                return {
                    "task_id": task.task_id,
                    "is_correct": is_correct,
                    "predicted": predicted_answer,
                    "ground_truth": task.ground_truth
                }
    
    # Evaluate small set
    small_results = await asyncio.gather(
        *[process_single_task(i, task) for i, task in enumerate(test_tasks[:small_set_size])],
        return_exceptions=True
    )
    
    # Process small set results
    for result in small_results:
        if isinstance(result, Exception):
            print(f"Exception in test task: {result}")
            continue
        if result is None:
            continue
        small_total += 1
        if result["is_correct"]:
            small_correct += 1
    
    small_acc = small_correct / small_total if small_total > 0 else 0.0
    print(f"\n前{small_set_size}条数据准确率: {small_acc:.3f} ({small_correct}/{small_total})")
    
    # Stage 2: Evaluate full dataset
    print(f"\n{'='*60}")
    print(f"初始评估：评估全部{len(test_tasks)}条数据...")
    print(f"{'='*60}\n")
    
    # Evaluate remaining data (from small_set_size onwards)
    full_results = await asyncio.gather(
        *[process_single_task(i, task) for i, task in enumerate(test_tasks[small_set_size:], start=small_set_size)],
        return_exceptions=True
    )
    
    # Process full set results (combining small + remaining)
    full_correct = small_correct
    full_total = small_total
    
    for result in full_results:
        if isinstance(result, Exception):
            print(f"Exception in test task: {result}")
            continue
        if result is None:
            continue
        full_total += 1
        if result["is_correct"]:
            full_correct += 1
    
    full_acc = full_correct / full_total if full_total > 0 else 0.0
    print(f"\n全部数据准确率: {full_acc:.3f} ({full_correct}/{full_total})")
    print(f"{'='*60}\n")
    
    return small_acc, full_acc


async def generate_test_predictions(
    cfg: DictConfig,
    test_tasks: List[MiroFlowTask],
    orchestrator: TextGradOrchestrator,
    output_file: Path,
    max_concurrent: int = 2
) -> None:
    """
    Generate predictions on test set (no GT available, for submission)
    
    Args:
        cfg: Configuration
        test_tasks: List of test tasks (without ground truth)
        orchestrator: TextGrad orchestrator with best prompts loaded
        output_file: Path to save predictions (JSONL format)
        max_concurrent: Maximum number of concurrent tasks
    """
    print(f"\n{'='*60}")
    print(f"Generating Test Predictions")
    print(f"{'='*60}")
    print(f"Test tasks: {len(test_tasks)}")
    print(f"Output file: {output_file}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"{'='*60}\n")
    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    logs_dir = Path(cfg.output_dir) / "test_inference"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = []
    
    async def process_single_test_task(i: int, task: MiroFlowTask):
        """Process a single test task"""
        async with semaphore:
            with task_logging_context(task.task_id, logs_dir):
                print(f"\n--- Test Task {i+1}/{len(test_tasks)}: {task.task_id} ---")
                
                result = await run_single_inference(
                    orchestrator, task, cfg,
                    is_training=False,
                    logs_dir=logs_dir,
                    epoch=-1,
                    max_retries=cfg.train.get("max_retries_per_task", 2),
                    log_subdir="test_inference_logs"
                )
                
                if "error" in result:
                    print(f"Test task {task.task_id} failed: {result['error']}")
                    predicted_answer = ""
                else:
                    predicted_answer = result.get("final_boxed_answer", "")
                
                print(f"Predicted: {predicted_answer}")
                
                return {
                    "task_id": task.task_id,
                    "model_answer": predicted_answer,
                    "question": task.question
                }
    
    # Run all test tasks in parallel
    results = await asyncio.gather(
        *[process_single_test_task(i, task) for i, task in enumerate(test_tasks)],
        return_exceptions=True
    )
    
    # Collect predictions
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception in test task: {result}")
            continue
        if result is not None:
            predictions.append(result)
    
    # Save predictions to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"Test Predictions Complete")
    print(f"{'='*60}")
    print(f"Generated {len(predictions)}/{len(test_tasks)} predictions")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")


async def main_training_loop(cfg: DictConfig):
    """Main training loop with train/test split"""
    init_logging_for_benchmark_evaluation(print_task_logs=False)
    
    # Initialize token counter and timing tracker
    token_counter = TokenCounter()
    timing_tracker = TimingTracker()
    timing_tracker.start_training()
    
    # Load datasets: train from browsecomp-train, test from browsecomp-test
    train_data_dir = Path(cfg.benchmark.data.train_data_dir)
    test_data_dir = Path(cfg.benchmark.data.test_data_dir)
    metadata_file = cfg.benchmark.data.metadata_file
    print("✅ -- Train/Test Data Dir -- ", train_data_dir, test_data_dir)
    train_tasks, test_tasks = load_train_test_datasets(
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        metadata_file=metadata_file,
        seed=cfg.train.seed
    )
    
    # Initialize pipeline components
    logs_dir = Path(cfg.output_dir)
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = \
        create_pipeline_components(cfg, logs_dir=str(logs_dir))
    
    # Initialize prompt manager
    prompt_manager = PromptVariableManager(cfg)
    
    # Initialize TextGrad orchestrator
    orchestrator = TextGradOrchestrator(
        cfg=cfg,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        prompt_manager=prompt_manager
    )
    
    # Initialize memory manager if available and enabled in config
    memory_manager = None
    if MemoryManager and cfg.train.get("use_memory", False):
        memory_strategy = cfg.train.get("memory_strategy", "loss_bank")
        memory_storage = cfg.train.get("memory_storage", str(logs_dir / "memory_storage"))
        print(f"\n{'='*60}")
        print(f"Initializing Memory Manager with strategy: {memory_strategy}")
        print(f"Memory storage path: {memory_storage}")
        print(f"{'='*60}\n")
        
        memory_manager = MemoryManager(
            strategy=memory_strategy,
            storage_path=memory_storage,
            max_entries_per_agent=cfg.train.get("memory_max_entries", 100),
            window_size=cfg.train.get("memory_window", 50)
        )
    
    # Initialize loss module with memory manager
    loss_module = MiroFlowTextualFeedbackLoss(
        engine=orchestrator.evaluator_engine,
        logger=logger,
        memory_manager=memory_manager
    )
    
    # Attach token_counter to evaluator_engine for tracking backward pass tokens
    if orchestrator.evaluator_engine:
        orchestrator.evaluator_engine.token_counter = token_counter
        logger.info("Attached token_counter to evaluator_engine")
    
    # Set global backward engine for textgrad
    tg.set_backward_engine(orchestrator.evaluator_engine)
    
    # Initialize optimizer
    trainable_params = prompt_manager.trainable_parameters()
    optimizer = tg.TGD(
        parameters=trainable_params,
        engine=orchestrator.evaluator_engine,
        constraints=[
            "Keep prompts clear, concise, and actionable",
            "Do not change the fundamental role of each agent",
            "Maintain consistency in prompt structure and format"
        ],
        gradient_memory=2,
        verbose=1
    )
    
    # Training loop
    best_accuracy = 0.0
    best_prompts = {param.role_description: param.get_value() for param in trainable_params}
    
    # Initialize prompts with actual tool definitions ONCE before training
    # CRITICAL: Do this BEFORE any training, and do NOT repeat in every epoch
    # After optimizer.step() updates prompts, they should NOT be regenerated
    print(f"\n{'='*60}")
    print("Initializing prompts with actual tool definitions")
    print(f"{'='*60}")
    from src.core.orchestrator import _list_tools
    tool_definitions_main = await main_agent_tool_manager.get_all_tool_definitions()
    if cfg.sub_agents is not None and cfg.sub_agents:
        from src.utils.tool_utils import expose_sub_agents_as_tools
        tool_definitions_main = expose_sub_agents_as_tools(cfg.sub_agents)
    _list_sub_agent_tools = _list_tools(sub_agent_tool_managers)
    tool_definitions_sub = await _list_sub_agent_tools()
    sub_agent_name = 'agent-worker'
    tool_definitions_sub = tool_definitions_sub.get(sub_agent_name, []) 
    chinese_context = cfg.main_agent.get("chinese_context", "false").lower() == "true"
    prompt_manager.initialize_prompts_with_tools(
        tool_definitions_main=tool_definitions_main,
        tool_definitions_sub=tool_definitions_sub,
        chinese_context=chinese_context
    )
    print(f"✅ Initialized prompts with main_agent:{tool_definitions_main} tool definitions and sub_agents:{tool_definitions_sub} tool definitions")
    print(f"   These prompts will be optimized by TextGrad and should NOT be regenerated")
    print(f"{'='*60}\n")
    
    # Get max_concurrent from config (default: 2)
    max_concurrent = cfg.train.get("max_concurrent", 2)
    print(f"Using max_concurrent={max_concurrent} for parallel execution")
    
    # Get selection_strategy from config (default: "max_feedback_length")
    selection_strategy = cfg.train.get("selection_strategy", "max_feedback_length")
    print(f"Using selection_strategy={selection_strategy} for trajectory selection")
    
    # Initial evaluation before training on test set (optional, using two-stage evaluation)
    run_initial_eval = cfg.train.get("run_initial_eval", False)
    best_small_performance = 0.0
    if run_initial_eval:
        print("\n" + "="*60)
        print("Initial Evaluation on Test Set (Before Training)")
        print("="*60)
        small_acc, full_acc = await evaluate_dataset_two_stage(
            cfg=cfg,
            test_tasks=test_tasks,
            orchestrator=orchestrator,
            dataset_name="test_initial",
            max_concurrent=max_concurrent,
            small_set_ratio=0.1
        )
        best_accuracy = full_acc
        best_small_performance = small_acc
        print(f"Initial test accuracy - Small set: {small_acc:.2%}, Full set: {full_acc:.2%}")
    else:
        print("\n" + "="*60)
        print("Skipping Initial Evaluation (run_initial_eval=False)")
        print("="*60)
        best_accuracy = 0.0
    
    for epoch in range(cfg.train.num_epochs):
        # Update memory manager epoch if available
        if memory_manager:
            memory_manager.update_iteration(iteration=0, epoch=epoch)
        
        # Reset token counter for this epoch
        token_counter.reset()
        
        # Train on train_tasks (no separate val set, will evaluate on test set)
        train_stats = await train_epoch(
            cfg=cfg,
            train_tasks=train_tasks[:cfg.train.get("max_train_tasks_per_epoch", len(train_tasks))],
            test_tasks=test_tasks,  # Pass test_tasks for two-stage evaluation
            orchestrator=orchestrator,
            prompt_manager=prompt_manager,
            loss_module=loss_module,
            optimizer=optimizer,
            epoch=epoch,
            batch_size=cfg.train.batch_size,
            max_concurrent=max_concurrent,
            selection_strategy=selection_strategy,
            best_accuracy=best_accuracy,
            best_prompts=best_prompts,
            memory_manager=memory_manager,
            token_counter=token_counter,
            timing_tracker=timing_tracker
        )
        
        # Update best accuracy and prompts from train_stats
        best_accuracy = train_stats.get("best_accuracy", best_accuracy)
        best_prompts = train_stats.get("best_prompts", best_prompts)
        
        # Save best prompts after each epoch
        save_path = logs_dir / f"best_prompts_epoch{epoch+1}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "epoch": epoch + 1,
                "best_accuracy": best_accuracy,
                "train_accuracy": train_stats["train_accuracy"],
                "prompts": best_prompts
            }, f, indent=2, ensure_ascii=False)
        print(f"Best prompts saved to {save_path}")
    
    print(f"\n{'='*60}\nTraining Complete\n{'='*60}")
    print(f"Best test accuracy during training: {best_accuracy:.2%}")
    
    # Load best prompts before final test evaluation
    print(f"\n{'='*60}")
    print("Loading Best Prompts for Final Test Evaluation")
    print(f"{'='*60}")
    for param in prompt_manager.trainable_parameters():
        if param.role_description in best_prompts:
            param.set_value(best_prompts[param.role_description])
            print(f"✓ Loaded best prompt for {param.role_description}")
    
    # Final test evaluation with ground truth (calculate accuracy)
    if cfg.train.get("generate_test_predictions", True):
        print(f"\n{'='*60}")
        print("Final Test Evaluation (with Ground Truth)")
        print(f"{'='*60}\n")
        
        small_acc, full_acc = await evaluate_dataset_two_stage(
            cfg=cfg,
            test_tasks=test_tasks,
            orchestrator=orchestrator,
            dataset_name="test_final",
            max_concurrent=max_concurrent,
            small_set_ratio=0.1
        )
        
        print(f"\n{'='*60}")
        print(f"Final Test Results:")
        print(f"  Small set (10%) accuracy: {small_acc:.2%}")
        print(f"  Full test set accuracy: {full_acc:.2%}")
        print(f"{'='*60}\n")
        
        # Save final test results
        final_results_path = logs_dir / "final_test_results.json"
        with open(final_results_path, "w", encoding="utf-8") as f:
            json.dump({
                "small_set_accuracy": small_acc,
                "full_test_accuracy": full_acc,
                "best_training_accuracy": best_accuracy,
                "total_test_tasks": len(test_tasks),
                "small_set_size": int(len(test_tasks) * 0.1)
            }, f, indent=2, ensure_ascii=False)
        print(f"Final test results saved to: {final_results_path}")
    else:
        print(f"\nSkipping final test evaluation (generate_test_predictions=False)")
    
    # Save memory and print statistics if available
    if memory_manager:
        print(f"\n{'='*60}")
        print("Memory Manager Statistics:")
        print(f"{'='*60}")
        memory_stats = memory_manager.get_statistics()
        for key, value in memory_stats.items():
            print(f"  {key}: {value}")
        
        memory_manager.save_all()
        print(f"\nMemory saved to disk.")
        print(f"{'='*60}\n")
    
    # Record end time and print final statistics
    timing_tracker.end_training()
    
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    print(f"Best test accuracy during training: {best_accuracy:.2%}")
    print(f"Best prompts saved to: {logs_dir}")
    print(f"{'='*60}\n")
    
    # Print final token statistics
    print(f"\n{'='*60}")
    print("Final Token Usage Summary (All Epochs)")
    print(f"{'='*60}")
    final_token_stats = token_counter.get_summary()
    print(f"Backward Pass (Gradient Computation):")
    print(f"  Total calls: {final_token_stats['backward']['calls']}")
    print(f"  Prompt tokens: {final_token_stats['backward']['prompt_tokens']}")
    print(f"  Completion tokens: {final_token_stats['backward']['completion_tokens']}")
    print(f"  Total backward tokens: {final_token_stats['backward']['total_tokens']}")
    print(f"API Calls (Forward Pass):")
    print(f"  Total calls: {final_token_stats['api_calls']['calls']}")
    print(f"  Prompt tokens: {final_token_stats['api_calls']['prompt_tokens']}")
    print(f"  Completion tokens: {final_token_stats['api_calls']['completion_tokens']}")
    print(f"  Total API tokens: {final_token_stats['api_calls']['total_tokens']}")
    print(f"Grand Total Tokens Used: {final_token_stats['total_tokens']}")
    print(f"{'='*60}\n")
    
    # Print final timing statistics
    print(f"\n{'='*60}")
    print("Final Timing Summary (All Epochs)")
    print(f"{'='*60}")
    final_timing_stats = timing_tracker.get_summary()
    print(f"Forward Pass:")
    print(f"  Total calls: {final_timing_stats['forward']['calls']}")
    print(f"  Total time: {final_timing_stats['forward']['total_time']:.2f}s")
    print(f"  Avg time per call: {final_timing_stats['forward']['avg_time']:.2f}s")
    print(f"Backward Pass:")
    print(f"  Total calls: {final_timing_stats['backward']['calls']}")
    print(f"  Total time: {final_timing_stats['backward']['total_time']:.2f}s")
    print(f"  Avg time per call: {final_timing_stats['backward']['avg_time']:.2f}s")
    print(f"Total Training Time: {final_timing_stats['total_training_time']:.2f}s")
    print(f"{'='*60}\n")
    
    # Save statistics to file
    stats_file = logs_dir / "training_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "token_stats": final_token_stats,
            "timing_stats": final_timing_stats,
            "best_accuracy": best_accuracy,
            "total_epochs": cfg.train.num_epochs
        }, f, indent=2, ensure_ascii=False)
    print(f"Training statistics saved to: {stats_file}\n")
    
    print(f"\nBest prompts preview:")
    for role, prompt in best_prompts.items():
        print(f"\n[{role}]:\n{prompt[:300]}...\n")


@hydra.main(version_base=None, config_path="config", config_name="train_textgrad_gaia")
def main(cfg: DictConfig):
    """Entry point with Hydra configuration"""
    print("="*60)
    print("MiroFlow TextGrad Training")
    print("="*60)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    asyncio.run(main_training_loop(cfg))


if __name__ == "__main__":
    main()