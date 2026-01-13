"""
Memory mechanisms for storing and utilizing historical feedback in MiroFlow TextGrad training.
Adapted from Over-TextGrad's memory_mechanisms.py for MiroFlow's multi-agent system.
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime

from src.logging.logger import bootstrap_logger

logger = bootstrap_logger()


class FeedbackMemory:
    """Base class for feedback memory mechanisms."""
    
    def __init__(self, storage_path: str = "./memory_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def store(self, feedback: Dict[str, Any]) -> str:
        """Store a feedback entry and return its ID."""
        raise NotImplementedError
        
    def retrieve(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant feedback entries."""
        raise NotImplementedError
        
    def save_to_disk(self):
        """Persist memory to disk."""
        raise NotImplementedError
        
    def load_from_disk(self):
        """Load memory from disk."""
        raise NotImplementedError


class LossBank(FeedbackMemory):
    """
    Loss Bank: Stores historical feedbacks and uses them to enhance current loss generation.
    
    Design:
    - Maintains a rolling window of recent feedbacks per agent
    - Stores feedback with metadata (iteration, accuracy, error patterns)
    - When generating new loss, retrieves relevant historical feedbacks
    - Uses LLM to synthesize historical patterns into better current feedback
    
    Benefits:
    - System prompts stay clean (only updated in backward)
    - Historical patterns inform current loss without explicit prompt modification
    - Can identify recurring error patterns across iterations
    """
    
    def __init__(self, 
                 storage_path: str = "./memory_storage/loss_bank",
                 max_entries_per_agent: int = 100,
                 window_size: int = 50):
        super().__init__(storage_path)
        self.max_entries_per_agent = max_entries_per_agent
        self.window_size = window_size
        
        # Separate storage for each agent type
        self.agent_feedbacks = {
            "main_agent": deque(maxlen=max_entries_per_agent),
            "agent-worker": deque(maxlen=max_entries_per_agent),
            # "agent-browser": deque(maxlen=max_entries_per_agent),
            # "agent-coder": deque(maxlen=max_entries_per_agent)
        }
        
        # Statistics tracking
        self.stats = {
            "total_stored": 0,
            "by_agent": defaultdict(int),
            "by_iteration": defaultdict(int)
        }
        
        # Load existing data if available
        self.load_from_disk()
        
    def store(self, feedback: Dict[str, Any]) -> str:
        """
        Store a feedback entry.
        
        Args:
            feedback: Dict containing:
                - agent_name: str
                - feedback_text: str
                - iteration: int
                - epoch: int
                - metadata: Dict (predicted, gold_answer, task_id, etc.)
        
        Returns:
            feedback_id: str
        """
        agent_name = feedback.get("agent_name")
        
        if agent_name not in self.agent_feedbacks:
            logger.warning(f"Unknown agent {agent_name}, adding to memory")
            self.agent_feedbacks[agent_name] = deque(maxlen=self.max_entries_per_agent)
        
        # Generate unique ID
        timestamp = datetime.now().isoformat()
        feedback_id = hashlib.md5(
            f"{agent_name}_{timestamp}_{feedback.get('feedback_text', '')}".encode()
        ).hexdigest()[:12]
        
        # Add metadata
        entry = {
            "id": feedback_id,
            "timestamp": timestamp,
            **feedback
        }
        
        # Store in agent-specific queue
        self.agent_feedbacks[agent_name].append(entry)
        
        # Update statistics
        self.stats["total_stored"] += 1
        self.stats["by_agent"][agent_name] += 1
        if "iteration" in feedback:
            self.stats["by_iteration"][feedback["iteration"]] += 1
        
        return feedback_id
    
    def retrieve(self, 
                 query: Dict[str, Any], 
                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant historical feedbacks for an agent.
        
        Args:
            query: Dict containing:
                - agent_name: str (required)
                - error_type: str (optional, filter by error pattern)
                - recent_only: bool (optional, only from last N iterations)
        
        Returns:
            List of relevant feedback entries
        """
        agent_name = query.get("agent_name")
        if not agent_name or agent_name not in self.agent_feedbacks:
            return []
        
        feedbacks = list(self.agent_feedbacks[agent_name])
        
        # Apply filters
        if query.get("error_type"):
            error_type = query["error_type"].lower()
            feedbacks = [
                fb for fb in feedbacks
                if error_type in fb.get("feedback_text", "").lower()
            ]
        
        if query.get("recent_only"):
            # Get last window_size feedbacks
            feedbacks = feedbacks[-self.window_size:]
        
        # Return most recent top_k
        return feedbacks[-top_k:] if len(feedbacks) > top_k else feedbacks
    
    def get_error_patterns(self, agent_name: str) -> Dict[str, Any]:
        """
        Analyze error patterns for an agent.
        
        Returns:
            Dict with pattern analysis:
                - common_errors: List of frequently occurring error types
                - trend: "improving", "stable", or "degrading"
                - total_feedbacks: int
        """
        feedbacks = list(self.agent_feedbacks.get(agent_name, []))
        if not feedbacks:
            return {"common_errors": [], "trend": "stable", "total_feedbacks": 0}
        
        # Extract error keywords
        error_keywords = defaultdict(int)
        
        for fb in feedbacks:
            feedback_text = fb.get("feedback_text", "").lower()
            # Look for common error patterns
            for keyword in ["incorrect", "wrong", "error", "missing", "incomplete", 
                           "inaccurate", "failed", "issue", "problem"]:
                if keyword in feedback_text:
                    error_keywords[keyword] += 1
        
        # Determine trend (simple heuristic)
        trend = "stable"
        if len(feedbacks) >= 10:
            recent = feedbacks[-5:]
            older = feedbacks[-10:-5]
            recent_count = len(recent)
            older_count = len(older)
            if recent_count < older_count * 0.8:
                trend = "improving"
            elif recent_count > older_count * 1.2:
                trend = "degrading"
        
        common_errors = sorted(error_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "common_errors": [err[0] for err in common_errors],
            "error_counts": dict(common_errors),
            "trend": trend,
            "total_feedbacks": len(feedbacks)
        }
    
    def enhance_current_loss(self, 
                            current_feedback: str,
                            agent_name: str,
                            llm_engine: Any,
                            max_history: int = 5) -> str:
        """
        Use historical feedbacks to enhance current loss feedback.
        
        Args:
            current_feedback: Current feedback text
            agent_name: Agent being evaluated
            llm_engine: LLM engine for synthesis
            max_history: Number of historical feedbacks to use
        
        Returns:
            Enhanced feedback text
        """
        # Retrieve relevant historical feedbacks
        historical = self.retrieve(
            query={"agent_name": agent_name, "recent_only": True},
            top_k=max_history
        )
        
        if not historical:
            return current_feedback
        
        # Get error patterns
        patterns = self.get_error_patterns(agent_name)
        
        # Construct enhancement prompt
        historical_summary = "\n".join([
            f"- Previous error {i+1}: {fb.get('feedback_text', '')[:200]}..."
            for i, fb in enumerate(historical[-3:])
        ])
        
        enhancement_prompt = f"""You are enhancing feedback for {agent_name} in a multi-agent system.

CURRENT FEEDBACK:
{current_feedback}

HISTORICAL ERROR PATTERNS:
Common errors: {', '.join(patterns['common_errors'])}
Trend: {patterns['trend']}
Recent examples:
{historical_summary}

TASK: Enhance the current feedback by:
1. Identifying if this error is recurring (compare with historical patterns)
2. If recurring, emphasize the pattern and suggest specific improvements
3. If new, highlight it as a novel error
4. Keep feedback concise and actionable
5. Do not repeat information already in current feedback

Enhanced feedback (2-3 sentences):"""

        try:
            response = llm_engine.generate(enhancement_prompt)
            enhanced = response.strip()
            logger.info(f"Enhanced feedback for {agent_name} using {len(historical)} historical entries")
            return enhanced
        except Exception as e:
            logger.error(f"Failed to enhance feedback: {e}")
            return current_feedback
    
    def save_to_disk(self):
        """Save loss bank to disk."""
        data = {
            "agent_feedbacks": {
                agent: list(feedbacks)
                for agent, feedbacks in self.agent_feedbacks.items()
            },
            "stats": dict(self.stats)
        }
        
        save_path = self.storage_path / "loss_bank.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Loss bank saved to {save_path}")
    
    def load_from_disk(self):
        """Load loss bank from disk."""
        load_path = self.storage_path / "loss_bank.json"
        if not load_path.exists():
            return
        
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Restore agent feedbacks
            for agent, feedbacks in data.get("agent_feedbacks", {}).items():
                if agent not in self.agent_feedbacks:
                    self.agent_feedbacks[agent] = deque(maxlen=self.max_entries_per_agent)
                self.agent_feedbacks[agent].extend(feedbacks)
            
            # Restore stats
            self.stats = data.get("stats", self.stats)
            
            logger.info(f"Loaded {self.stats['total_stored']} feedbacks from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load loss bank: {e}")


class EpisodicMemory(FeedbackMemory):
    """
    Episodic Memory: Embedding-based retrieval of relevant historical feedbacks.
    
    Design:
    - Stores feedbacks with semantic embeddings
    - Retrieves similar past errors using cosine similarity
    - Can identify analogous problems and their solutions
    - Supports cross-agent learning (similar errors across different agents)
    
    Benefits:
    - Semantic search for relevant past errors
    - Better than keyword matching
    - Can identify subtle pattern similarities
    """
    
    def __init__(self,
                 storage_path: str = "./memory_storage/episodic_memory",
                 embedding_dim: int = 384,
                 max_entries: int = 500,
                 lazy_init: bool = True):
        super().__init__(storage_path)
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.lazy_init = lazy_init
        
        self.memories = []  # List of memory entries
        self.embeddings = None  # Numpy array of embeddings
        self.embedder = None  # Will be initialized on first use if lazy_init=True
        
        # Try to import sentence transformers for embeddings
        if not lazy_init:
            self._init_embedder()
        self.load_from_disk()
    
    def _init_embedder(self):
        """Initialize embedding model."""
        if self.embedder is not None:
            return  # Already initialized
            
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers for episodic memory")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers ({type(e).__name__}). Using hash-based similarity.")
            self.embedder = None
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Lazy initialization on first use
        if self.lazy_init and self.embedder is None:
            self._init_embedder()
            
        if self.embedder:
            return self.embedder.encode(text, convert_to_numpy=True)
        else:
            # Fallback: simple hash-based embedding
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val % (2**32))
            return np.random.randn(self.embedding_dim)
    
    def store(self, feedback: Dict[str, Any]) -> str:
        """
        Store a feedback with its embedding.
        
        Args:
            feedback: Dict containing feedback information
        
        Returns:
            memory_id: str
        """
        # Generate ID
        timestamp = datetime.now().isoformat()
        memory_id = hashlib.md5(
            f"{timestamp}_{feedback.get('feedback_text', '')}".encode()
        ).hexdigest()[:12]
        
        # Create memory entry
        memory = {
            "id": memory_id,
            "timestamp": timestamp,
            **feedback
        }
        
        # Generate embedding
        text_to_embed = feedback.get("feedback_text", "")
        embedding = self._embed_text(text_to_embed)
        
        # Store
        self.memories.append(memory)
        
        # Update embeddings array
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        # Maintain max size
        if len(self.memories) > self.max_entries:
            self.memories.pop(0)
            self.embeddings = self.embeddings[1:]
        
        return memory_id
    
    def retrieve(self, 
                 query: Dict[str, Any], 
                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar memories using semantic search.
        
        Args:
            query: Dict containing:
                - query_text: str (required) - text to find similar memories
                - agent_name: str (optional) - filter by agent
                - min_similarity: float (optional) - minimum cosine similarity
        
        Returns:
            List of relevant memories with similarity scores
        """
        if not self.memories or self.embeddings is None:
            return []
        
        query_text = query.get("query_text", "")
        if not query_text:
            return []
        
        # Generate query embedding
        query_embedding = self._embed_text(query_text)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter and return
        results = []
        min_sim = query.get("min_similarity", 0.3)
        agent_filter = query.get("agent_name")
        
        for idx in top_indices:
            if similarities[idx] < min_sim:
                continue
            
            memory = self.memories[idx].copy()
            
            # Apply agent filter
            if agent_filter and memory.get("agent_name") != agent_filter:
                continue
            
            memory["similarity"] = float(similarities[idx])
            results.append(memory)
        
        return results
    
    def enhance_current_loss(self,
                            current_feedback: str,
                            agent_name: str,
                            llm_engine: Any,
                            top_k: int = 3) -> str:
        """
        Enhance current loss using similar historical episodes.
        
        Args:
            current_feedback: Current feedback text
            agent_name: Agent being evaluated
            llm_engine: LLM engine for synthesis
            top_k: Number of similar episodes to retrieve
        
        Returns:
            Enhanced feedback text
        """
        # Retrieve similar episodes
        similar = self.retrieve(
            query={
                "query_text": current_feedback,
                "agent_name": agent_name,
                "min_similarity": 0.4
            },
            top_k=top_k
        )
        
        if not similar:
            return current_feedback
        
        # Build context from similar episodes
        similar_context = "\n".join([
            f"- Similar error (similarity: {ep['similarity']:.2f}): {ep.get('feedback_text', '')[:150]}..."
            for ep in similar
        ])
        
        enhancement_prompt = f"""You are enhancing feedback for {agent_name} in MiroFlow multi-agent system.

CURRENT FEEDBACK:
{current_feedback}

SIMILAR PAST ERRORS:
{similar_context}

TASK: Enhance the current feedback by:
1. Noting if similar errors occurred before
2. Suggesting solutions based on what worked historically (if available)
3. Highlighting novel aspects of this error
4. Keep it concise (2-3 sentences)

Enhanced feedback:"""

        try:
            response = llm_engine.generate(enhancement_prompt)
            enhanced = response.strip()
            logger.info(f"Enhanced feedback for {agent_name} using {len(similar)} similar episodes")
            return enhanced
        except Exception as e:
            logger.error(f"Failed to enhance with episodic memory: {e}")
            return current_feedback
    
    def save_to_disk(self):
        """Save episodic memory to disk."""
        data = {
            "memories": self.memories,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
        
        save_path = self.storage_path / "episodic_memory.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Episodic memory saved to {save_path}")
    
    def load_from_disk(self):
        """Load episodic memory from disk."""
        load_path = self.storage_path / "episodic_memory.json"
        if not load_path.exists():
            return
        
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.memories = data.get("memories", [])
            embeddings_list = data.get("embeddings")
            if embeddings_list:
                self.embeddings = np.array(embeddings_list)
            
            logger.info(f"Loaded episodic memory with {len(self.memories)} entries")
        except Exception as e:
            logger.error(f"Failed to load episodic memory: {e}")


class MemoryManager:
    """
    Unified manager for memory mechanisms in MiroFlow TextGrad training.
    
    Usage:
        manager = MemoryManager(strategy="loss_bank")
        
        # Store feedback
        manager.store_feedback(agent_name, feedback_text, metadata)
        
        # Enhance current loss
        enhanced = manager.enhance_feedback(current_feedback, agent_name, llm_engine)
    """
    
    def __init__(self, 
                 strategy: str = "loss_bank",
                 storage_path: str = "./memory_storage",
                 **kwargs):
        """
        Initialize memory manager.
        
        Args:
            strategy: Memory strategy ("loss_bank", "episodic", or "hybrid")
            storage_path: Base directory for memory storage
            **kwargs: Additional arguments for specific memory implementations
        """
        self.strategy = strategy
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Track current iteration/epoch
        self.current_iteration = 0
        self.current_epoch = 0
        
        # Initialize memory backend(s)
        self.loss_bank = None
        self.episodic_memory = None
        
        if strategy in ["loss_bank", "hybrid"]:
            self.loss_bank = LossBank(
                storage_path=str(self.storage_path / "loss_bank"),
                max_entries_per_agent=kwargs.get("max_entries_per_agent", 100),
                window_size=kwargs.get("window_size", 50)
            )
        
        if strategy in ["episodic", "hybrid"]:
            self.episodic_memory = EpisodicMemory(
                storage_path=str(self.storage_path / "episodic_memory"),
                embedding_dim=kwargs.get("embedding_dim", 384),
                max_entries=kwargs.get("max_entries", 500),
                lazy_init=kwargs.get("lazy_init", True)
            )
        
        if strategy not in ["loss_bank", "episodic", "hybrid"]:
            raise ValueError(f"Unknown memory strategy: {strategy}. Must be 'loss_bank', 'episodic', or 'hybrid'")
        
        logger.info(f"MemoryManager initialized with strategy: {strategy}")
    
    def store_feedback(self,
                       agent_name: str,
                       feedback_text: str,
                       metadata: Dict[str, Any]) -> None:
        """
        Store feedback for an agent.
        
        Args:
            agent_name: Name of the agent
            feedback_text: Feedback text from loss module
            metadata: Additional metadata (predicted, gold_answer, task_id, etc.)
        """
        feedback_entry = {
            "agent_name": agent_name,
            "feedback_text": feedback_text,
            "iteration": self.current_iteration,
            "epoch": self.current_epoch,
            "metadata": metadata
        }
        
        # Store in active memory backends
        if self.loss_bank:
            self.loss_bank.store(feedback_entry)
        
        if self.episodic_memory:
            self.episodic_memory.store(feedback_entry)
        
        logger.debug(f"Stored feedback for {agent_name} at iteration {self.current_iteration}")
    
    def enhance_feedback(self,
                        current_feedback: str,
                        agent_name: str,
                        llm_engine: Any) -> str:
        """
        Enhance current feedback using historical memory.
        
        Args:
            current_feedback: Current feedback text
            agent_name: Agent being evaluated
            llm_engine: LLM engine for synthesis
        
        Returns:
            Enhanced feedback text
        """
        if self.strategy == "loss_bank":
            if self.loss_bank:
                return self.loss_bank.enhance_current_loss(
                    current_feedback=current_feedback,
                    agent_name=agent_name,
                    llm_engine=llm_engine
                )
        
        elif self.strategy == "episodic":
            if self.episodic_memory:
                return self.episodic_memory.enhance_current_loss(
                    current_feedback=current_feedback,
                    agent_name=agent_name,
                    llm_engine=llm_engine
                )
        
        elif self.strategy == "hybrid":
            # Hybrid: Two-stage enhancement
            # Stage 1: Loss Bank for pattern analysis
            if self.loss_bank:
                enhanced_by_bank = self.loss_bank.enhance_current_loss(
                    current_feedback=current_feedback,
                    agent_name=agent_name,
                    llm_engine=llm_engine
                )
            else:
                enhanced_by_bank = current_feedback
            
            # Stage 2: Episodic Memory for similar case retrieval
            if self.episodic_memory:
                enhanced_final = self.episodic_memory.enhance_current_loss(
                    current_feedback=enhanced_by_bank,
                    agent_name=agent_name,
                    llm_engine=llm_engine
                )
                return enhanced_final
            
            return enhanced_by_bank
        
        return current_feedback
    
    def update_iteration(self, iteration: int, epoch: int):
        """Update current iteration and epoch counters."""
        self.current_iteration = iteration
        self.current_epoch = epoch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "strategy": self.strategy,
            "iteration": self.current_iteration,
            "epoch": self.current_epoch
        }
        
        if self.loss_bank:
            stats["loss_bank"] = {
                "total_stored": self.loss_bank.stats["total_stored"],
                "by_agent": dict(self.loss_bank.stats["by_agent"]),
                "error_patterns": {
                    agent: self.loss_bank.get_error_patterns(agent)
                    for agent in self.loss_bank.agent_feedbacks.keys()
                }
            }
        
        if self.episodic_memory:
            stats["episodic_memory"] = {
                "total_memories": len(self.episodic_memory.memories),
                "has_embeddings": self.episodic_memory.embeddings is not None
            }
        
        return stats
    
    def save_all(self):
        """Save all memory to disk."""
        if self.loss_bank:
            self.loss_bank.save_to_disk()
        
        if self.episodic_memory:
            self.episodic_memory.save_to_disk()
        
        logger.info(f"All memory saved to disk (strategy: {self.strategy})")
    
    def load_all(self):
        """Load all memory from disk."""
        if self.loss_bank:
            self.loss_bank.load_from_disk()
        
        if self.episodic_memory:
            self.episodic_memory.load_from_disk()
        
        logger.info(f"All memory loaded from disk (strategy: {self.strategy})")
