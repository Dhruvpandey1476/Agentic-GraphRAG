"""
Agent harness: owns state, tool registry, evidence accumulation and
stopping criteria for the Agentic GraphRAG pipeline. The orchestrator
(orchestrator.py) drives this harness one step at a time.
"""
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


@dataclass
class EvidenceItem:
    step: int
    source_agent: str
    kind: str          # "entity" | "chunk" | "relation" | "note"
    ref_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    step: int
    action: str                 # which specialized agent was invoked
    reason: str                 # orchestrator's stated reason for picking it
    input_args: Dict[str, Any]
    output_summary: str
    new_evidence_count: int
    strategy_changed: bool = False
    tokens: int = 0
    latency_s: float = 0.0


class AgentState:
    """Everything the orchestrator needs to decide the next action."""

    def __init__(self, question: str, max_steps: int = None, max_tokens: int = None):
        self.question = question
        self.max_steps = max_steps or config.MAX_AGENT_STEPS
        self.max_tokens = max_tokens or config.MAX_TOKENS_PER_RUN
        self.step_count = 0
        self.total_tokens = 0
        self.evidence: List[EvidenceItem] = []
        self.trace: List[TraceStep] = []
        self.linked_entities: Dict[str, dict] = {}   # entity_id -> data
        self.visited_entities: set = set()
        self.stopped_reason: Optional[str] = None
        self.last_strategy: Optional[str] = None

    def add_evidence(self, items: List[EvidenceItem]):
        self.evidence.extend(items)
        return len(items)

    def log_step(self, action, reason, input_args, output_summary,
                 new_evidence_count, tokens=0, latency_s=0.0):
        strategy_changed = self.last_strategy is not None and self.last_strategy != action
        self.last_strategy = action
        self.step_count += 1
        self.total_tokens += tokens
        self.trace.append(TraceStep(
            step=self.step_count, action=action, reason=reason,
            input_args=input_args, output_summary=output_summary,
            new_evidence_count=new_evidence_count,
            strategy_changed=strategy_changed, tokens=tokens, latency_s=latency_s,
        ))

    def should_stop(self) -> Optional[str]:
        if self.step_count >= self.max_steps:
            return f"reached max_steps ({self.max_steps})"
        if self.total_tokens >= self.max_tokens:
            return f"reached max_tokens ({self.max_tokens})"
        return None

    def evidence_text_block(self) -> str:
        return "\n".join(
            f"[{e.kind}:{e.ref_id}] {e.content[:400]}" for e in self.evidence
        )

    def to_dict(self):
        return {
            "question": self.question,
            "steps_taken": self.step_count,
            "total_tokens": self.total_tokens,
            "stopped_reason": self.stopped_reason,
            "num_evidence_items": len(self.evidence),
            "trace": [
                {
                    "step": t.step, "action": t.action, "reason": t.reason,
                    "input_args": t.input_args, "output_summary": t.output_summary,
                    "new_evidence_count": t.new_evidence_count,
                    "strategy_changed": t.strategy_changed,
                    "tokens": t.tokens, "latency_s": t.latency_s,
                }
                for t in self.trace
            ],
        }
