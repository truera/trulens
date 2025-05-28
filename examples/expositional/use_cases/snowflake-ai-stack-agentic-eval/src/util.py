# Custom State class with specific keys
from typing import Any, Dict, List, Optional
from langgraph.graph import MessagesState


class State(MessagesState):
    execution_trace: Optional[Dict[int, List[Dict[str, Any]]]]
    user_query: Optional[str]
    current_step: int
    last_reason: Optional[str]

# Helper method to append entries to Execution Trace
def append_to_step_trace(state, step: int, new_entry: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    trace = state.get("execution_trace", {}) or {}
    if step not in trace:
        trace[step] = []
    trace[step].append(new_entry)
    return trace
