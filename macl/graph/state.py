from __future__ import annotations

from typing import Annotated, Dict, List, TypedDict
import operator

from memory.amem_memory import CSCLAgenticMemory
from schemas.cscl import CSCLNote, ContextWindow, Decision, GroupProfile, IndividualProfile, InterventionMessage, StudentUtterance


class CSCLGraphState(TypedDict, total=False):
    memory_system: CSCLAgenticMemory
    current_turn: StudentUtterance
    context_window: ContextWindow
    sensor_notes: Annotated[List[CSCLNote], operator.add]
    written_notes: Annotated[List[CSCLNote], operator.add]
    retrieved_notes: List[CSCLNote]
    triggered_notes: List[Dict[str, object]]
    individual_profile: IndividualProfile
    group_profile: GroupProfile
    decision: Decision
    intervention: InterventionMessage
    debug: Dict[str, object]
