from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class StudentUtterance:
    turn_id: str
    student_id: str
    text: str
    timestamp: str
    speaker_name: str = ""
    mentions: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ContextWindow:
    focus_student_id: str
    latest_turn: StudentUtterance
    recent_turns: List[StudentUtterance] = field(default_factory=list)
    max_turns: int = 6

    def to_dict(self) -> Dict[str, object]:
        return {
            "focus_student_id": self.focus_student_id,
            "latest_turn": self.latest_turn.to_dict(),
            "recent_turns": [turn.to_dict() for turn in self.recent_turns],
            "max_turns": self.max_turns,
        }


@dataclass
class CSCLNote:
    note_id: str
    student_id: str
    content: str
    timestamp: str
    speaker_name: str = ""
    cognitive_tag: str = ""
    affective_tag: str = ""
    social_tag: str = ""
    diagnosis: Dict[str, object] = field(default_factory=dict)
    context_window: Dict[str, object] = field(default_factory=dict)
    drawer_id: str = ""
    box_id: str = ""
    box_ids: List[str] = field(default_factory=list)
    box_count: int = 0
    source_turn_id: str = ""
    urgency: float = 0.0
    sensor_name: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DrawerSnapshot:
    drawer_id: str
    note_ids: List[str] = field(default_factory=list)
    summary: str = ""
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class StudentNoteBox:
    student_id: str
    drawers: Dict[str, DrawerSnapshot] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "student_id": self.student_id,
            "drawers": {drawer_id: drawer.to_dict() for drawer_id, drawer in self.drawers.items()},
            "summary": self.summary,
        }


@dataclass
class Link:
    source_id: str
    target_id: str
    relation: str
    weight: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class IndividualProfile:
    student_id: str
    name: str = ""
    knowledge_state: str = ""
    cognitive_pattern: str = ""
    affective_pattern: str = ""
    social_pattern: str = ""
    higher_order_ability: str = ""
    main_difficulties: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommended_support: List[str] = field(default_factory=list)
    last_updated: str = ""
    cognitive_profile: Dict[str, int] = field(default_factory=dict)
    affective_trend: List[Dict[str, object]] = field(default_factory=list)
    social_role: Dict[str, int] = field(default_factory=dict)
    note_ids: List[str] = field(default_factory=list)
    bloom_trajectory: List[Dict[str, object]] = field(default_factory=list)
    cognitive_anomaly_trajectory: List[Dict[str, object]] = field(default_factory=list)
    affective_state_trajectory: List[Dict[str, object]] = field(default_factory=list)
    social_participation_trajectory: List[Dict[str, object]] = field(default_factory=list)
    anomaly_counts: Dict[str, int] = field(default_factory=dict)
    drawer_summaries: Dict[str, str] = field(default_factory=dict)
    llm_summary: str = ""
    llm_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class GroupProfile:
    group_id: str = "g1"
    overall_knowledge_state: str = ""
    common_misconceptions: List[str] = field(default_factory=list)
    discussion_quality: str = ""
    collaboration_pattern: str = ""
    affective_climate: str = ""
    participation_balance: str = ""
    higher_order_ability_level: str = ""
    main_group_risks: List[str] = field(default_factory=list)
    recommended_group_support: List[str] = field(default_factory=list)
    last_updated: str = ""
    interaction_patterns: Dict[str, int] = field(default_factory=dict)
    conflict_level: int = 0
    collaboration_mode: str = "unformed"
    note_ids: List[str] = field(default_factory=list)
    collaboration_mode_trajectory: List[Dict[str, object]] = field(default_factory=list)
    conflict_event_trajectory: List[Dict[str, object]] = field(default_factory=list)
    participation_balance_trajectory: List[Dict[str, object]] = field(default_factory=list)
    group_risk_trajectory: List[Dict[str, object]] = field(default_factory=list)
    llm_summary: str = ""
    llm_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class Decision:
    type: str
    action: str
    reason: str
    intervention_needed: bool = False
    target: str = "none"
    target_student_id: str | None = None
    intervention_type: str = "none"
    based_on_notes: List[str] = field(default_factory=list)
    based_on_profile: str = ""
    intervention_content: str = ""
    expected_effect: str = ""
    response_text: str = ""
    target_scope: str = ""
    used_profiles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class InterventionMessage:
    turn_id: str
    target_scope: str
    content: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
