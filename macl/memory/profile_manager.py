from __future__ import annotations

from config import LLMRuntimeConfig
from dataclasses import dataclass, field
from typing import Dict, List

from schemas.cscl import CSCLNote, GroupProfile, IndividualProfile, Link, StudentNoteBox
from services.profile_llm import ProfileSynthesizer, build_llm_controller
from services.urgency_calibrator import DEFAULT_TRIGGER_THRESHOLD

AFFECTIVE_POLARITY = {
    "好奇": 2,
    "兴趣": 2,
    "惊讶": 0,
    "困惑": -1,
    "焦虑": -2,
    "挫败": -3,
    "无聊": -3,
    "愉悦": 2,
    "engagement": 2,
    "confusion": -1,
    "frustration": -2,
    "disengagement": -3,
    "Engagement": 2,
    "Confusion": -1,
    "Frustration": -2,
    "Boredom": -3,
}

COGNITIVE_OK = "无明显问题"
SOCIAL_RISK_TAGS = {"沉默观察", "冲突对立", "压制同伴", "忽视同伴", "不平等参与"}
SOCIAL_SUPPORT_TAGS = {"回应同伴", "促进讨论", "协作支持", "积极参与", "无明显问题"}


def _social_relation(tag: str) -> str:
    if tag in {"回应同伴", "促进讨论", "协作支持"}:
        return "support"
    if tag in {"冲突对立", "压制同伴"}:
        return "conflict"
    if tag in {"沉默观察", "忽视同伴"}:
        return "ignored"
    if tag in {"平行发言", "不平等参与"}:
        return "parallel"
    return "participation"


@dataclass
class ProfileManager:
    student_ids: List[str]
    llm_config: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
    synthesizer: ProfileSynthesizer = field(init=False)
    individual_profiles: Dict[str, IndividualProfile] = field(default_factory=dict)
    group_profile: GroupProfile = field(default_factory=GroupProfile)

    def __post_init__(self) -> None:
        self.synthesizer = ProfileSynthesizer(llm=build_llm_controller(self.llm_config))
        for student_id in self.student_ids:
            self.individual_profiles.setdefault(student_id, IndividualProfile(student_id=student_id))

    def evolve(self, note: CSCLNote, links: List[Link]) -> None:
        profile = self.individual_profiles[note.student_id]
        if note.speaker_name and not profile.name:
            profile.name = note.speaker_name
        profile.last_updated = note.timestamp
        diagnosis = note.diagnosis or {}
        theory_state = diagnosis.get("theory_state", {})
        anomaly_detection = diagnosis.get("anomaly_detection", {})
        if not isinstance(theory_state, dict):
            theory_state = {}
        if not isinstance(anomaly_detection, dict):
            anomaly_detection = {}

        if note.note_id not in profile.note_ids:
            profile.note_ids.append(note.note_id)

        if note.cognitive_tag:
            profile.cognitive_profile[note.cognitive_tag] = profile.cognitive_profile.get(note.cognitive_tag, 0) + 1
            profile.bloom_trajectory.append(
                {
                    "timestamp": note.timestamp,
                    "turn_id": note.source_turn_id,
                    "bloom_level": diagnosis.get("bloom_level", theory_state.get("bloom_level", "")),
                    "confidence": diagnosis.get("confidence", ""),
                    "note_id": note.note_id,
                }
            )
            profile.bloom_trajectory = profile.bloom_trajectory[-30:]

            cognitive_category = str(diagnosis.get("category", note.cognitive_tag))
            has_cognitive_issue = (
                cognitive_category != COGNITIVE_OK
                or bool(anomaly_detection.get("has_error"))
                or bool(diagnosis.get("has_error"))
            )
            if has_cognitive_issue:
                anomaly_category = anomaly_detection.get(
                    "error_category",
                    cognitive_category or diagnosis.get("error_category", "Unknown"),
                )
                profile.anomaly_counts[anomaly_category] = profile.anomaly_counts.get(anomaly_category, 0) + 1
                profile.cognitive_anomaly_trajectory.append(
                    {
                        "timestamp": note.timestamp,
                        "turn_id": note.source_turn_id,
                        "category": anomaly_category,
                        "specific_evidence": anomaly_detection.get("specific_evidence", diagnosis.get("specific_evidence", "")),
                        "urgency_score": diagnosis.get("urgency_score", 0.0),
                        "note_id": note.note_id,
                    }
                )
                profile.cognitive_anomaly_trajectory = profile.cognitive_anomaly_trajectory[-20:]

        if note.affective_tag:
            polarity = AFFECTIVE_POLARITY.get(note.affective_tag, 0)
            profile.affective_trend.append({"timestamp": note.timestamp, "tag": note.affective_tag, "polarity": polarity})
            profile.affective_trend = profile.affective_trend[-20:]
            profile.affective_state_trajectory.append(
                {
                    "timestamp": note.timestamp,
                    "turn_id": note.source_turn_id,
                    "affective_state": diagnosis.get("category", theory_state.get("affective_state", note.affective_tag)),
                    "valence": diagnosis.get("valence", theory_state.get("valence", "")),
                    "is_critical": bool(
                        anomaly_detection.get("is_critical")
                        or diagnosis.get("is_critical")
                        or float(diagnosis.get("urgency_score", 0.0) or 0.0) >= DEFAULT_TRIGGER_THRESHOLD
                    ),
                    "emotional_trigger": anomaly_detection.get("emotional_trigger", diagnosis.get("specific_evidence", "")),
                    "behavioral_manifestation": anomaly_detection.get("behavioral_manifestation", diagnosis.get("specific_evidence", "")),
                    "urgency_score": diagnosis.get("urgency_score", 0.0),
                    "note_id": note.note_id,
                }
            )
            profile.affective_state_trajectory = profile.affective_state_trajectory[-30:]

        if note.social_tag:
            profile.social_role[note.social_tag] = profile.social_role.get(note.social_tag, 0) + 1
            profile.social_participation_trajectory.append(
                {
                    "timestamp": note.timestamp,
                    "turn_id": note.source_turn_id,
                    "social_mode": diagnosis.get("category", theory_state.get("social_mode", note.social_tag)),
                    "role_behavior": theory_state.get("role_behavior", diagnosis.get("role_behavior", "")),
                    "needs_moderation": bool(
                        anomaly_detection.get("needs_moderation")
                        or diagnosis.get("needs_moderation")
                        or note.social_tag in SOCIAL_RISK_TAGS
                        or float(diagnosis.get("urgency_score", 0.0) or 0.0) >= DEFAULT_TRIGGER_THRESHOLD
                    ),
                    "interaction_flaw": anomaly_detection.get("interaction_flaw", diagnosis.get("specific_evidence", "")),
                    "target_student": diagnosis.get(
                        "interaction_target",
                        anomaly_detection.get("target_student", diagnosis.get("target_student", "")),
                    ),
                    "urgency_score": diagnosis.get("urgency_score", 0.0),
                    "note_id": note.note_id,
                }
            )
            profile.social_participation_trajectory = profile.social_participation_trajectory[-30:]

        self._evolve_group_profile(note=note, links=links, theory_state=theory_state, anomaly_detection=anomaly_detection, diagnosis=diagnosis)

    def refresh_llm_profiles(self, student_boxes: Dict[str, StudentNoteBox], notes: Dict[str, CSCLNote], focus_student_id: str) -> tuple[IndividualProfile, GroupProfile]:
        focus_profile = self.individual_profiles[focus_student_id]
        latest_ids = focus_profile.note_ids[-6:]
        latest_notes = [notes[note_id].to_dict() for note_id in latest_ids if note_id in notes]
        updated_individual = self.synthesizer.summarize_individual(
            profile=focus_profile,
            box=student_boxes[focus_student_id],
            latest_notes=latest_notes,
        )
        member_profiles = list(self.individual_profiles.values())
        recent_note_ids = self.group_profile.note_ids[-12:]
        recent_notes = [notes[note_id].to_dict() for note_id in recent_note_ids if note_id in notes]
        updated_group = self.synthesizer.summarize_group(
            group_profile=self.group_profile,
            member_profiles=member_profiles,
            recent_notes=recent_notes,
        )
        self.individual_profiles[focus_student_id] = updated_individual
        self.group_profile = updated_group
        return updated_individual, updated_group

    def _evolve_group_profile(self, note: CSCLNote, links: List[Link], theory_state: Dict[str, object], anomaly_detection: Dict[str, object], diagnosis: Dict[str, object]) -> None:
        if note.note_id not in self.group_profile.note_ids:
            self.group_profile.note_ids.append(note.note_id)
        self.group_profile.last_updated = note.timestamp

        for link in links:
            self.group_profile.interaction_patterns[link.relation] = self.group_profile.interaction_patterns.get(link.relation, 0) + 1
        if note.social_tag:
            self.group_profile.interaction_patterns[note.social_tag] = self.group_profile.interaction_patterns.get(note.social_tag, 0) + 1
            relation = _social_relation(note.social_tag)
            self.group_profile.interaction_patterns[relation] = self.group_profile.interaction_patterns.get(relation, 0) + 1

        self.group_profile.conflict_level = self._compute_conflict_level()
        self.group_profile.collaboration_mode = self._infer_mode()

        self.group_profile.collaboration_mode_trajectory.append(
            {
                "timestamp": note.timestamp,
                "turn_id": note.source_turn_id,
                "collaboration_mode": self.group_profile.collaboration_mode,
                "conflict_level": self.group_profile.conflict_level,
                "note_id": note.note_id,
            }
        )
        self.group_profile.collaboration_mode_trajectory = self.group_profile.collaboration_mode_trajectory[-40:]

        if (
            anomaly_detection.get("needs_moderation")
            or diagnosis.get("needs_moderation")
            or note.social_tag in SOCIAL_RISK_TAGS
            or float(diagnosis.get("urgency_score", 0.0) or 0.0) >= DEFAULT_TRIGGER_THRESHOLD
        ):
            self.group_profile.conflict_event_trajectory.append(
                {
                    "timestamp": note.timestamp,
                    "turn_id": note.source_turn_id,
                    "social_mode": diagnosis.get("category", theory_state.get("social_mode", note.social_tag)),
                    "interaction_flaw": anomaly_detection.get("interaction_flaw", diagnosis.get("specific_evidence", "")),
                    "target_student": diagnosis.get(
                        "interaction_target",
                        anomaly_detection.get("target_student", diagnosis.get("target_student", "")),
                    ),
                    "urgency_score": diagnosis.get("urgency_score", 0.0),
                    "note_id": note.note_id,
                }
            )
            self.group_profile.conflict_event_trajectory = self.group_profile.conflict_event_trajectory[-25:]

        participation_counts = {
            student_id: len(self.individual_profiles[student_id].note_ids)
            for student_id in self.student_ids
        }
        max_count = max(participation_counts.values()) if participation_counts else 0
        min_count = min(participation_counts.values()) if participation_counts else 0
        imbalance = max_count - min_count
        self.group_profile.participation_balance_trajectory.append(
            {
                "timestamp": note.timestamp,
                "turn_id": note.source_turn_id,
                "counts": participation_counts,
                "imbalance": imbalance,
                "note_id": note.note_id,
            }
        )
        self.group_profile.participation_balance_trajectory = self.group_profile.participation_balance_trajectory[-40:]

        risk_flags: List[str] = []
        if self.group_profile.conflict_level >= 4:
            risk_flags.append("high_conflict")
        if imbalance >= 3:
            risk_flags.append("participation_imbalance")
        if self.group_profile.collaboration_mode == "parallel_play":
            risk_flags.append("parallel_play_stall")
        if note.social_tag in {"沉默观察", "忽视同伴"}:
            risk_flags.append("marginalization_risk")

        self.group_profile.group_risk_trajectory.append(
            {
                "timestamp": note.timestamp,
                "turn_id": note.source_turn_id,
                "risk_flags": risk_flags,
                "conflict_level": self.group_profile.conflict_level,
                "collaboration_mode": self.group_profile.collaboration_mode,
                "note_id": note.note_id,
            }
        )
        self.group_profile.group_risk_trajectory = self.group_profile.group_risk_trajectory[-40:]

    def _compute_conflict_level(self) -> int:
        base = 0
        base += self.group_profile.interaction_patterns.get("conflict", 0)
        base += self.group_profile.interaction_patterns.get("冲突对立", 0)
        base += self.group_profile.interaction_patterns.get("压制同伴", 0)
        base += self.group_profile.interaction_patterns.get("silent", 0)
        base += self.group_profile.interaction_patterns.get("沉默观察", 0)
        base += self.group_profile.interaction_patterns.get("ignored", 0)
        base += self.group_profile.interaction_patterns.get("忽视同伴", 0)
        base += self.group_profile.interaction_patterns.get("不平等参与", 0)
        base -= self.group_profile.interaction_patterns.get("support", 0)
        base -= self.group_profile.interaction_patterns.get("consensus_building", 0)
        for tag in SOCIAL_SUPPORT_TAGS:
            base -= self.group_profile.interaction_patterns.get(tag, 0)
        return max(0, base)

    def _infer_mode(self) -> str:
        support = self.group_profile.interaction_patterns.get("support", 0)
        consensus = self.group_profile.interaction_patterns.get("consensus_building", 0) + support
        parallel = (
            self.group_profile.interaction_patterns.get("parallel_play", 0)
            + self.group_profile.interaction_patterns.get("parallel", 0)
            + self.group_profile.interaction_patterns.get("平行发言", 0)
            + self.group_profile.interaction_patterns.get("不平等参与", 0)
        )
        silent = (
            self.group_profile.interaction_patterns.get("silent", 0)
            + self.group_profile.interaction_patterns.get("ignored", 0)
            + self.group_profile.interaction_patterns.get("沉默观察", 0)
            + self.group_profile.interaction_patterns.get("忽视同伴", 0)
        )
        conflict = self.group_profile.conflict_level
        if conflict >= 4:
            return "conflictual"
        if consensus >= 2:
            return "consensus_building"
        if parallel >= 2 and consensus == 0:
            return "parallel_play"
        if silent >= 1 and consensus == 0:
            return "fragile_participation"
        return "forming"
