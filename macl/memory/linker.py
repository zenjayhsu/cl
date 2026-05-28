from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from schemas.cscl import CSCLNote, Link, StudentUtterance
from services.retriever import TfidfRetriever


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _relation_for_social_tag(social_tag: str) -> str:
    if social_tag in {"consensus_building", "回应同伴", "促进讨论", "协作支持"}:
        return "support"
    if social_tag in {"conflict", "Conflict", "冲突对立", "压制同伴"}:
        return "conflict"
    if social_tag in {"silent", "Silent", "沉默观察", "忽视同伴"}:
        return "ignored"
    if social_tag in {"parallel_play", "平行发言", "不平等参与"}:
        return "parallel"
    return "cause"


@dataclass
class LinkGenerator:
    retriever: TfidfRetriever

    def generate(self, new_note: CSCLNote, utterance: StudentUtterance, notes: Dict[str, CSCLNote], student_note_index: Dict[str, List[str]]) -> List[Link]:
        links: List[Link] = []
        links.extend(self._temporal_links(new_note, notes, student_note_index))
        links.extend(self._cross_student_links(new_note, utterance, notes, student_note_index))
        links.extend(self._group_links(new_note))
        return links

    def _temporal_links(self, new_note: CSCLNote, notes: Dict[str, CSCLNote], student_note_index: Dict[str, List[str]]) -> List[Link]:
        history = [note_id for note_id in student_note_index.get(new_note.student_id, []) if note_id != new_note.note_id][-3:]
        links: List[Link] = []
        for note_id in history:
            previous = notes[note_id]
            gap_minutes = abs((_parse_ts(new_note.timestamp) - _parse_ts(previous.timestamp)).total_seconds()) / 60.0
            if gap_minutes <= 30:
                weight = max(0.1, 1.0 - min(gap_minutes / 30.0, 0.9))
                links.append(Link(source_id=previous.note_id, target_id=new_note.note_id, relation="cause", weight=round(weight, 3)))
        return links

    def _cross_student_links(self, new_note: CSCLNote, utterance: StudentUtterance, notes: Dict[str, CSCLNote], student_note_index: Dict[str, List[str]]) -> List[Link]:
        if not new_note.social_tag:
            return []
        relation = _relation_for_social_tag(new_note.social_tag)
        links: List[Link] = []
        mentioned_students = list(utterance.mentions)
        interaction_target = new_note.diagnosis.get("interaction_target") if isinstance(new_note.diagnosis, dict) else None
        if isinstance(interaction_target, str) and interaction_target and interaction_target not in mentioned_students:
            mentioned_students.append(interaction_target)
        for mentioned_student in mentioned_students:
            if mentioned_student == new_note.student_id:
                continue
            for peer_note_id in reversed(student_note_index.get(mentioned_student, [])):
                peer_note = notes[peer_note_id]
                if peer_note.source_turn_id == new_note.source_turn_id:
                    continue
                weight = 0.85
                if relation == "parallel":
                    weight = 0.55
                elif relation == "ignored":
                    weight = 0.7
                elif relation == "cause":
                    weight = 0.65
                links.append(Link(source_id=new_note.note_id, target_id=peer_note.note_id, relation=relation, weight=weight))
                break
        return links

    def _group_links(self, new_note: CSCLNote) -> List[Link]:
        if "group:shared" not in new_note.box_ids:
            return []
        relation = _relation_for_social_tag(new_note.social_tag)
        return [Link(source_id=new_note.note_id, target_id="group_profile", relation=relation, weight=0.9)]
