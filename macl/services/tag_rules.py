from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import re


MENTION_PATTERN = re.compile(r"\b(?:s|student)([1-4])\b|@([A-Za-z0-9_\-]+)", re.IGNORECASE)


def detect_mentions(text: str, fallback_students: Sequence[str]) -> List[str]:
    mentions: List[str] = []
    for match in MENTION_PATTERN.finditer(text):
        student_num, handle = match.groups()
        if student_num:
            mentions.append(f"s{student_num}")
        elif handle:
            mentions.append(handle.lower())
    for student in fallback_students:
        if student.lower() in text.lower() and student not in mentions:
            mentions.append(student)
    deduped: List[str] = []
    for item in mentions:
        if item not in deduped:
            deduped.append(item)
    return deduped


def compute_urgency(
    cognitive_tag: str = "",
    affective_tag: str = "",
    social_tag: str = "",
    diagnosis: Optional[Dict[str, object]] = None,
) -> float:
    if diagnosis and "urgency_score" in diagnosis:
        return max(0.0, min(float(diagnosis.get("urgency_score", 0.0) or 0.0), 1.0))
    return 0.0
