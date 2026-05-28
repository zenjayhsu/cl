from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from schemas.cscl import CSCLNote, DrawerSnapshot, StudentNoteBox

DRAWER_IDS = ("cognitive_drawer", "affective_drawer", "social_drawer")


@dataclass
class BoxStore:
    student_ids: List[str]
    boxes: Dict[str, StudentNoteBox] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for student_id in self.student_ids:
            self.boxes.setdefault(student_id, self._new_box(student_id))

    def _new_box(self, student_id: str) -> StudentNoteBox:
        return StudentNoteBox(
            student_id=student_id,
            drawers={drawer_id: DrawerSnapshot(drawer_id=drawer_id) for drawer_id in DRAWER_IDS},
        )

    def write(self, note: CSCLNote) -> CSCLNote:
        box = self.boxes.setdefault(note.student_id, self._new_box(note.student_id))
        drawer_id = self._resolve_drawer_id(note)
        drawer = box.drawers[drawer_id]
        drawer.note_ids.append(note.note_id)
        note.drawer_id = drawer_id
        note.box_id = note.student_id
        note.box_ids = [note.student_id, drawer_id]
        note.box_count = 1
        return note

    def student_box(self, student_id: str) -> StudentNoteBox:
        return self.boxes.setdefault(student_id, self._new_box(student_id))

    def notes_in_box(self, student_id: str) -> List[str]:
        box = self.student_box(student_id)
        note_ids: List[str] = []
        for drawer in box.drawers.values():
            note_ids.extend(drawer.note_ids)
        return note_ids

    def notes_in_drawer(self, student_id: str, drawer_id: str) -> List[str]:
        return list(self.student_box(student_id).drawers[drawer_id].note_ids)

    def _resolve_drawer_id(self, note: CSCLNote) -> str:
        if note.cognitive_tag:
            return "cognitive_drawer"
        if note.affective_tag:
            return "affective_drawer"
        return "social_drawer"
