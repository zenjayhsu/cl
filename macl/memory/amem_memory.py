from __future__ import annotations

from config import CSCLRuntimeConfig
from dataclasses import dataclass, field
from typing import Dict, List

from memory.box_manager import BoxStore
from memory.linker import LinkGenerator
from memory.profile_manager import ProfileManager
from schemas.cscl import CSCLNote, ContextWindow, GroupProfile, IndividualProfile, Link, StudentNoteBox, StudentUtterance
from services.retriever import TfidfRetriever


@dataclass
class CSCLAgenticMemory:
    student_ids: List[str]
    runtime_config: CSCLRuntimeConfig = field(default_factory=CSCLRuntimeConfig)
    retriever: TfidfRetriever = field(default_factory=TfidfRetriever)

    def __post_init__(self) -> None:
        self.notes: Dict[str, CSCLNote] = {}
        self.links: List[Link] = []
        self.student_note_index: Dict[str, List[str]] = {student_id: [] for student_id in self.student_ids}
        self.turn_history: List[StudentUtterance] = []
        self.box_store = BoxStore(student_ids=self.student_ids)
        self.profile_manager = ProfileManager(student_ids=self.student_ids, llm_config=self.runtime_config.llm)
        self.link_generator = LinkGenerator(retriever=self.retriever)

    def append_turn(self, utterance: StudentUtterance) -> None:
        self.turn_history.append(utterance)

    def build_context_window(self, utterance: StudentUtterance, max_turns: int | None = None) -> ContextWindow:
        if max_turns is None:
            max_turns = self.runtime_config.context_window.max_turns
        recent_turns = self.turn_history[-max_turns:]
        return ContextWindow(
            focus_student_id=utterance.student_id,
            latest_turn=utterance,
            recent_turns=recent_turns,
            max_turns=max_turns,
        )

    def write_note(self, note: CSCLNote, utterance: StudentUtterance) -> CSCLNote:
        if not (note.cognitive_tag or note.affective_tag or note.social_tag):
            return note
        note = self.box_store.write(note=note)
        self.notes[note.note_id] = note
        self.student_note_index[note.student_id].append(note.note_id)
        self.retriever.add_document(
            doc_id=note.note_id,
            document=self._enhanced_document(note),
            metadata={
                "student_id": note.student_id,
                "timestamp": note.timestamp,
                "box_id": note.box_id,
                "drawer_id": note.drawer_id,
                "cognitive_tag": note.cognitive_tag,
                "affective_tag": note.affective_tag,
                "social_tag": note.social_tag,
            },
        )
        new_links = self.link_generator.generate(new_note=note, utterance=utterance, notes=self.notes, student_note_index=self.student_note_index)
        self.links.extend(new_links)
        self.profile_manager.evolve(note=note, links=new_links)
        return note

    def retrieve_relative_memory(self, query_text: str, student_id: str, k: int = 5) -> List[CSCLNote]:
        hits = self.retriever.search(query=query_text, k=k)
        related_ids: List[str] = []
        for hit in hits:
            if hit.doc_id not in related_ids:
                related_ids.append(hit.doc_id)
            note = self.notes[hit.doc_id]
            for note_id in self.box_store.notes_in_drawer(note.student_id, note.drawer_id):
                if note_id not in related_ids:
                    related_ids.append(note_id)
                    if len(related_ids) >= k * 2:
                        break
            if len(related_ids) >= k * 2:
                break
        return [self.notes[note_id] for note_id in related_ids if note_id in self.notes]

    def individual_profile(self, student_id: str) -> IndividualProfile:
        return self.profile_manager.individual_profiles[student_id]

    def group_profile(self) -> GroupProfile:
        return self.profile_manager.group_profile

    def student_box(self, student_id: str) -> StudentNoteBox:
        return self.box_store.student_box(student_id)

    def refresh_profiles(self, focus_student_id: str) -> tuple[IndividualProfile, GroupProfile]:
        return self.profile_manager.refresh_llm_profiles(
            student_boxes=self.box_store.boxes,
            notes=self.notes,
            focus_student_id=focus_student_id,
        )

    def _enhanced_document(self, note: CSCLNote) -> str:
        summary_bits = [note.content, note.cognitive_tag, note.affective_tag, note.social_tag, note.drawer_id, note.box_id]
        return " ".join(item for item in summary_bits if item)
