from __future__ import annotations

from config import CSCLRuntimeConfig, load_runtime_config_from_env
from typing import Dict, List

from langgraph.graph import END, START, StateGraph

from agents.meta_agent import MetaAgent
from agents.sensor_agents import AffectiveSensor, CognitiveSensor, SocialSensor
from graph.state import CSCLGraphState
from memory.amem_memory import CSCLAgenticMemory
from schemas.cscl import CSCLNote, InterventionMessage, StudentUtterance
from services.runtime_logging import get_logger
from services.urgency_calibrator import DEFAULT_TRIGGER_THRESHOLD, UrgencyCalibrator


logger = get_logger("workflow")


def _is_meta_trigger(diagnosis: Dict[str, object]) -> bool:
    return float(diagnosis.get("urgency_score", 0.0) or 0.0) >= DEFAULT_TRIGGER_THRESHOLD


class CSCLWorkflow:
    def __init__(self, student_ids: List[str], runtime_config: CSCLRuntimeConfig | None = None) -> None:
        self.runtime_config = runtime_config or load_runtime_config_from_env()
        self.memory = CSCLAgenticMemory(student_ids=student_ids, runtime_config=self.runtime_config)
        self.cognitive_sensor = CognitiveSensor(llm_config=self.runtime_config.llm)
        self.affective_sensor = AffectiveSensor(llm_config=self.runtime_config.llm)
        self.social_sensor = SocialSensor(llm_config=self.runtime_config.llm)
        self.urgency_calibrator = UrgencyCalibrator(trigger_threshold=DEFAULT_TRIGGER_THRESHOLD)
        self.meta_agent = MetaAgent(llm_config=self.runtime_config.llm)
        self.graph = self._build().compile()

    def _build(self):
        graph = StateGraph(CSCLGraphState)
        graph.add_node("prepare_context", self._context_node)
        graph.add_node("cognitive_sensor", self._cognitive_node)
        graph.add_node("affective_sensor", self._affective_node)
        graph.add_node("social_sensor", self._social_node)
        graph.add_node("memory_update", self._memory_node)
        graph.add_node("profile_update", self._profile_node)
        graph.add_node("meta_decision", self._meta_node)
        graph.add_edge(START, "prepare_context")
        graph.add_edge("prepare_context", "cognitive_sensor")
        graph.add_edge("prepare_context", "affective_sensor")
        graph.add_edge("prepare_context", "social_sensor")
        graph.add_edge("cognitive_sensor", "memory_update")
        graph.add_edge("affective_sensor", "memory_update")
        graph.add_edge("social_sensor", "memory_update")
        graph.add_edge("memory_update", "profile_update")
        graph.add_edge("profile_update", "meta_decision")
        graph.add_edge("meta_decision", END)
        return graph

    def run_turn(self, utterance: StudentUtterance) -> Dict[str, object]:
        logger.info("Workflow turn start | turn_id=%s | student=%s", utterance.turn_id, utterance.student_id)
        state: CSCLGraphState = {
            "memory_system": self.memory,
            "current_turn": utterance,
            "sensor_notes": [],
            "written_notes": [],
            "retrieved_notes": [],
            "debug": {},
        }
        result = self.graph.invoke(state)
        logger.info(
            "Workflow turn end | turn_id=%s | written_notes=%s | retrieved_notes=%s",
            utterance.turn_id,
            len(result.get("written_notes", [])),
            len(result.get("retrieved_notes", [])),
        )
        return result

    def _context_node(self, state: CSCLGraphState) -> Dict[str, object]:
        utterance = state["current_turn"]
        memory = state["memory_system"]
        logger.info("Node start | prepare_context | turn_id=%s", utterance.turn_id)
        context_window = memory.build_context_window(utterance)
        memory.append_turn(utterance)
        logger.info("Node end | prepare_context | turn_id=%s | recent_turns=%s", utterance.turn_id, len(context_window.recent_turns))
        return {"context_window": context_window}

    def _cognitive_node(self, state: CSCLGraphState) -> Dict[str, object]:
        utterance = state["current_turn"]
        logger.info("Node start | cognitive_sensor | turn_id=%s", utterance.turn_id)
        note = self.cognitive_sensor.observe(utterance, state["context_window"])
        logger.info("Node end | cognitive_sensor | turn_id=%s | tag=%s", utterance.turn_id, note.cognitive_tag or "none")
        return {"sensor_notes": [note]}

    def _affective_node(self, state: CSCLGraphState) -> Dict[str, object]:
        utterance = state["current_turn"]
        logger.info("Node start | affective_sensor | turn_id=%s", utterance.turn_id)
        note = self.affective_sensor.observe(utterance, state["context_window"])
        logger.info("Node end | affective_sensor | turn_id=%s | tag=%s", utterance.turn_id, note.affective_tag or "none")
        return {"sensor_notes": [note]}

    def _social_node(self, state: CSCLGraphState) -> Dict[str, object]:
        utterance = state["current_turn"]
        logger.info("Node start | social_sensor | turn_id=%s", utterance.turn_id)
        note = self.social_sensor.observe(utterance, state["context_window"])
        logger.info("Node end | social_sensor | turn_id=%s | tag=%s", utterance.turn_id, note.social_tag or "none")
        return {"sensor_notes": [note]}

    def _memory_node(self, state: CSCLGraphState) -> Dict[str, object]:
        utterance = state["current_turn"]
        memory = state["memory_system"]
        logger.info("Node start | memory_update | turn_id=%s | sensor_notes=%s", utterance.turn_id, len(state.get("sensor_notes", [])))
        written: List[CSCLNote] = []
        triggered: List[Dict[str, object]] = []
        # Parallel sensor execution can yield notes in completion order, so keep
        # downstream processing deterministic.
        sensor_notes = sorted(state.get("sensor_notes", []), key=lambda note: note.sensor_name)
        for note in sensor_notes:
            note = self.urgency_calibrator.calibrate_note(
                note=note,
                context_window=state["context_window"],
                individual_profile=memory.individual_profile(utterance.student_id),
                group_profile=memory.group_profile(),
            )
            written_note = memory.write_note(note=note, utterance=utterance)
            written.append(written_note)
            if _is_meta_trigger(written_note.diagnosis or {}):
                triggered.append(written_note.to_dict())
        retrieved = memory.retrieve_relative_memory(query_text=utterance.text, student_id=utterance.student_id, k=5)
        logger.info(
            "Node end | memory_update | turn_id=%s | written=%s | triggered=%s | retrieved=%s",
            utterance.turn_id,
            len(written),
            len(triggered),
            len(retrieved),
        )
        return {"written_notes": written, "retrieved_notes": retrieved, "triggered_notes": triggered, "debug": {"triggered_notes": triggered}}

    def _profile_node(self, state: CSCLGraphState) -> Dict[str, object]:
        utterance = state["current_turn"]
        memory = state["memory_system"]
        logger.info("Node start | profile_update | turn_id=%s", utterance.turn_id)
        individual_profile, group_profile = memory.refresh_profiles(utterance.student_id)
        logger.info("Node end | profile_update | turn_id=%s | individual=%s", utterance.turn_id, individual_profile.student_id)
        return {
            "individual_profile": individual_profile,
            "group_profile": group_profile,
        }

    def _meta_node(self, state: CSCLGraphState) -> Dict[str, object]:
        utterance = state["current_turn"]
        logger.info("Node start | meta_decision | turn_id=%s", utterance.turn_id)
        decision = self.meta_agent.decide(
            utterance=utterance,
            context_window=state["context_window"],
            individual_profile=state["individual_profile"],
            group_profile=state["group_profile"],
            triggered_notes=state.get("triggered_notes", []),
        )
        intervention_content = decision.intervention_content or decision.response_text
        target_scope = "none"
        if decision.intervention_needed:
            if decision.target == "individual" and decision.target_student_id:
                target_scope = decision.target_student_id
            elif decision.target == "group":
                target_scope = "group"
            else:
                target_scope = decision.target_scope or decision.target
        intervention = InterventionMessage(
            turn_id=utterance.turn_id,
            target_scope=target_scope,
            content=intervention_content if decision.intervention_needed else "",
            reason=decision.reason,
        )
        logger.info(
            "Node end | meta_decision | turn_id=%s | type=%s | target=%s",
            utterance.turn_id,
            decision.type,
            intervention.target_scope,
        )
        return {"decision": decision, "intervention": intervention}
