from __future__ import annotations

from config import LLMRuntimeConfig
from dataclasses import dataclass, field

from schemas.cscl import ContextWindow, Decision, GroupProfile, IndividualProfile, StudentUtterance
from services.intervention_llm import InterventionPlanner, build_llm_controller


@dataclass
class MetaAgent:
    llm_config: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
    planner: InterventionPlanner = field(init=False)

    def __post_init__(self) -> None:
        self.planner = InterventionPlanner(llm=build_llm_controller(self.llm_config))

    def decide(
        self,
        utterance: StudentUtterance,
        context_window: ContextWindow,
        individual_profile: IndividualProfile,
        group_profile: GroupProfile,
        triggered_notes: list[dict[str, object]],
    ) -> Decision:
        return self.planner.plan(
            utterance=utterance,
            context_window=context_window,
            individual_profile=individual_profile,
            group_profile=group_profile,
            triggered_notes=triggered_notes,
        )
