from __future__ import annotations

from config import LLMRuntimeConfig
from dataclasses import dataclass, field
from typing import Dict, List
import json

from schemas.cscl import ContextWindow, Decision, GroupProfile, IndividualProfile, StudentUtterance
from services.llm_controller import LLMController
from services.structured_output import coerce_structured_output
from services.urgency_calibrator import DEFAULT_TRIGGER_THRESHOLD


def build_llm_controller(config: LLMRuntimeConfig) -> LLMController:
    return LLMController(backend=config.backend, model=config.model, base_url=config.base_url, api_key=config.api_key)


def _trigger_view(triggered_notes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for note in triggered_notes:
        diagnosis = note.get("diagnosis", {})
        if not isinstance(diagnosis, dict):
            diagnosis = {}
        theory_state = diagnosis.get("theory_state", {})
        if not isinstance(theory_state, dict):
            theory_state = {}
        normalized.append(
            {
                "sensor_name": note.get("sensor_name", ""),
                "student_id": note.get("student_id", ""),
                "drawer_id": note.get("drawer_id", ""),
                "dimension": diagnosis.get("dimension", ""),
                "category": diagnosis.get("category", diagnosis.get("code", "")),
                "theory_state": theory_state,
                "bloom_level": diagnosis.get("bloom_level", ""),
                "specific_evidence": diagnosis.get("specific_evidence", ""),
                "diagnosis": diagnosis.get("diagnosis", ""),
                "interaction_target": diagnosis.get("interaction_target", ""),
                "urgency_score": diagnosis.get("urgency_score", 0.0),
                "urgency_level": diagnosis.get("urgency_level", ""),
                "urgency_factors": diagnosis.get("urgency_factors", []),
                "urgency_explanation": diagnosis.get("urgency_explanation", ""),
                "content": note.get("content", ""),
            }
        )
    return normalized


@dataclass
class InterventionPlanner:
    llm: LLMController = field(default_factory=LLMController)

    def plan(
        self,
        utterance: StudentUtterance,
        context_window: ContextWindow,
        individual_profile: IndividualProfile,
        group_profile: GroupProfile,
        triggered_notes: List[Dict[str, object]],
    ) -> Decision:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "intervention_plan",
                "schema": {
                    "type": "object",
                    "properties": {
                        "intervention_needed": {"type": "boolean"},
                        "target": {"type": "string"},
                        "target_student_id": {"type": ["string", "null"]},
                        "intervention_type": {"type": "string"},
                        "reason": {"type": "string"},
                        "based_on_notes": {"type": "array", "items": {"type": "string"}},
                        "based_on_profile": {"type": "string"},
                        "intervention_content": {"type": "string"},
                        "expected_effect": {"type": "string"},
                    },
                    "required": [
                        "intervention_needed",
                        "target",
                        "target_student_id",
                        "intervention_type",
                        "reason",
                        "based_on_notes",
                        "based_on_profile",
                        "intervention_content",
                        "expected_effect",
                    ],
                    "additionalProperties": False,
                },
            },
        }

        trigger_summary = _trigger_view(triggered_notes)
        prompt = (
            "你是 CSCL 系统中的教学干预规划器。只返回 JSON。\n"
            "你必须明确使用 sensor 触发信息和可解释紧急度来做决策。\n"
            "优先关注这些触发字段:\n"
            "- specific_evidence\n"
            "- urgency_score\n"
            "- urgency_level / urgency_factors / urgency_explanation\n\n"
            f"当前学生发言: {json.dumps(utterance.to_dict(), ensure_ascii=False)}\n"
            f"上下文窗口: {json.dumps(context_window.to_dict(), ensure_ascii=False)}\n"
            f"已触发的 sensor notes: {json.dumps(trigger_summary, ensure_ascii=False)}\n"
            f"个体画像: {json.dumps(individual_profile.to_dict(), ensure_ascii=False)}\n"
            f"小组画像: {json.dumps(group_profile.to_dict(), ensure_ascii=False)}\n\n"
            "决策规则:\n"
            f"1. 这些 note 是因为 urgency_score >= {DEFAULT_TRIGGER_THRESHOLD:.2f} 进入 Meta 决策阶段；Sensor 不再输出干预判断标签。\n"
            "2. 你必须综合触发 note、当前上下文、个人画像和小组画像，判断可见干预是否会比继续同伴讨论更有价值。\n"
            "3. 如果风险轻中度、同伴正在自然纠正、画像显示学生能继续推进，可以保持静默观察，但 reason 必须说明依据。\n"
            "4. 如果认知风险会误导小组或反复出现，选择 cognitive_scaffold，用问题或提示引导学生重新分析，不直接给完整答案。\n"
            "5. 如果情感风险影响参与意愿，选择 affective_support，先支持情绪，再给一个小步认知提示。\n"
            "6. 如果社交风险导致沉默、压制或不平等参与，选择 social_regulation，平衡发言机会或邀请同伴回应。\n"
            "7. 如果小组已有基础理解但停留在解释层，选择 higher_order_prompt，引导比较、评价或迁移。\n"
            "8. urgency_score 应影响干预优先级和语气。\n"
            "target 只能是 individual、group 或 none。\n"
            "intervention_type 只能是 cognitive_scaffold、affective_support、social_regulation、higher_order_prompt 或 none。\n"
            "如果 intervention_needed=false，则 target 必须是 none，target_student_id 必须是 null，intervention_type 必须是 none，intervention_content 必须为空字符串，reason 必须解释为什么继续观察。\n"
            "如果 intervention_needed=true，则 intervention_type 不能是 none，intervention_content 必须给出要回传给学生的简短引导语。\n"
            "intervention_content 是唯一会回传到学生聊天框的内容，必须简短自然，使用中文，不直接给完整答案。"
        )

        try:
            completion = self.llm.get_completion(prompt=prompt, response_format=response_format)
            data = coerce_structured_output(completion, response_format)
        except Exception:
            data = {}
        if self._has_meaningful_decision(data):
            if self._is_inconsistent_decision(data):
                return self._fallback(utterance, individual_profile, group_profile, trigger_summary)
            intervention_needed = bool(data.get("intervention_needed"))
            target = str(data.get("target") or ("individual" if intervention_needed else "none"))
            target_student_id = data.get("target_student_id")
            if not isinstance(target_student_id, str) or not target_student_id.strip():
                target_student_id = None
            intervention_type = str(data.get("intervention_type") or ("none" if not intervention_needed else "cognitive_scaffold"))
            intervention_content = str(data.get("intervention_content") or "")
            if intervention_needed and not intervention_content.strip():
                return self._fallback(utterance, individual_profile, group_profile, trigger_summary)
            target_scope = "none"
            if intervention_needed:
                target_scope = target_student_id if target == "individual" and target_student_id else target
            return Decision(
                type=target if intervention_needed else "silence",
                action=intervention_type if intervention_needed else "保持静默",
                reason=str(data.get("reason", "")),
                intervention_needed=intervention_needed,
                target=target,
                target_student_id=target_student_id,
                intervention_type=intervention_type,
                based_on_notes=list(data.get("based_on_notes") or []),
                based_on_profile=str(data.get("based_on_profile", "")),
                intervention_content=intervention_content if intervention_needed else "",
                expected_effect=str(data.get("expected_effect", "")),
                response_text=intervention_content if intervention_needed else "",
                target_scope=target_scope,
                used_profiles=["individual", "group", "triggered_notes"],
            )
        return self._fallback(utterance, individual_profile, group_profile, trigger_summary)

    def _has_meaningful_decision(self, data: Dict[str, object]) -> bool:
        if not data:
            return False
        intervention_needed = data.get("intervention_needed")
        if intervention_needed is True:
            return bool(str(data.get("intervention_content", "")).strip())
        if intervention_needed is False:
            return bool(
                str(data.get("target", "")).strip()
                and str(data.get("intervention_type", "")).strip()
                and str(data.get("reason", "")).strip()
            )
        return any(
            str(data.get(field, "")).strip()
            for field in ["target", "intervention_type", "reason", "intervention_content", "expected_effect"]
        )

    def _is_inconsistent_decision(self, data: Dict[str, object]) -> bool:
        intervention_needed = data.get("intervention_needed")
        target = str(data.get("target") or "").strip()
        intervention_type = str(data.get("intervention_type") or "").strip()
        intervention_content = str(data.get("intervention_content") or "").strip()
        if intervention_needed is False:
            return (
                target not in {"", "none"}
                or intervention_type not in {"", "none"}
                or bool(intervention_content)
            )
        if intervention_needed is True:
            return (
                target in {"", "none"}
                or intervention_type in {"", "none"}
                or not intervention_content
            )
        return False

    def _fallback_should_intervene(
        self,
        highest: Dict[str, object],
        individual_profile: IndividualProfile,
        group_profile: GroupProfile,
    ) -> bool:
        urgency = float(highest.get("urgency_score", 0.0) or 0.0)
        if urgency >= 0.72:
            return True
        category = str(highest.get("category", ""))
        repeated_same_issue = individual_profile.anomaly_counts.get(category, 0) >= 2
        if repeated_same_issue and urgency >= DEFAULT_TRIGGER_THRESHOLD:
            return True
        return urgency >= DEFAULT_TRIGGER_THRESHOLD and group_profile.collaboration_mode in {
            "parallel_play",
            "fragile_participation",
            "conflictual",
        }

    def _fallback(
        self,
        utterance: StudentUtterance,
        individual_profile: IndividualProfile,
        group_profile: GroupProfile,
        trigger_summary: List[Dict[str, object]],
    ) -> Decision:
        if not trigger_summary:
            return Decision(
                type="silence",
                action="保持静默",
                reason="no_triggered_notes",
                intervention_needed=False,
                target="none",
                target_student_id=None,
                intervention_type="none",
                based_on_notes=[],
                based_on_profile=group_profile.llm_summary or individual_profile.llm_summary,
                intervention_content="",
                expected_effect="继续观察，不打断学生讨论。",
                response_text="",
                target_scope="none",
                used_profiles=["individual", "group", "triggered_notes"],
            )

        highest = max(trigger_summary, key=lambda item: float(item.get("urgency_score", 0.0) or 0.0))
        note_basis = [
            (
                f"{item.get('dimension', '')}:{item.get('category', '')}:"
                f"urgency={float(item.get('urgency_score', 0.0) or 0.0):.2f}:"
                f"{item.get('specific_evidence', '')}"
            )
            for item in trigger_summary[:3]
        ]
        profile_basis = individual_profile.llm_summary or group_profile.llm_summary
        if not self._fallback_should_intervene(highest, individual_profile, group_profile):
            return Decision(
                type="silence",
                action="保持静默观察",
                reason=(
                    "triggered_note_but_meta_observe:"
                    f"最高紧急度={float(highest.get('urgency_score', 0.0) or 0.0):.2f}，"
                    "当前更适合保留同伴解释空间；继续参考画像和后续发言观察是否重复出现。"
                ),
                intervention_needed=False,
                target="none",
                target_student_id=None,
                intervention_type="none",
                based_on_notes=note_basis,
                based_on_profile=profile_basis,
                intervention_content="",
                expected_effect="不打断轻中度认知冲突，给小组自然澄清和同伴解释的机会。",
                response_text="",
                target_scope="none",
                used_profiles=["individual", "group", "triggered_notes"],
            )
        dimension = str(highest.get("dimension", ""))
        if dimension == "affective":
            content = (
                f"{utterance.speaker_name or utterance.student_id}，你这个卡点很正常。先别急着下结论，"
                "能不能只指出当前最不确定的一步：是概念含义、代码执行顺序，还是运行结果和预期对不上？"
            )
            return Decision(
                type="individual",
                action="affective_support",
                reason=f"critical_affect_trigger:{highest.get('specific_evidence', '')}",
                intervention_needed=True,
                target="individual",
                target_student_id=utterance.student_id,
                intervention_type="affective_support",
                based_on_notes=note_basis,
                based_on_profile=profile_basis,
                intervention_content=content,
                expected_effect="降低学生焦虑或挫败感，并把注意力收束到一个可回答的小问题。",
                response_text=content,
                target_scope=utterance.student_id,
                used_profiles=["individual", "group", "triggered_notes"],
            )
        if dimension == "cognitive":
            content = (
                f"{utterance.speaker_name or utterance.student_id}，这里可能有一个概念边界需要再确认。"
                "请你把相关概念分别对应到代码位置和运行现象里，先不用急着给最终答案。"
            )
            return Decision(
                type="individual",
                action="cognitive_scaffold",
                reason=f"cognitive_error:{highest.get('specific_evidence', '')}",
                intervention_needed=True,
                target="individual",
                target_student_id=utterance.student_id,
                intervention_type="cognitive_scaffold",
                based_on_notes=note_basis,
                based_on_profile=profile_basis,
                intervention_content=content,
                expected_effect="促使学生重新建立概念、代码位置和运行现象之间的映射关系。",
                response_text=content,
                target_scope=utterance.student_id,
                used_profiles=["individual", "group", "triggered_notes"],
            )
        if dimension == "social":
            target_student = highest.get("interaction_target", "")
            content = (
                "先停一下，我们把发言机会拉平一点。"
                f"请{target_student or '还没充分表达的同学'}先说说自己怎么理解刚才这个例子，其他同学先补充或追问。"
            )
            return Decision(
                type="group",
                action="social_regulation",
                reason=f"social_trigger:{highest.get('specific_evidence', '')}",
                intervention_needed=True,
                target="group",
                target_student_id=None,
                intervention_type="social_regulation",
                based_on_notes=note_basis,
                based_on_profile=profile_basis,
                intervention_content=content,
                expected_effect="缓解发言不均衡或互动断裂，促进更多成员参与。",
                response_text=content,
                target_scope="group",
                used_profiles=["individual", "group", "triggered_notes"],
            )
        if group_profile.collaboration_mode == "parallel_play":
            content = "请每位同学先引用一位同伴刚才的观点，再补充自己的判断：你同意哪一步，想验证哪一步？"
            return Decision(
                type="group",
                action="higher_order_prompt",
                reason="parallel_play_detected",
                intervention_needed=True,
                target="group",
                target_student_id=None,
                intervention_type="higher_order_prompt",
                based_on_notes=note_basis,
                based_on_profile=profile_basis,
                intervention_content=content,
                expected_effect="把平行发言转化为观点比较和证据评价。",
                response_text=content,
                target_scope="group",
                used_profiles=["individual", "group", "triggered_notes"],
            )
        return Decision(
            type="silence",
            action="保持静默",
            reason="insufficient_risk",
            intervention_needed=False,
            target="none",
            target_student_id=None,
            intervention_type="none",
            based_on_notes=note_basis,
            based_on_profile=profile_basis,
            intervention_content="",
            expected_effect="当前风险不足，继续观察。",
            response_text="",
            target_scope="none",
            used_profiles=["individual", "group", "triggered_notes"],
        )
