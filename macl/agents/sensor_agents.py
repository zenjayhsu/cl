from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import json
import uuid

from pydantic import ValidationError

from config import LLMRuntimeConfig
from schemas.cscl import CSCLNote, ContextWindow, StudentUtterance
from schemas.sensor_diagnosis import (
    AffectiveDiagnosis,
    CognitiveDiagnosis,
    DiagnosisModel,
    SocialDiagnosis,
    fallback_diagnosis,
    response_format_from_model,
)
from services.llm_controller import LLMController
from services.runtime_logging import get_logger
from services.structured_output import parse_json_object


logger = get_logger("sensor")


def _build_llm_controller(config: LLMRuntimeConfig) -> LLMController:
    return LLMController(backend=config.backend, model=config.model, base_url=config.base_url, api_key=config.api_key)

def _theory_value(diagnosis: Dict[str, object], field_name: str, *aliases: str) -> str:
    theory_state = diagnosis.get("theory_state")
    if not isinstance(theory_state, dict):
        theory_state = {}

    candidates: List[object] = [theory_state.get(field_name), diagnosis.get(field_name)]
    candidates.extend(diagnosis.get(alias) for alias in aliases)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            normalized = candidate.strip()
            theory_state[field_name] = normalized
            diagnosis["theory_state"] = theory_state
            return normalized

    theory_state[field_name] = ""
    diagnosis["theory_state"] = theory_state
    return ""


def _context_payload(context_window: ContextWindow) -> Dict[str, object]:
    return {
        "focus_student_id": context_window.focus_student_id,
        "latest_turn_id": context_window.latest_turn.turn_id,
        "recent_turn_ids": [turn.turn_id for turn in context_window.recent_turns],
        "recent_turns": [turn.to_dict() for turn in context_window.recent_turns],
        "max_turns": context_window.max_turns,
    }


@dataclass
class BaseSensor:
    sensor_name: str
    llm_config: LLMRuntimeConfig

    def __post_init__(self) -> None:
        self.llm = _build_llm_controller(self.llm_config)

    def _build_note(
        self,
        utterance: StudentUtterance,
        context_window: ContextWindow,
        diagnosis: Dict[str, object],
        cognitive_tag: str = "",
        affective_tag: str = "",
        social_tag: str = "",
    ) -> CSCLNote:
        diagnosis["student_id"] = str(diagnosis.get("student_id") or utterance.student_id)
        return CSCLNote(
            note_id=f"{self.sensor_name.lower()}-{uuid.uuid4().hex[:10]}",
            student_id=utterance.student_id,
            speaker_name=utterance.speaker_name,
            content=f"[{self.sensor_name}::{diagnosis.get('category', '')}] {utterance.student_id}: {utterance.text}",
            timestamp=utterance.timestamp,
            cognitive_tag=cognitive_tag,
            affective_tag=affective_tag,
            social_tag=social_tag,
            diagnosis=diagnosis,
            context_window=_context_payload(context_window),
            source_turn_id=utterance.turn_id,
            urgency=max(0.0, min(float(diagnosis.get("urgency_score", 0.0) or 0.0), 1.0)),
            sensor_name=self.sensor_name,
        )

    def _validate_completion(self, completion: str, model: DiagnosisModel) -> Dict[str, object]:
        payload = parse_json_object(completion)
        return model.model_validate(payload).model_dump(mode="json")

    def _generate(self, prompt: str, response_format: Dict[str, object], model: DiagnosisModel, turn_id: str) -> Dict[str, object]:
        logger.info("Sensor generation start | sensor=%s", self.sensor_name)
        try:
            completion = self.llm.get_completion(prompt=prompt, response_format=response_format)
        except Exception as exc:
            diagnosis = fallback_diagnosis(model=model, turn_id=turn_id, reason=str(exc))
            diagnosis["schema_fallback"] = True
            logger.exception("Sensor generation LLM failure | sensor=%s", self.sensor_name)
            return diagnosis
        try:
            diagnosis = self._validate_completion(completion=completion, model=model)
            logger.info("Sensor generation success | sensor=%s | category=%s", self.sensor_name, diagnosis.get("category", ""))
            return diagnosis
        except (ValidationError, ValueError) as exc:
            validation_error = str(exc)

        repair_prompt = (
            f"{prompt}\n\n"
            "你上一次回答无效或不完整。\n"
            f"Pydantic 校验错误: {validation_error}\n"
            "只返回一个 JSON 对象，并填满所有必填字段。\n"
            "枚举标签必须严格匹配 schema 中给出的取值。\n"
            "不要输出 schema 未列出的字段。\n"
            "不要添加解释、Markdown 或代码块。"
        )
        logger.warning("Sensor generation repair requested | sensor=%s", self.sensor_name)
        repaired_completion = self.llm.get_completion(prompt=repair_prompt, response_format=response_format, temperature=0.0)
        try:
            repaired_diagnosis = self._validate_completion(completion=repaired_completion, model=model)
            repaired_diagnosis["schema_repaired"] = True
            logger.warning("Sensor generation repaired | sensor=%s | category=%s", self.sensor_name, repaired_diagnosis.get("category", ""))
            return repaired_diagnosis
        except (ValidationError, ValueError) as repaired_exc:
            validation_error = f"{validation_error}; repair_error={repaired_exc}"

        diagnosis = fallback_diagnosis(model=model, turn_id=turn_id, reason=validation_error)
        diagnosis["schema_fallback"] = True
        logger.error("Sensor generation fallback | sensor=%s", self.sensor_name)
        return diagnosis


def _cognitive_prompt(utterance: StudentUtterance, context_window: ContextWindow) -> tuple[str, Dict[str, object]]:
    response_format = response_format_from_model("cognitive_diagnosis", CognitiveDiagnosis)
    prompt = (
        "你是 CSCL 系统中的认知 Sensor。只返回 JSON。\n"
        "你的任务是根据固定大小的对话上下文窗口和最新学生发言，诊断该学生的 C 语言认知状态。\n"
        "不要把 Bloom 水平当作任务达标评价，它只描述学生当前认知加工层次。\n\n"
        "只返回这些字段: student_id, dimension, category, bloom_level, specific_evidence, diagnosis。\n"
        f"student_id 必须是 {utterance.student_id}。\n"
        "dimension 必须是 cognitive。\n"
        "category 必须从这些中文标签中选择: 概念性误解、概念混淆、程序性错误、条件适用错误、因果推理错误、表征映射错误、知识碎片化、浅层加工、证据不足、迁移失败、元认知监控错误、偏题、无明显问题。\n"
        "bloom_level 必须从这些英文标签中选择: Remember, Understand, Apply, Analyze, Evaluate, Create。\n"
        "specific_evidence 必须提取或概括学生发言中的具体证据。\n"
        "diagnosis 用中文简要说明学生认知状态。\n"
        "不要输出紧急度或干预判断字段；紧急度由系统的可解释计算模块统一计算。\n\n"
        f"当前学生发言: {json.dumps(utterance.to_dict(), ensure_ascii=False)}\n"
        f"上下文窗口: {json.dumps(context_window.to_dict(), ensure_ascii=False)}\n"
        "返回一个符合 schema 的 JSON 对象。"
    )
    return prompt, response_format


def _affective_prompt(utterance: StudentUtterance, context_window: ContextWindow) -> tuple[str, Dict[str, object]]:
    response_format = response_format_from_model("affective_diagnosis", AffectiveDiagnosis)
    prompt = (
        "你是 CSCL 系统中的情感 Sensor。只返回 JSON。\n"
        "你的任务是根据固定大小的对话上下文窗口和最新学生发言，诊断学生在协作学习中的情绪和学习情感状态。\n\n"
        "只返回这些字段: student_id, dimension, category, specific_evidence, diagnosis。\n"
        f"student_id 必须是 {utterance.student_id}。\n"
        "dimension 必须是 affective。\n"
        "category 必须从这些中文标签中选择: 好奇、兴趣、惊讶、困惑、焦虑、挫败、无聊、愉悦。\n"
        "specific_evidence 必须用中文综合说明情绪触发原因和行为表现。\n"
        "diagnosis 用中文简要说明学生情感状态。\n"
        "不要输出紧急度或干预判断字段；紧急度由系统的可解释计算模块统一计算。\n\n"
        f"当前学生发言: {json.dumps(utterance.to_dict(), ensure_ascii=False)}\n"
        f"上下文窗口: {json.dumps(context_window.to_dict(), ensure_ascii=False)}\n"
        "返回一个符合 schema 的 JSON 对象。"
    )
    return prompt, response_format


def _social_prompt(utterance: StudentUtterance, context_window: ContextWindow) -> tuple[str, Dict[str, object]]:
    response_format = response_format_from_model("social_diagnosis", SocialDiagnosis)
    prompt = (
        "你是 CSCL 系统中的社交 Sensor。只返回 JSON。\n"
        "你的任务是根据固定大小的对话上下文窗口和最新学生发言，诊断学生在小组协作中的互动状态。\n\n"
        "只返回这些字段: student_id, dimension, category, interaction_target, specific_evidence, diagnosis。\n"
        f"student_id 必须是 {utterance.student_id}。\n"
        "dimension 必须是 social。\n"
        "category 必须从这些中文标签中选择: 积极参与、回应同伴、促进讨论、沉默观察、平行发言、冲突对立、压制同伴、忽视同伴、不平等参与、协作支持、无明显问题。\n"
        "interaction_target 是被回应、被影响、被忽视或被压制的学生编号；如果没有明确对象，填 null。\n"
        "specific_evidence 必须用中文说明互动证据。\n"
        "diagnosis 用中文简要说明学生社交协作状态。\n"
        "不要输出紧急度或干预判断字段；紧急度由系统的可解释计算模块统一计算。\n\n"
        f"当前学生发言: {json.dumps(utterance.to_dict(), ensure_ascii=False)}\n"
        f"上下文窗口: {json.dumps(context_window.to_dict(), ensure_ascii=False)}\n"
        "返回一个符合 schema 的 JSON 对象。"
    )
    return prompt, response_format


class CognitiveSensor(BaseSensor):
    def __init__(self, llm_config: LLMRuntimeConfig) -> None:
        super().__init__(sensor_name="CognitiveSensor", llm_config=llm_config)

    def observe(self, utterance: StudentUtterance, context_window: ContextWindow) -> CSCLNote:
        prompt, response_format = _cognitive_prompt(utterance, context_window)
        diagnosis = self._generate(
            prompt=prompt,
            response_format=response_format,
            model=CognitiveDiagnosis,
            turn_id=utterance.turn_id,
        )
        diagnosis["student_id"] = utterance.student_id
        cognitive_tag = str(diagnosis.get("category", ""))
        return self._build_note(utterance, context_window, diagnosis=diagnosis, cognitive_tag=cognitive_tag)


class AffectiveSensor(BaseSensor):
    def __init__(self, llm_config: LLMRuntimeConfig) -> None:
        super().__init__(sensor_name="AffectiveSensor", llm_config=llm_config)

    def observe(self, utterance: StudentUtterance, context_window: ContextWindow) -> CSCLNote:
        prompt, response_format = _affective_prompt(utterance, context_window)
        diagnosis = self._generate(
            prompt=prompt,
            response_format=response_format,
            model=AffectiveDiagnosis,
            turn_id=utterance.turn_id,
        )
        diagnosis["student_id"] = utterance.student_id
        affective_tag = str(diagnosis.get("category", ""))
        return self._build_note(utterance, context_window, diagnosis=diagnosis, affective_tag=affective_tag)


class SocialSensor(BaseSensor):
    def __init__(self, llm_config: LLMRuntimeConfig) -> None:
        super().__init__(sensor_name="SocialSensor", llm_config=llm_config)

    def observe(self, utterance: StudentUtterance, context_window: ContextWindow) -> CSCLNote:
        prompt, response_format = _social_prompt(utterance, context_window)
        diagnosis = self._generate(
            prompt=prompt,
            response_format=response_format,
            model=SocialDiagnosis,
            turn_id=utterance.turn_id,
        )
        diagnosis["student_id"] = utterance.student_id
        social_tag = str(diagnosis.get("category", ""))
        return self._build_note(utterance, context_window, diagnosis=diagnosis, social_tag=social_tag)
