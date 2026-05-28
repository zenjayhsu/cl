from __future__ import annotations

from typing import Any, Dict, Literal, Type

from pydantic import BaseModel, ConfigDict, field_validator


class StrictSensorModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _normalize_choice(value: Any, choices: Dict[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().replace("-", "_").replace(" ", "_").lower()
    return choices.get(normalized, value.strip())


class CognitiveDiagnosis(StrictSensorModel):
    student_id: str
    dimension: Literal["cognitive"]
    category: Literal[
        "概念性误解",
        "概念混淆",
        "程序性错误",
        "条件适用错误",
        "因果推理错误",
        "表征映射错误",
        "知识碎片化",
        "浅层加工",
        "证据不足",
        "迁移失败",
        "元认知监控错误",
        "偏题",
        "无明显问题",
    ]
    bloom_level: Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    specific_evidence: str
    diagnosis: str

    @field_validator("dimension", mode="before")
    @classmethod
    def normalize_dimension(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("category", mode="before")
    @classmethod
    def normalize_cognitive_code(cls, value: Any) -> Any:
        return _normalize_choice(
            value,
            {
                "misconception": "概念性误解",
                "conceptual_misconception": "概念性误解",
                "conceptual_confusion": "概念混淆",
                "procedure_error": "程序性错误",
                "procedural_error": "程序性错误",
                "condition_application_error": "条件适用错误",
                "causal_reasoning_error": "因果推理错误",
                "representation_mapping_error": "表征映射错误",
                "fragmented_knowledge": "知识碎片化",
                "shallow_processing": "浅层加工",
                "insufficient_evidence": "证据不足",
                "transfer_failure": "迁移失败",
                "metacognitive_monitoring_error": "元认知监控错误",
                "off_topic": "偏题",
                "none": "无明显问题",
                "no_obvious_problem": "无明显问题",
            },
        )

    @field_validator("bloom_level", mode="before")
    @classmethod
    def normalize_bloom_level(cls, value: Any) -> Any:
        return _normalize_choice(
            value,
            {
                "remember": "Remember",
                "understand": "Understand",
                "understanding": "Understand",
                "apply": "Apply",
                "analyze": "Analyze",
                "analyse": "Analyze",
                "evaluate": "Evaluate",
                "create": "Create",
            },
        )


class AffectiveDiagnosis(StrictSensorModel):
    student_id: str
    dimension: Literal["affective"]
    category: Literal["好奇", "兴趣", "惊讶", "困惑", "焦虑", "挫败", "无聊", "愉悦"]
    specific_evidence: str
    diagnosis: str

    @field_validator("dimension", mode="before")
    @classmethod
    def normalize_dimension(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("category", mode="before")
    @classmethod
    def normalize_affective_code(cls, value: Any) -> Any:
        return _normalize_choice(
            value,
            {
                "curiosity": "好奇",
                "interest": "兴趣",
                "engagement": "兴趣",
                "surprise": "惊讶",
                "confusion": "困惑",
                "anxiety": "焦虑",
                "frustration": "挫败",
                "boredom": "无聊",
                "enjoyment": "愉悦",
                "enjoymentt": "愉悦",
            },
        )


class SocialDiagnosis(StrictSensorModel):
    student_id: str
    dimension: Literal["social"]
    category: Literal[
        "积极参与",
        "回应同伴",
        "促进讨论",
        "沉默观察",
        "平行发言",
        "冲突对立",
        "压制同伴",
        "忽视同伴",
        "不平等参与",
        "协作支持",
        "无明显问题",
    ]
    interaction_target: str | None
    specific_evidence: str
    diagnosis: str

    @field_validator("dimension", mode="before")
    @classmethod
    def normalize_dimension(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("category", mode="before")
    @classmethod
    def normalize_social_code(cls, value: Any) -> Any:
        return _normalize_choice(
            value,
            {
                "silent": "沉默观察",
                "externalization": "积极参与",
                "conflict": "冲突对立",
                "transactivity": "回应同伴",
                "active_participation": "积极参与",
                "peer_response": "回应同伴",
                "discussion_facilitation": "促进讨论",
                "parallel_talk": "平行发言",
                "suppression": "压制同伴",
                "ignoring_peer": "忽视同伴",
                "unequal_participation": "不平等参与",
                "collaborative_support": "协作支持",
                "none": "无明显问题",
            },
        )

    @field_validator("interaction_target", mode="before")
    @classmethod
    def normalize_interaction_target(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return None if cleaned.lower() in {"", "null", "none", "无"} else cleaned
        return value


DiagnosisModel = Type[StrictSensorModel]


def _clean_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"title", "$defs"}:
            continue
        if key == "const":
            cleaned["enum"] = [value]
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [_clean_json_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def response_format_from_model(name: str, model: DiagnosisModel) -> Dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": _clean_json_schema(model.model_json_schema()),
        },
    }


def fallback_diagnosis(model: DiagnosisModel, turn_id: str, reason: str) -> Dict[str, object]:
    del turn_id
    evidence = reason[:240] if reason else "Pydantic 校验失败，传感器输出不可用。"
    if model is CognitiveDiagnosis:
        diagnosis = CognitiveDiagnosis(
            student_id="unknown",
            dimension="cognitive",
            category="无明显问题",
            bloom_level="Remember",
            specific_evidence=f"未生成有效的认知诊断。{evidence}",
            diagnosis="认知诊断不可用，暂不判定学生存在明确认知问题。",
        )
    elif model is AffectiveDiagnosis:
        diagnosis = AffectiveDiagnosis(
            student_id="unknown",
            dimension="affective",
            category="兴趣",
            specific_evidence=f"未生成有效的情感诊断。{evidence}",
            diagnosis="情感诊断不可用，暂按稳定投入状态处理。",
        )
    else:
        diagnosis = SocialDiagnosis(
            student_id="unknown",
            dimension="social",
            category="无明显问题",
            interaction_target=None,
            specific_evidence=f"未生成有效的社交诊断。{evidence}",
            diagnosis="社交诊断不可用，暂不判定存在协作风险。",
        )
    return diagnosis.model_dump(mode="json")
