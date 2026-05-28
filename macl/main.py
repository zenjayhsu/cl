from __future__ import annotations

import warnings
import os
from typing import Dict, List

warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    category=Warning,
    module=r"langgraph\.cache\.base.*",
)

from config import load_runtime_config_from_env

from discussion_simulator import (
    DEFAULT_TOPIC,
    DiscussionSimulator,
)
try:
    from langchain_core._api.deprecation import suppress_langchain_deprecation_warning
except Exception:
    suppress_langchain_deprecation_warning = None

if suppress_langchain_deprecation_warning is not None:
    with suppress_langchain_deprecation_warning():
        from graph.workflow import CSCLWorkflow
else:
    from graph.workflow import CSCLWorkflow
from services.runtime_logging import configure_logging, get_logger
from services.urgency_calibrator import DEFAULT_TRIGGER_THRESHOLD


logger = get_logger("main")
SEPARATOR = "─" * 78

SENSOR_LABELS = {
    "CognitiveSensor": "认知诊断",
    "AffectiveSensor": "情感诊断",
    "SocialSensor": "社交诊断",
}

SENSOR_ICONS = {
    "CognitiveSensor": "🧠",
    "AffectiveSensor": "💭",
    "SocialSensor": "🤝",
}

DECISION_LABELS = {
    "silence": "静默观察",
    "individual": "个体干预",
    "group": "小组干预",
    "none": "无干预",
}

INTERVENTION_LABELS = {
    "cognitive_scaffold": "认知支架",
    "affective_support": "情感支持",
    "social_regulation": "社交调节",
    "higher_order_prompt": "高阶思维促进",
    "none": "无",
}


def _shorten(text: object, limit: int = 220) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _label(mapping: Dict[str, str], value: object, fallback: str = "未说明") -> str:
    text = str(value or "").strip()
    return mapping.get(text, text or fallback)


def _note_evidence(note: Dict[str, object]) -> tuple[str, str, str]:
    diagnosis = note.get("diagnosis") or {}
    if not isinstance(diagnosis, dict):
        return "无", "", ""
    category = str(diagnosis.get("category") or note.get("cognitive_tag") or note.get("affective_tag") or note.get("social_tag") or "无")
    evidence = str(diagnosis.get("specific_evidence") or "")
    bloom = str(diagnosis.get("bloom_level") or "")
    return category, evidence, bloom


def _format_urgency_factors(diagnosis: Dict[str, object]) -> str:
    factors = diagnosis.get("urgency_factors") or []
    if not isinstance(factors, list):
        return ""
    compact: List[str] = []
    for factor in factors[:4]:
        if not isinstance(factor, dict):
            continue
        name = str(factor.get("name") or "")
        value = factor.get("value", 0.0)
        try:
            compact.append(f"{name}={float(value):+.2f}")
        except (TypeError, ValueError):
            compact.append(f"{name}={value}")
    return "；".join(item for item in compact if item)


def _urgent_notes(record: Dict[str, object]) -> List[Dict[str, object]]:
    urgent: List[Dict[str, object]] = []
    for note in record.get("written_notes", []):
        if not isinstance(note, dict):
            continue
        diagnosis = note.get("diagnosis") or {}
        if not isinstance(diagnosis, dict):
            continue
        urgency = float(diagnosis.get("urgency_score", note.get("urgency", 0.0)) or 0.0)
        if urgency >= DEFAULT_TRIGGER_THRESHOLD:
            urgent.append(note)
    return urgent


def _format_profile_snapshot(record: Dict[str, object]) -> tuple[str, str]:
    profile = record.get("individual_profile") or {}
    group_profile = record.get("group_profile") or {}
    if isinstance(profile, dict) and profile:
        individual = (
            f"学生={profile.get('student_id', '')}; "
            f"姓名={profile.get('name', '')}; "
            f"知识={profile.get('knowledge_state', '')}; "
            f"认知={profile.get('cognitive_pattern', '')}; "
            f"情感={profile.get('affective_pattern', '')}; "
            f"社交={profile.get('social_pattern', '')}"
        )
    else:
        individual = ""
    if isinstance(group_profile, dict) and group_profile:
        group = (
            f"小组={group_profile.get('group_id', '')}; "
            f"知识={group_profile.get('overall_knowledge_state', '')}; "
            f"共同误区={group_profile.get('common_misconceptions', [])}; "
            f"协作={group_profile.get('collaboration_pattern', '')}; "
            f"情感氛围={group_profile.get('affective_climate', '')}"
        )
    else:
        group = ""
    if individual and group:
        return individual, group

    student_response = record.get("student_response") or {}
    if not isinstance(student_response, dict):
        student_response = {}
    dynamic_states = record.get("dynamic_states_after_turn_update") or {}
    if not isinstance(dynamic_states, dict):
        dynamic_states = {}

    turn = record.get("turn") or {}
    student_id = turn.get("student_id", "") if isinstance(turn, dict) else ""
    student_state = dynamic_states.get(student_id, "")

    written_notes = [note for note in record.get("written_notes", []) if isinstance(note, dict)]
    cognitive_tags = [note.get("cognitive_tag") for note in written_notes if note.get("cognitive_tag")]
    affective_tags = [note.get("affective_tag") for note in written_notes if note.get("affective_tag")]
    social_tags = [note.get("social_tag") for note in written_notes if note.get("social_tag")]

    individual = (
        f"学生={student_id}; "
        f"本轮认知={cognitive_tags[-1] if cognitive_tags else 'none'}; "
        f"情感={affective_tags[-1] if affective_tags else 'none'}; "
        f"社交={social_tags[-1] if social_tags else 'none'}; "
        f"动态状态={student_state or student_response.get('dynamic_state_before_workflow', 'none')}"
    )
    group = "; ".join(f"{key}={value}" for key, value in dynamic_states.items()) or "none"
    return individual, group


def _print_record(record: Dict[str, object]) -> None:
    turn = record.get("turn") or {}
    intervention = record.get("intervention") or {}
    decision = record.get("decision") or {}
    if not isinstance(turn, dict) or not isinstance(intervention, dict) or not isinstance(decision, dict):
        return

    speaker = turn.get("speaker_name", turn.get("student_id", "学生"))
    student_id = turn.get("student_id", "")
    print(f"\n{SEPARATOR}", flush=True)
    print(f"💬 学生发言 | {speaker}（{student_id}）", flush=True)
    print(f"   {turn.get('text', '')}", flush=True)

    urgent_notes = _urgent_notes(record)
    if urgent_notes:
        individual_profile, group_profile = _format_profile_snapshot(record)
        print(f"\n👤 当前学生画像", flush=True)
        print(f"   {_shorten(individual_profile, 360)}", flush=True)
        print(f"\n👥 小组画像", flush=True)
        print(f"   {_shorten(group_profile, 500)}", flush=True)
        print(f"\n🚨 触发诊断", flush=True)
        for note in urgent_notes:
            diagnosis = note.get("diagnosis") or {}
            if not isinstance(diagnosis, dict):
                diagnosis = {}
            sensor_name = str(note.get("sensor_name", ""))
            category, evidence, bloom = _note_evidence(note)
            urgency = float(diagnosis.get("urgency_score", note.get("urgency", 0.0)) or 0.0)
            urgency_level = str(diagnosis.get("urgency_level") or "")
            bloom_text = f" | Bloom: {bloom}" if bloom else ""
            level_text = f" | 等级: {urgency_level}" if urgency_level else ""
            print(
                f"   {SENSOR_ICONS.get(sensor_name, '•')} "
                f"{_label(SENSOR_LABELS, sensor_name)} | 标签: {category}{bloom_text} | "
                f"紧急度: {urgency:.2f}{level_text}",
                flush=True,
            )
            factor_text = _format_urgency_factors(diagnosis)
            if factor_text:
                print(f"      因子: {_shorten(factor_text, 220)}", flush=True)
            urgency_explanation = str(diagnosis.get("urgency_explanation") or "")
            if urgency_explanation:
                print(f"      解释: {_shorten(urgency_explanation, 260)}", flush=True)
            if evidence:
                print(f"      证据: {_shorten(evidence, 260)}", flush=True)
            diagnosis_text = str(diagnosis.get("diagnosis") or "")
            if diagnosis_text:
                print(f"      诊断: {_shorten(diagnosis_text, 260)}", flush=True)

        intervention_needed = bool(decision.get("intervention_needed"))
        decision_type = _label(DECISION_LABELS, decision.get("type"))
        intervention_type = _label(INTERVENTION_LABELS, decision.get("intervention_type") if intervention_needed else "none")
        target = intervention.get("target_scope") or decision.get("target_scope") or decision.get("target")
        target_text = _label(DECISION_LABELS, target if intervention_needed else "none", fallback=str(target or "无"))
        needed_text = "执行干预" if intervention_needed else "暂不干预"
        print(
            f"\n🤖 Meta-Agent 决策 | {needed_text} | 类型: {decision_type} | "
            f"方式: {intervention_type} | 对象: {target_text}",
            flush=True,
        )
        reason = decision.get("reason") or intervention.get("reason") or ""
        if reason:
            print(f"   原因: {_shorten(reason, 260)}", flush=True)
        expected = decision.get("expected_effect") or ""
        if expected:
            print(f"   预期效果: {_shorten(expected, 260)}", flush=True)

    meta_content = intervention.get("content") or ""
    if meta_content:
        print(f"\n📣 回传给学生", flush=True)
        print(f"   {meta_content}", flush=True)


def load_topic() -> str:
    return os.getenv("CSCL_TOPIC", DEFAULT_TOPIC).strip() or DEFAULT_TOPIC


def main() -> None:
    configure_logging()
    runtime_config = load_runtime_config_from_env()
    topic = load_topic()
    logger.info("Loaded runtime config with backend=%s model=%s", runtime_config.llm.backend, runtime_config.llm.model)
    logger.info("Loaded topic: %s", topic)
    student_count = int(os.getenv("CSCL_STUDENT_COUNT", "4"))
    workflow = CSCLWorkflow(student_ids=[f"s{index + 1}" for index in range(student_count)], runtime_config=runtime_config)
    simulator = DiscussionSimulator(topic=topic, prompt_template="", llm_config=runtime_config.llm)
    logger.info("Starting simulation with random max_turns in [30, 50]")
    print(f"{SEPARATOR}", flush=True)
    print(f"📌 议题: {topic}", flush=True)
    print(f"{SEPARATOR}", flush=True)
    records = simulator.run(workflow=workflow, max_turns=None, on_record=_print_record)
    logger.info("Simulation completed with %s records", len(records))


if __name__ == "__main__":
    main()
