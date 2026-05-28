from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from schemas.cscl import CSCLNote, ContextWindow, GroupProfile, IndividualProfile


DEFAULT_TRIGGER_THRESHOLD = 0.61

COGNITIVE_BASE = {
    "无明显问题": 0.08,
    "证据不足": 0.34,
    "浅层加工": 0.38,
    "知识碎片化": 0.42,
    "偏题": 0.46,
    "概念混淆": 0.52,
    "条件适用错误": 0.56,
    "表征映射错误": 0.58,
    "程序性错误": 0.60,
    "迁移失败": 0.64,
    "概念性误解": 0.66,
    "因果推理错误": 0.66,
    "元认知监控错误": 0.72,
}

AFFECTIVE_BASE = {
    "愉悦": 0.08,
    "兴趣": 0.10,
    "好奇": 0.12,
    "惊讶": 0.24,
    "困惑": 0.42,
    "无聊": 0.58,
    "焦虑": 0.62,
    "挫败": 0.72,
}

SOCIAL_BASE = {
    "促进讨论": 0.08,
    "协作支持": 0.08,
    "积极参与": 0.10,
    "回应同伴": 0.10,
    "无明显问题": 0.08,
    "平行发言": 0.46,
    "沉默观察": 0.48,
    "不平等参与": 0.58,
    "忽视同伴": 0.62,
    "冲突对立": 0.70,
    "压制同伴": 0.78,
}

POSITIVE_AFFECTIVE = {"好奇", "兴趣", "愉悦"}
COGNITIVE_OK = "无明显问题"
SOCIAL_RISK = {"沉默观察", "平行发言", "冲突对立", "压制同伴", "忽视同伴", "不平等参与"}
PEER_REPAIR_CUES = ("我补充", "总结", "同意", "举个例子", "试一下", "验证", "你说", "大家觉得")


@dataclass
class UrgencyFactor:
    name: str
    value: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "value": round(self.value, 3), "reason": self.reason}


@dataclass
class UrgencyCalibration:
    score: float
    level: str
    triggered: bool
    factors: List[UrgencyFactor] = field(default_factory=list)
    explanation: str = ""
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD

    def to_dict(self) -> Dict[str, object]:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "triggered": self.triggered,
            "trigger_threshold": self.trigger_threshold,
            "factors": [factor.to_dict() for factor in self.factors],
            "explanation": self.explanation,
        }


def _clip(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _level(score: float, trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD) -> str:
    if score >= 0.81:
        return "high"
    if score >= trigger_threshold:
        return "medium_high"
    if score >= 0.41:
        return "medium"
    if score >= 0.21:
        return "low"
    return "minimal"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_trajectory_items(items: List[Dict[str, object]], size: int = 3) -> List[Dict[str, object]]:
    return list(items[-size:]) if items else []


class UrgencyCalibrator:
    def __init__(self, trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD) -> None:
        self.trigger_threshold = trigger_threshold

    def calibrate_note(
        self,
        note: CSCLNote,
        context_window: ContextWindow,
        individual_profile: IndividualProfile,
        group_profile: GroupProfile,
    ) -> CSCLNote:
        calibration = self.calibrate(
            diagnosis=note.diagnosis or {},
            context_window=context_window,
            individual_profile=individual_profile,
            group_profile=group_profile,
        )
        diagnosis = dict(note.diagnosis or {})
        diagnosis["urgency_score"] = calibration.score
        diagnosis["urgency_level"] = calibration.level
        diagnosis["urgency_triggered"] = calibration.triggered
        diagnosis["urgency_threshold"] = calibration.trigger_threshold
        diagnosis["urgency_factors"] = [factor.to_dict() for factor in calibration.factors]
        diagnosis["urgency_explanation"] = calibration.explanation
        diagnosis["urgency_source"] = "rule_based_calibrator_v1"
        note.diagnosis = diagnosis
        note.urgency = calibration.score
        return note

    def calibrate(
        self,
        diagnosis: Dict[str, object],
        context_window: ContextWindow,
        individual_profile: IndividualProfile,
        group_profile: GroupProfile,
    ) -> UrgencyCalibration:
        dimension = str(diagnosis.get("dimension") or "")
        category = str(diagnosis.get("category") or "")
        factors: List[UrgencyFactor] = []

        base = self._category_base(dimension, category)
        factors.append(UrgencyFactor("category_base", base, f"{dimension}:{category} 的基础风险权重"))

        evidence = str(diagnosis.get("specific_evidence") or "")
        if 0 < len(evidence) < 18 and category not in {COGNITIVE_OK, "兴趣", "好奇", "愉悦", "积极参与", "回应同伴", "促进讨论", "协作支持"}:
            factors.append(UrgencyFactor("evidence_weakness", 0.03, "诊断证据较短，说明该判断还需要进一步核实"))
        elif len(evidence) >= 45 and base >= 0.41:
            factors.append(UrgencyFactor("evidence_specificity", 0.04, "诊断证据较具体，风险判断可信度提高"))

        if dimension == "cognitive":
            self._add_cognitive_factors(factors, diagnosis, category, individual_profile, group_profile)
        elif dimension == "affective":
            self._add_affective_factors(factors, category, individual_profile)
        elif dimension == "social":
            self._add_social_factors(factors, category, group_profile)

        peer_repair_delta = self._peer_repair_delta(context_window, category, base)
        if peer_repair_delta:
            factors.append(
                UrgencyFactor(
                    "peer_repair_potential",
                    peer_repair_delta,
                    "上下文中出现同伴回应、总结或验证线索，可先保留一定协作修复空间",
                )
            )

        score = _clip(sum(factor.value for factor in factors))
        level = _level(score, self.trigger_threshold)
        triggered = score >= self.trigger_threshold
        explanation = self._explain(dimension, category, score, level, triggered, factors)
        return UrgencyCalibration(
            score=score,
            level=level,
            triggered=triggered,
            factors=factors,
            explanation=explanation,
            trigger_threshold=self.trigger_threshold,
        )

    def _category_base(self, dimension: str, category: str) -> float:
        if dimension == "cognitive":
            return COGNITIVE_BASE.get(category, 0.40)
        if dimension == "affective":
            return AFFECTIVE_BASE.get(category, 0.35)
        if dimension == "social":
            return SOCIAL_BASE.get(category, 0.35)
        return 0.30

    def _add_cognitive_factors(
        self,
        factors: List[UrgencyFactor],
        diagnosis: Dict[str, object],
        category: str,
        individual_profile: IndividualProfile,
        group_profile: GroupProfile,
    ) -> None:
        if category == COGNITIVE_OK:
            factors.append(UrgencyFactor("no_cognitive_issue", -0.04, "认知诊断未发现明显问题"))
            return

        bloom_level = str(diagnosis.get("bloom_level") or "")
        if bloom_level in {"Apply", "Analyze", "Evaluate", "Create"} and category in {
            "概念性误解",
            "概念混淆",
            "因果推理错误",
            "表征映射错误",
            "迁移失败",
        }:
            factors.append(UrgencyFactor("higher_order_error", 0.05, f"{bloom_level} 层次出现误解，可能影响迁移和推理"))

        recurrence = individual_profile.anomaly_counts.get(category, 0)
        if recurrence:
            factors.append(
                UrgencyFactor(
                    "individual_recurrence",
                    min(0.18, recurrence * 0.06),
                    f"学生历史画像中同类问题已出现 {recurrence} 次",
                )
            )

        profile_text = " ".join(
            [
                individual_profile.knowledge_state,
                individual_profile.cognitive_pattern,
                " ".join(individual_profile.main_difficulties),
            ]
        )
        if category and category in profile_text:
            factors.append(UrgencyFactor("profile_consistency", 0.05, "该问题与个人画像中的主要困难一致"))

        group_text = " ".join([group_profile.overall_knowledge_state, " ".join(group_profile.common_misconceptions)])
        if category and category in group_text:
            factors.append(UrgencyFactor("group_common_risk", 0.06, "该问题也出现在小组共同误区中"))

    def _add_affective_factors(
        self,
        factors: List[UrgencyFactor],
        category: str,
        individual_profile: IndividualProfile,
    ) -> None:
        if category in POSITIVE_AFFECTIVE:
            factors.append(UrgencyFactor("positive_affect", -0.03, "当前情绪状态偏积极"))
            return

        recent = _latest_trajectory_items(individual_profile.affective_state_trajectory)
        critical_count = sum(1 for item in recent if bool(item.get("is_critical")))
        if critical_count:
            factors.append(UrgencyFactor("recent_affective_risk", min(0.12, critical_count * 0.04), "近期画像中已有情感风险记录"))

        if category in {"焦虑", "挫败", "无聊"}:
            factors.append(UrgencyFactor("participation_risk", 0.05, f"{category} 可能降低持续参与意愿"))

    def _add_social_factors(
        self,
        factors: List[UrgencyFactor],
        category: str,
        group_profile: GroupProfile,
    ) -> None:
        if category not in SOCIAL_RISK:
            factors.append(UrgencyFactor("collaboration_support", -0.03, "当前互动对协作有支持作用或无明显社交风险"))
            return

        if group_profile.conflict_level >= 3:
            factors.append(UrgencyFactor("group_conflict_level", 0.07, f"小组冲突水平为 {group_profile.conflict_level}"))
        if group_profile.collaboration_mode in {"parallel_play", "fragile_participation", "conflictual"}:
            factors.append(UrgencyFactor("group_collaboration_mode", 0.06, f"小组协作模式为 {group_profile.collaboration_mode}"))

        latest_balance = _latest_trajectory_items(group_profile.participation_balance_trajectory, size=1)
        imbalance = _safe_float(latest_balance[0].get("imbalance")) if latest_balance else 0.0
        if imbalance >= 3:
            factors.append(UrgencyFactor("participation_imbalance", 0.05, f"小组发言次数差距为 {imbalance:g}"))

    def _peer_repair_delta(self, context_window: ContextWindow, category: str, base: float) -> float:
        if base < 0.41 or category in {"元认知监控错误", "挫败", "压制同伴", "冲突对立"}:
            return 0.0
        recent_speakers = {turn.student_id for turn in context_window.recent_turns[-4:]}
        if len(recent_speakers) < 2:
            return 0.0
        recent_text = " ".join(turn.text for turn in context_window.recent_turns[-4:])
        if any(cue in recent_text for cue in PEER_REPAIR_CUES):
            return -0.05
        return 0.0

    def _explain(
        self,
        dimension: str,
        category: str,
        score: float,
        level: str,
        triggered: bool,
        factors: List[UrgencyFactor],
    ) -> str:
        factor_text = "；".join(f"{factor.name}={factor.value:+.2f}" for factor in factors)
        trigger_text = "达到 Meta 触发阈值" if triggered else "未达到 Meta 触发阈值"
        return (
            f"{dimension}:{category} 的紧急度由可解释规则计算，"
            f"因子为 {factor_text}，最终分数 {score:.2f}，等级 {level}，{trigger_text}。"
        )
