from __future__ import annotations

from config import LLMRuntimeConfig
from dataclasses import dataclass, field
from typing import Dict, List
import json

from schemas.cscl import GroupProfile, IndividualProfile, StudentNoteBox
from services.llm_controller import LLMController
from services.structured_output import coerce_structured_output


def build_llm_controller(config: LLMRuntimeConfig) -> LLMController:
    return LLMController(backend=config.backend, model=config.model, base_url=config.base_url, api_key=config.api_key)


@dataclass
class ProfileSynthesizer:
    llm: LLMController = field(default_factory=LLMController)

    def summarize_individual(self, profile: IndividualProfile, box: StudentNoteBox, latest_notes: List[Dict[str, object]]) -> IndividualProfile:
        fallback_summary = self._fallback_individual_summary(profile, box)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "individual_profile_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "string"},
                        "name": {"type": "string"},
                        "knowledge_state": {"type": "string"},
                        "cognitive_pattern": {"type": "string"},
                        "affective_pattern": {"type": "string"},
                        "social_pattern": {"type": "string"},
                        "higher_order_ability": {"type": "string"},
                        "main_difficulties": {"type": "array", "items": {"type": "string"}},
                        "strengths": {"type": "array", "items": {"type": "string"}},
                        "recommended_support": {"type": "array", "items": {"type": "string"}},
                        "last_updated": {"type": "string"},
                    },
                    "required": [
                        "student_id",
                        "name",
                        "knowledge_state",
                        "cognitive_pattern",
                        "affective_pattern",
                        "social_pattern",
                        "higher_order_ability",
                        "main_difficulties",
                        "strengths",
                        "recommended_support",
                        "last_updated",
                    ],
                    "additionalProperties": False,
                },
            },
        }
        prompt = (
            "你正在总结一名学生在 CSCL 系统中的可解释学习画像。\n"
            "只返回 JSON。\n"
            f"学生 id: {profile.student_id}\n"
            f"已有画像档案: {json.dumps(profile.to_dict(), ensure_ascii=False)}\n"
            f"抽屉快照: {json.dumps(box.to_dict(), ensure_ascii=False)}\n"
            f"最新 notes: {json.dumps(latest_notes, ensure_ascii=False)}\n"
            "请输出结构化个人画像，覆盖当前 C 语言知识掌握、常见认知问题或优势、情感变化、协作参与、高阶能力、主要困难、优势和个性化支持建议。"
        )
        try:
            completion = self.llm.get_completion(prompt=prompt, response_format=response_format)
            data = coerce_structured_output(completion, response_format)
        except Exception:
            data = {}
        profile.student_id = str(data.get("student_id") or fallback_summary["student_id"])
        profile.name = str(data.get("name") or fallback_summary["name"])
        profile.knowledge_state = str(data.get("knowledge_state") or fallback_summary["knowledge_state"])
        profile.cognitive_pattern = str(data.get("cognitive_pattern") or fallback_summary["cognitive_pattern"])
        profile.affective_pattern = str(data.get("affective_pattern") or fallback_summary["affective_pattern"])
        profile.social_pattern = str(data.get("social_pattern") or fallback_summary["social_pattern"])
        profile.higher_order_ability = str(data.get("higher_order_ability") or fallback_summary["higher_order_ability"])
        profile.main_difficulties = list(data.get("main_difficulties") or fallback_summary["main_difficulties"])
        profile.strengths = list(data.get("strengths") or fallback_summary["strengths"])
        profile.recommended_support = list(data.get("recommended_support") or fallback_summary["recommended_support"])
        profile.last_updated = str(data.get("last_updated") or fallback_summary["last_updated"])
        profile.drawer_summaries = dict(fallback_summary["drawer_summaries"])
        profile.llm_summary = (
            f"知识={profile.knowledge_state}; 认知={profile.cognitive_pattern}; "
            f"情感={profile.affective_pattern}; 社交={profile.social_pattern}; "
            f"高阶能力={profile.higher_order_ability}"
        )
        profile.llm_evidence = list(fallback_summary["evidence"])
        return profile

    def summarize_group(self, group_profile: GroupProfile, member_profiles: List[IndividualProfile], recent_notes: List[Dict[str, object]]) -> GroupProfile:
        fallback_summary = self._fallback_group_summary(group_profile, member_profiles)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "group_profile_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "string"},
                        "overall_knowledge_state": {"type": "string"},
                        "common_misconceptions": {"type": "array", "items": {"type": "string"}},
                        "discussion_quality": {"type": "string"},
                        "collaboration_pattern": {"type": "string"},
                        "affective_climate": {"type": "string"},
                        "participation_balance": {"type": "string"},
                        "higher_order_ability_level": {"type": "string"},
                        "main_group_risks": {"type": "array", "items": {"type": "string"}},
                        "recommended_group_support": {"type": "array", "items": {"type": "string"}},
                        "last_updated": {"type": "string"},
                    },
                    "required": [
                        "group_id",
                        "overall_knowledge_state",
                        "common_misconceptions",
                        "discussion_quality",
                        "collaboration_pattern",
                        "affective_climate",
                        "participation_balance",
                        "higher_order_ability_level",
                        "main_group_risks",
                        "recommended_group_support",
                        "last_updated",
                    ],
                    "additionalProperties": False,
                },
            },
        }
        prompt = (
            "你正在总结一个 CSCL 小组的可解释协作档案。\n"
            "只返回 JSON。\n"
            f"小组档案: {json.dumps(group_profile.to_dict(), ensure_ascii=False)}\n"
            f"成员画像: {json.dumps([profile.to_dict() for profile in member_profiles], ensure_ascii=False)}\n"
            f"近期 notes: {json.dumps(recent_notes, ensure_ascii=False)}\n"
            "请输出结构化小组画像，覆盖整体 C 语言理解水平、讨论质量、共同误区、协作结构、情感氛围、参与均衡、高阶能力和整体干预方向。"
        )
        try:
            completion = self.llm.get_completion(prompt=prompt, response_format=response_format)
            data = coerce_structured_output(completion, response_format)
        except Exception:
            data = {}
        group_profile.group_id = str(data.get("group_id") or fallback_summary["group_id"])
        group_profile.overall_knowledge_state = str(data.get("overall_knowledge_state") or fallback_summary["overall_knowledge_state"])
        group_profile.common_misconceptions = list(data.get("common_misconceptions") or fallback_summary["common_misconceptions"])
        group_profile.discussion_quality = str(data.get("discussion_quality") or fallback_summary["discussion_quality"])
        group_profile.collaboration_pattern = str(data.get("collaboration_pattern") or fallback_summary["collaboration_pattern"])
        group_profile.affective_climate = str(data.get("affective_climate") or fallback_summary["affective_climate"])
        group_profile.participation_balance = str(data.get("participation_balance") or fallback_summary["participation_balance"])
        group_profile.higher_order_ability_level = str(data.get("higher_order_ability_level") or fallback_summary["higher_order_ability_level"])
        group_profile.main_group_risks = list(data.get("main_group_risks") or fallback_summary["main_group_risks"])
        group_profile.recommended_group_support = list(data.get("recommended_group_support") or fallback_summary["recommended_group_support"])
        group_profile.last_updated = str(data.get("last_updated") or fallback_summary["last_updated"])
        group_profile.llm_summary = (
            f"知识={group_profile.overall_knowledge_state}; 讨论={group_profile.discussion_quality}; "
            f"协作={group_profile.collaboration_pattern}; 情感={group_profile.affective_climate}; "
            f"高阶能力={group_profile.higher_order_ability_level}"
        )
        group_profile.llm_evidence = list(fallback_summary["evidence"])
        return group_profile

    def _fallback_individual_summary(self, profile: IndividualProfile, box: StudentNoteBox) -> Dict[str, object]:
        drawer_summaries = {
            drawer_id: f"{drawer_id} 抽屉包含 {len(drawer.note_ids)} 条 notes"
            for drawer_id, drawer in box.drawers.items()
        }
        recent_bloom = [item.get("bloom_level", "") for item in profile.bloom_trajectory[-5:] if item.get("bloom_level")]
        recent_cognitive = [
            item.get("category", item.get("error_category", "Unknown"))
            for item in profile.cognitive_anomaly_trajectory[-5:]
        ]
        recent_affective = [item.get("affective_state", "") for item in profile.affective_state_trajectory[-5:] if item.get("affective_state")]
        recent_social = [item.get("social_mode", "") for item in profile.social_participation_trajectory[-5:] if item.get("social_mode")]
        high_order_count = sum(1 for level in recent_bloom if level in {"Analyze", "Evaluate", "Create"})
        difficulties = [item for item, count in sorted(profile.anomaly_counts.items(), key=lambda pair: pair[1], reverse=True) if item != "无明显问题"][:3]
        if not difficulties:
            difficulties = ["暂未发现稳定高频认知困难"]
        strengths: List[str] = []
        if high_order_count:
            strengths.append("能够出现分析、评价或创造层面的高阶加工")
        if any(tag in {"回应同伴", "促进讨论", "协作支持"} for tag in recent_social):
            strengths.append("具备一定同伴回应和协作支持行为")
        if not strengths:
            strengths.append("能够持续留下可诊断的学习轨迹")
        support = ["结合代码、命令和运行结果进行证据化解释"]
        if any(tag in {"焦虑", "挫败", "困惑"} for tag in recent_affective):
            support.append("先给予情绪确认，再用小步追问降低表达压力")
        if any(tag in {"沉默观察", "忽视同伴"} for tag in recent_social):
            support.append("通过点名邀请和同伴复述提升参与度")
        knowledge_state = f"近期 Bloom 层次主要为 {recent_bloom or ['暂无']}，C 语言知识掌握仍需结合具体发言继续判断。"
        cognitive_pattern = f"近期认知问题集中在 {recent_cognitive or ['暂无明显问题']}。"
        affective_pattern = f"近期情感状态表现为 {recent_affective or ['暂无稳定趋势']}。"
        social_pattern = f"近期协作互动表现为 {recent_social or ['暂无稳定模式']}。"
        higher_order_ability = "已出现高阶加工迹象。" if high_order_count else "当前主要停留在记忆、理解或应用层面，高阶分析证据仍不足。"
        evidence = profile.note_ids[-5:]
        return {
            "student_id": profile.student_id,
            "name": profile.name,
            "knowledge_state": knowledge_state,
            "cognitive_pattern": cognitive_pattern,
            "affective_pattern": affective_pattern,
            "social_pattern": social_pattern,
            "higher_order_ability": higher_order_ability,
            "main_difficulties": difficulties,
            "strengths": strengths,
            "recommended_support": support,
            "last_updated": profile.last_updated or "未更新",
            "evidence": evidence,
            "drawer_summaries": drawer_summaries,
        }

    def _fallback_group_summary(self, group_profile: GroupProfile, member_profiles: List[IndividualProfile]) -> Dict[str, object]:
        recent_modes = [item.get("collaboration_mode", "") for item in group_profile.collaboration_mode_trajectory[-5:]]
        recent_conflicts = [item.get("interaction_flaw", "") for item in group_profile.conflict_event_trajectory[-3:] if item.get("interaction_flaw")]
        recent_imbalances = [item.get("imbalance", 0) for item in group_profile.participation_balance_trajectory[-3:]]
        recent_risks_nested = [item.get("risk_flags", []) for item in group_profile.group_risk_trajectory[-3:]]
        recent_risks = sorted({risk for risks in recent_risks_nested for risk in risks})
        misconception_counts: Dict[str, int] = {}
        high_order_count = 0
        affective_tags: List[str] = []
        for member in member_profiles:
            for category, count in member.anomaly_counts.items():
                if category != "无明显问题":
                    misconception_counts[category] = misconception_counts.get(category, 0) + count
            high_order_count += sum(1 for item in member.bloom_trajectory[-5:] if item.get("bloom_level") in {"Analyze", "Evaluate", "Create"})
            affective_tags.extend(item.get("affective_state", "") for item in member.affective_state_trajectory[-3:])
        common_misconceptions = [item for item, _ in sorted(misconception_counts.items(), key=lambda pair: pair[1], reverse=True)[:3]]
        if not common_misconceptions:
            common_misconceptions = ["暂未形成稳定共同误区"]
        max_imbalance = max(recent_imbalances) if recent_imbalances else 0
        support = ["引导小组用同一段 C 代码、命令和数据流图共同验证观点"]
        if max_imbalance >= 3:
            support.append("分配轮流解释机会，优先邀请低参与成员表达")
        if recent_conflicts:
            support.append("要求成员先复述同伴观点，再提出修正或反例")
        return {
            "group_id": group_profile.group_id,
            "overall_knowledge_state": "小组已有若干 C 语言输入输出概念线索，但仍需通过证据化讨论整合成稳定模型。",
            "common_misconceptions": common_misconceptions,
            "discussion_quality": f"近期协作模式为 {recent_modes or ['暂无']}，讨论质量仍需结合证据链提升。",
            "collaboration_pattern": group_profile.collaboration_mode,
            "affective_climate": f"近期情感状态包括 {affective_tags[-6:] or ['暂无稳定趋势']}。",
            "participation_balance": "参与不均衡风险较高。" if max_imbalance >= 3 else "参与暂未出现严重不均衡。",
            "higher_order_ability_level": "已有高阶分析迹象。" if high_order_count else "整体高阶能力证据不足，仍以理解和应用层讨论为主。",
            "main_group_risks": recent_risks or ["暂无显著小组风险"],
            "recommended_group_support": support,
            "last_updated": group_profile.last_updated or "未更新",
            "evidence": group_profile.note_ids[-8:],
        }
