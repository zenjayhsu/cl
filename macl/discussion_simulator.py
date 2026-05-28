from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List
import json
import os
import random

from config import LLMRuntimeConfig
from schemas.cscl import StudentUtterance
from services.llm_controller import LLMController
from services.runtime_logging import get_logger
from services.structured_output import coerce_structured_output, has_meaningful_value

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPIC = "C语言课程：命令行、标准输入输出与重定向"
logger = get_logger("simulator")


def extract_topic_keywords(topic: str) -> List[str]:
    topic_lower = topic.lower()
    keyword_map = {
        "c语言": ["命令行参数", "标准输入", "标准输出", "重定向", "管道"],
        "c 程序": ["命令行参数", "stdin", "stdout", "stderr", "shell重定向"],
        "命令行": ["argc/argv", "stdin", "stdout", "stderr", "shell重定向"],
        "标准输入输出": ["stdin", "stdout", "stderr", "缓冲区", "EOF"],
        "重定向": ["输入重定向", "输出重定向", "错误重定向", "管道", "文件描述符"],
        "c language": ["command-line arguments", "stdin", "stdout", "redirection", "pipes"],
    }
    for key, keywords in keyword_map.items():
        if key in topic_lower:
            return keywords
    return ["命令行参数", "标准输入", "标准输出", "标准错误", "重定向"]


def build_topic_knowledge_map(topic: str) -> Dict[str, List[str]]:
    keywords = extract_topic_keywords(topic)
    return {
        "key_concepts": keywords,
        "evidence_points": [
            "main(int argc, char *argv[]) 接收命令行参数，argv[0] 通常是程序名",
            "scanf/fgets 从 stdin 读取数据，不等同于命令行参数 argv",
            "printf 默认写入 stdout，fprintf(stderr, ...) 写入 stderr",
            "shell 的 <、>、2>、| 改变进程的标准流连接方式，而不是修改 C 源代码逻辑",
        ],
        "controversial_points": [
            "什么时候用命令行参数，什么时候用标准输入",
            "printf 调试信息是否应该输出到 stderr",
            "重定向和在程序里 fopen 文件之间的边界",
            "管道连接多个程序时 EOF 和缓冲行为如何影响结果",
        ],
        "common_misconceptions": [
            "把命令行参数 argv 和标准输入 stdin 混为一谈",
            "认为重定向会改变 C 程序源码中的输入输出语句",
            "认为 printf、fprintf(stdout, ...) 和 fprintf(stderr, ...) 在重定向时完全一样",
        ],
    }


@dataclass
class StudentHiddenTraits:
    knowledge_level: str
    misconception_biases: List[str]
    confidence: float
    motivation: float
    anxiety: float
    dominance: float
    agreeableness: float
    criticality: float
    confusion_tolerance: float
    help_seeking: float

    def to_prompt_text(self) -> str:
        return (
            f"知识水平={self.knowledge_level}; "
            f"误解倾向={self.misconception_biases}; "
            f"自信={self.confidence:.2f}; 动机={self.motivation:.2f}; "
            f"焦虑={self.anxiety:.2f}; 支配性={self.dominance:.2f}; "
            f"宜人性={self.agreeableness:.2f}; 批判性={self.criticality:.2f}; "
            f"困惑耐受={self.confusion_tolerance:.2f}; 求助倾向={self.help_seeking:.2f}"
        )


@dataclass
class StudentDynamicState:
    confusion: float = 0.2
    frustration: float = 0.1
    engagement: float = 0.75
    trust_in_group: float = 0.65
    speaking_count: int = 0
    turns_since_spoke: int = 0
    last_intervention_received: str = ""
    corrected_misconceptions: List[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        return (
            f"困惑={self.confusion:.2f}; 挫败={self.frustration:.2f}; "
            f"投入={self.engagement:.2f}; 小组信任={self.trust_in_group:.2f}; "
            f"发言次数={self.speaking_count}; 距上次发言轮数={self.turns_since_spoke}; "
            f"最近收到的干预={self.last_intervention_received or '无'}; "
            f"已修正误解={self.corrected_misconceptions[-3:]}"
        )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class StudentPersona:
    student_id: str
    name: str
    age: int
    gender: str
    major: str
    learning_level_label: str
    personality: str
    learning_style: str
    emotional_motivation_state: str
    profile: Dict[str, object]
    system_prompt: str
    hidden_traits: StudentHiddenTraits


def build_virtual_profile(persona: StudentPersona, topic: str) -> Dict[str, object]:
    return {
        "学生编号": persona.student_id,
        "姓名": persona.name,
        "性别": persona.gender,
        "年龄": persona.age,
        "专业": persona.major,
        "能力水平": persona.learning_level_label,
        "性格特征": persona.personality,
        "学习风格": persona.learning_style,
        "情感与动机状态": persona.emotional_motivation_state,
    }


GENDER_NAMES = {
    "男": ["周泽", "刘洋", "张睿", "陈一鸣", "许晨", "高远"],
    "女": ["林悦", "孙琳", "吴佳", "何雨桐", "沈安琪", "邱然"],
}

MAJORS = ["计算机科学与技术", "软件工程", "人工智能", "网络空间安全", "数据科学与大数据技术"]
ABILITY_LEVELS = ["高水平学生", "中水平学生", "低水平学生"]
PERSONALITY_TYPES = ["积极主动型", "沉默观察型", "谨慎保守型", "自信表达型", "合作支持型"]
LEARNING_STYLES = ["逻辑分析型", "实践操作型", "模仿迁移型", "探究反思型", "被动接受型"]
MOTIVATION_STATES = ["高动机", "中动机", "低动机", "焦虑型", "成长型"]

def _sample_cycle(rng: random.Random, values: List[str], count: int) -> List[str]:
    result: List[str] = []
    pool: List[str] = []
    while len(result) < count:
        if not pool:
            pool = list(values)
            rng.shuffle(pool)
        result.append(pool.pop())
    return result


def _build_student_name(rng: random.Random, gender: str, used_names: set[str]) -> str:
    candidates = list(GENDER_NAMES[gender])
    rng.shuffle(candidates)
    for candidate in candidates:
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
    fallback = f"{gender}生{len(used_names) + 1}"
    used_names.add(fallback)
    return fallback


def _ability_traits(ability: str) -> Dict[str, float | str]:
    if ability == "高水平学生":
        return {
            "knowledge_level": "high",
            "confidence": 0.78,
            "motivation": 0.78,
            "anxiety": 0.18,
            "confusion_tolerance": 0.76,
            "help_seeking": 0.35,
        }
    if ability == "低水平学生":
        return {
            "knowledge_level": "low",
            "confidence": 0.38,
            "motivation": 0.48,
            "anxiety": 0.58,
            "confusion_tolerance": 0.36,
            "help_seeking": 0.72,
        }
    return {
        "knowledge_level": "medium",
        "confidence": 0.56,
        "motivation": 0.62,
        "anxiety": 0.36,
        "confusion_tolerance": 0.55,
        "help_seeking": 0.55,
    }


def _personality_traits(personality: str) -> Dict[str, float | str]:
    profiles = {
        "积极主动型": {"dominance": 0.62, "agreeableness": 0.68, "speech_hint": "愿意主动发言和提问，会推动小组继续往下讨论。"},
        "沉默观察型": {"dominance": 0.24, "agreeableness": 0.58, "speech_hint": "发言较少，但会认真听同伴观点，被点到时能说出自己的思考。"},
        "谨慎保守型": {"dominance": 0.32, "agreeableness": 0.62, "speech_hint": "表达前会反复确认，常用不确定语气，害怕把同伴带偏。"},
        "自信表达型": {"dominance": 0.78, "agreeableness": 0.42, "speech_hint": "表达欲强，喜欢直接给判断，但有时会忽略同伴的疑问。"},
        "合作支持型": {"dominance": 0.48, "agreeableness": 0.82, "speech_hint": "会倾听同伴并鼓励别人补充，善于把讨论拉回共同任务。"},
    }
    return profiles[personality]


def _learning_style_traits(style: str) -> Dict[str, float | str]:
    profiles = {
        "逻辑分析型": {"criticality": 0.68, "learning_hint": "倾向先澄清概念、规则和执行流程，再判断程序行为。"},
        "实践操作型": {"criticality": 0.46, "learning_hint": "更相信运行结果，喜欢通过写代码和调试来确认理解。"},
        "模仿迁移型": {"criticality": 0.38, "learning_hint": "会先套用例题或模板，遇到变式任务时容易犹豫。"},
        "探究反思型": {"criticality": 0.72, "learning_hint": "喜欢追问为什么，也会主动总结错误原因和规律。"},
        "被动接受型": {"criticality": 0.28, "learning_hint": "主要等待同伴解释，需要明确提示后才会继续探索。"},
    }
    return profiles[style]


def _motivation_traits(state: str) -> Dict[str, float]:
    profiles = {
        "高动机": {"motivation_delta": 0.16, "anxiety_delta": -0.04},
        "中动机": {"motivation_delta": 0.0, "anxiety_delta": 0.0},
        "低动机": {"motivation_delta": -0.18, "anxiety_delta": 0.08},
        "焦虑型": {"motivation_delta": -0.04, "anxiety_delta": 0.22},
        "成长型": {"motivation_delta": 0.10, "anxiety_delta": 0.04},
    }
    return profiles[state]


def _topic_background(ability: str, learning_style: str, knowledge_map: Dict[str, List[str]]) -> tuple[str, List[str], List[str]]:
    concepts = knowledge_map["key_concepts"]
    misconceptions = knowledge_map["common_misconceptions"]
    if ability == "高水平学生":
        prior = [f"较熟悉{concepts[0]}", f"能解释{concepts[1]}和{concepts[2]}的基本区别", "愿意用例子帮助同伴"]
        weaknesses = [misconceptions[-1], "解释时可能跳过同伴尚未理解的中间步骤"]
        background = f"基础较扎实，已经接触过{concepts[0]}、{concepts[1]}和{concepts[2]}，能主动解释概念。"
    elif ability == "低水平学生":
        prior = [f"听说过{concepts[0]}", "能照着例题运行简单 C 程序"]
        weaknesses = misconceptions[:2]
        background = f"C 语言基础薄弱，对{concepts[0]}、{concepts[1]}等概念还不稳定，需要更多提示。"
    else:
        prior = [f"做过{concepts[0]}相关练习", f"知道{concepts[1]}的一些表面用法"]
        weaknesses = misconceptions[:1] + ["知识点比较零散，遇到变式任务时容易混淆"]
        background = f"具备一定 C 语言基础，但对{concepts[0]}、{concepts[1]}、{concepts[2]}之间的联系还不系统。"
    if learning_style == "实践操作型":
        prior.append("喜欢通过运行结果确认判断")
    elif learning_style == "逻辑分析型":
        prior.append("习惯先画出概念关系或执行流程")
    elif learning_style == "被动接受型":
        weaknesses.append("主动探索不足，容易等待同伴直接解释")
    return background, weaknesses, prior


def _build_hidden_traits(
    ability: str,
    personality: str,
    learning_style: str,
    motivation_state: str,
    misconception_biases: List[str],
) -> StudentHiddenTraits:
    ability_profile = _ability_traits(ability)
    personality_profile = _personality_traits(personality)
    style_profile = _learning_style_traits(learning_style)
    motivation_profile = _motivation_traits(motivation_state)
    return StudentHiddenTraits(
        knowledge_level=str(ability_profile["knowledge_level"]),
        misconception_biases=misconception_biases,
        confidence=_clamp(float(ability_profile["confidence"])),
        motivation=_clamp(float(ability_profile["motivation"]) + motivation_profile["motivation_delta"]),
        anxiety=_clamp(float(ability_profile["anxiety"]) + motivation_profile["anxiety_delta"]),
        dominance=_clamp(float(personality_profile["dominance"])),
        agreeableness=_clamp(float(personality_profile["agreeableness"])),
        criticality=_clamp(float(style_profile["criticality"])),
        confusion_tolerance=_clamp(float(ability_profile["confusion_tolerance"])),
        help_seeking=_clamp(float(ability_profile["help_seeking"])),
    )


def _select_misconceptions(ability: str, misconceptions: List[str], rng: random.Random) -> List[str]:
    if ability == "高水平学生":
        return rng.sample(misconceptions, k=1) if misconceptions and rng.random() < 0.45 else []
    if ability == "低水平学生":
        return rng.sample(misconceptions, k=min(2, len(misconceptions)))
    return rng.sample(misconceptions, k=1) if misconceptions else []


def build_four_personas(topic: str, template_text: str, rng: random.Random | None = None, count: int = 4) -> List[StudentPersona]:
    del template_text
    rng = rng or random.Random(int(os.getenv("CSCL_SIM_RANDOM_SEED", "42")))
    knowledge_map = build_topic_knowledge_map(topic)
    misconceptions = knowledge_map["common_misconceptions"]
    genders = _sample_cycle(rng, ["男", "女"], count)
    abilities = _sample_cycle(rng, ["高水平学生", "中水平学生", "低水平学生", "中水平学生"], count)
    personalities = _sample_cycle(rng, PERSONALITY_TYPES, count)
    learning_styles = _sample_cycle(rng, LEARNING_STYLES, count)
    motivation_states = _sample_cycle(rng, MOTIVATION_STATES, count)
    majors = _sample_cycle(rng, MAJORS, count)

    personas: List[StudentPersona] = []
    used_names: set[str] = set()
    for index in range(count):
        gender = genders[index]
        ability = abilities[index]
        personality = personalities[index]
        learning_style = learning_styles[index]
        motivation_state = motivation_states[index]
        personality_profile = _personality_traits(personality)
        style_profile = _learning_style_traits(learning_style)
        misconception_biases = _select_misconceptions(ability, misconceptions, rng)
        current_background, weaknesses, prior_knowledge = _topic_background(ability, learning_style, knowledge_map)
        persona = StudentPersona(
            student_id=f"s{index + 1}",
            name=_build_student_name(rng, gender, used_names),
            age=rng.randint(18, 25),
            gender=gender,
            major=majors[index],
            learning_level_label=ability,
            personality=personality,
            learning_style=learning_style,
            emotional_motivation_state=motivation_state,
            profile={},
            system_prompt="",
            hidden_traits=_build_hidden_traits(
                ability=ability,
                personality=personality,
                learning_style=learning_style,
                motivation_state=motivation_state,
                misconception_biases=misconception_biases,
            ),
        )
        profile_dict = build_virtual_profile(persona, topic)
        persona.profile = profile_dict
        profile_json = json.dumps(profile_dict, ensure_ascii=False, indent=2)
        prompt = (
            "你正在扮演一名真实的计算机类专业学生，而不是教师或助手。\n"
            f"学生姓名: {persona.name}\n"
            f"学生编号: {persona.student_id}\n"
            f"讨论主题: {topic}\n"
            "课程场景: C语言程序设计小组协作学习。\n"
            f"学生画像(JSON):\n{profile_json}\n"
            f"当前主题相关基础: {current_background}\n"
            f"可能薄弱点: {json.dumps(weaknesses, ensure_ascii=False)}\n"
            f"已有知识: {json.dumps(prior_knowledge, ensure_ascii=False)}\n"
            f"内部发言提示: {personality_profile['speech_hint']}\n"
            f"内部学习提示: {style_profile['learning_hint']}\n"
            "发言要求:\n"
            "1. 必须使用中文，语气像真实学生在小组中发言。\n"
            "2. 不要表现得全知全能，要受自己的学习水平、性格、学习风格和薄弱点影响。\n"
            "3. 可以自然出现困惑、误解、跳步推理、沉默后的试探性发言或对同伴观点的回应。\n"
            "4. 如果收到 Meta-Agent 干预，要尝试修正概念、调整情绪或改善协作方式。\n"
        )
        persona.system_prompt = prompt
        personas.append(persona)
    return personas


def _truncate_words(text: str, limit: int = 100) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


@dataclass
class StudentAgent:
    persona: StudentPersona
    llm: LLMController
    topic: str

    def _fallback_response(self, stage: str, teacher_feedback: str, dynamic_state: StudentDynamicState) -> Dict[str, str]:
        topic_phrase = self.topic.strip() or "当前议题"
        feedback_hint = teacher_feedback.strip() if teacher_feedback.strip() else "当前讨论"
        if dynamic_state.last_intervention_received:
            message = f"我觉得刚才的反馈提醒我先慢下来。关于{topic_phrase}，我需要重新区分几个核心概念分别在什么条件下起作用。"
        elif dynamic_state.confusion > self.persona.hidden_traits.confusion_tolerance:
            message = f"我对{topic_phrase}还不是很确定，可能把相近概念混在一起了，需要有人用一段 C 代码和运行现象帮我对照一下。"
        else:
            message = f"我觉得{topic_phrase}可以先按三个问题推进：概念是什么、代码里怎么体现、运行结果能不能验证。"
        return {
            "message": message,
            "self-regulation": "monitoring",
            "reason for self-regulation": f"模型输出格式异常，因此使用安全兜底发言，保持{stage}阶段讨论连贯。",
            "co-regulation": "prompt_peer_elaboration",
            "reason for co-regulation": f"下一轮应重新连接{feedback_hint}，并邀请同伴继续补充。",
        }

    def generate(
        self,
        history: List[Dict[str, object]],
        teacher_feedback: str,
        stage: str,
        dynamic_state: StudentDynamicState,
        topic_knowledge_map: Dict[str, List[str]],
    ) -> Dict[str, str]:
        logger.info(
            "Student generation start | student=%s | stage=%s | history_items=%s",
            self.persona.student_id,
            stage,
            len(history),
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "student_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "self-regulation": {"type": "string"},
                        "reason for self-regulation": {"type": "string"},
                        "co-regulation": {"type": "string"},
                        "reason for co-regulation": {"type": "string"},
                    },
                    "required": [
                        "message",
                        "self-regulation",
                        "reason for self-regulation",
                        "co-regulation",
                        "reason for co-regulation",
                    ],
                    "additionalProperties": False,
                },
            },
        }
        history_text = self._format_history(history)
        prompt = (
            f"{self.persona.system_prompt}\n\n"
            f"你是学生 {self.persona.name}，正在参加一个关于“{self.topic}”的协作讨论。\n"
            "你必须以该学生身份回答，不要以助手身份回答。\n"
            "请用中文发言。\n"
            "你不是全知的。请根据自己不完整的先验知识、隐藏特质和当前情绪状态来发言。\n"
            "如果知识有限，可以表现出局部推理、过度概括、不确定或符合自身特质的误解。\n"
            "如果你的社交风格偏批判或强势，可以质疑同伴，但要像真实学生讨论一样自然。\n"
            "如果当前困惑或挫败较高，发言可以体现困惑、犹豫或紧张。\n"
            "如果存在教师或 Meta-Agent 反馈，你应该尝试回应反馈，修正推理、调节情绪或改善协作。\n"
            f"当前讨论阶段: {stage}\n"
            f"教师 / Meta-Agent 反馈: {teacher_feedback or '无'}\n"
            f"议题知识图谱: {json.dumps(topic_knowledge_map, ensure_ascii=False)}\n"
            f"学生隐藏特质: {self.persona.hidden_traits.to_prompt_text()}\n"
            f"当前动态状态: {dynamic_state.to_prompt_text()}\n"
            f"对话历史:\n{history_text}\n\n"
            "只返回一个符合 schema 的 JSON 对象。\n"
            "不要在 JSON 外添加 Markdown、代码块或解释。\n"
            "message 字段控制在 100 个中文词以内，并让它听起来像真实学生在小组中发言。"
        )
        completion = self.llm.get_completion(prompt=prompt, response_format=response_format)
        data = coerce_structured_output(completion, response_format)
        if has_meaningful_value(data, ("message",)):
            data["message"] = _truncate_words(str(data["message"]), 100)
            logger.info(
                "Student generation success | student=%s | stage=%s | chars=%s",
                self.persona.student_id,
                stage,
                len(data["message"]),
            )
            return {
                "message": str(data.get("message", "")),
                "self-regulation": str(data.get("self-regulation", "")),
                "reason for self-regulation": str(data.get("reason for self-regulation", "")),
                "co-regulation": str(data.get("co-regulation", "")),
                "reason for co-regulation": str(data.get("reason for co-regulation", "")),
            }

        repair_prompt = (
            f"{prompt}\n\n"
            "你上一次回答无效或为空。\n"
            "只返回一个有效 JSON 对象，并填满所有必填字段。\n"
            "'message' 字段必须是一句自然的中文学生发言。"
        )
        repaired_completion = self.llm.get_completion(prompt=repair_prompt, response_format=response_format, temperature=0.0)
        repaired_data = coerce_structured_output(repaired_completion, response_format)
        if has_meaningful_value(repaired_data, ("message",)):
            repaired_data["message"] = _truncate_words(str(repaired_data["message"]), 100)
            logger.warning(
                "Student generation repaired | student=%s | stage=%s | chars=%s",
                self.persona.student_id,
                stage,
                len(repaired_data["message"]),
            )
            return {
                "message": str(repaired_data.get("message", "")),
                "self-regulation": str(repaired_data.get("self-regulation", "")),
                "reason for self-regulation": str(repaired_data.get("reason for self-regulation", "")),
                "co-regulation": str(repaired_data.get("co-regulation", "")),
                "reason for co-regulation": str(repaired_data.get("reason for co-regulation", "")),
            }

        fallback = self._fallback_response(stage=stage, teacher_feedback=teacher_feedback, dynamic_state=dynamic_state)
        fallback["reason for self-regulation"] += " [schema_fallback]"
        fallback["reason for co-regulation"] += " [schema_fallback]"
        logger.error(
            "Student generation fallback | student=%s | stage=%s",
            self.persona.student_id,
            stage,
        )
        return fallback

    def _format_history(self, history: List[Dict[str, object]]) -> str:
        lines: List[str] = []
        for item in history[-8:]:
            speaker = item.get("speaker_name") or item.get("student_id", "")
            text = item.get("message", "")
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)


@dataclass
class DiscussionSimulator:
    topic: str
    prompt_template: str
    llm_config: LLMRuntimeConfig
    personas: List[StudentPersona] = field(init=False)
    student_agents: Dict[str, StudentAgent] = field(init=False)
    dynamic_states: Dict[str, StudentDynamicState] = field(init=False)
    topic_knowledge_map: Dict[str, List[str]] = field(init=False)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        seed = int(os.getenv("CSCL_SIM_RANDOM_SEED", "42"))
        self.rng = random.Random(seed)
        self.topic_knowledge_map = build_topic_knowledge_map(self.topic)
        student_count = int(os.getenv("CSCL_STUDENT_COUNT", "4"))
        self.personas = build_four_personas(self.topic, self.prompt_template, rng=self.rng, count=student_count)
        llm = LLMController(
            backend=self.llm_config.backend,
            model=self.llm_config.model,
            base_url=self.llm_config.base_url,
            api_key=self.llm_config.api_key,
        )
        self.student_agents = {persona.student_id: StudentAgent(persona=persona, llm=llm, topic=self.topic) for persona in self.personas}
        self.dynamic_states = {persona.student_id: self._initial_dynamic_state(persona) for persona in self.personas}
        logger.info("Discussion simulator seed=%s", seed)

    def _initial_dynamic_state(self, persona: StudentPersona) -> StudentDynamicState:
        traits = persona.hidden_traits
        return StudentDynamicState(
            confusion=_clamp(0.35 - traits.confusion_tolerance * 0.2 + traits.anxiety * 0.2),
            frustration=_clamp(0.15 + traits.anxiety * 0.15 - traits.agreeableness * 0.08),
            engagement=_clamp(0.45 + traits.motivation * 0.45 - traits.anxiety * 0.1),
            trust_in_group=_clamp(0.45 + traits.agreeableness * 0.35 - traits.dominance * 0.08),
        )

    def _stage_affinity(self, persona: StudentPersona, stage: str) -> float:
        affinities = {
            "积极主动型": {"问题界定": 1.45, "概念澄清": 1.25, "样例推演": 1.1, "错误辨析": 1.0, "整合解释": 0.95, "迁移应用": 1.05, "反思总结": 1.0},
            "沉默观察型": {"问题界定": 0.75, "概念澄清": 0.9, "样例推演": 1.05, "错误辨析": 1.15, "整合解释": 0.95, "迁移应用": 1.0, "反思总结": 1.1},
            "谨慎保守型": {"问题界定": 0.85, "概念澄清": 1.05, "样例推演": 1.0, "错误辨析": 1.25, "整合解释": 1.05, "迁移应用": 0.95, "反思总结": 1.1},
            "自信表达型": {"问题界定": 1.1, "概念澄清": 1.25, "样例推演": 1.35, "错误辨析": 1.15, "整合解释": 1.0, "迁移应用": 1.1, "反思总结": 0.9},
            "合作支持型": {"问题界定": 0.95, "概念澄清": 1.0, "样例推演": 1.0, "错误辨析": 1.05, "整合解释": 1.45, "迁移应用": 1.15, "反思总结": 1.35},
        }
        return affinities.get(persona.personality, {}).get(stage, 1.0)

    def _student_was_mentioned(self, student_id: str, history: List[Dict[str, object]], teacher_feedback: str) -> bool:
        needle = student_id.lower()
        recent_text = " ".join(str(item.get("message", "")) for item in history[-4:]).lower()
        return needle in recent_text or needle in teacher_feedback.lower()

    def _infer_mentions(self, text: str, speaker_student_id: str) -> List[str]:
        mentions: List[str] = []
        lowered = text.lower()
        for persona in self.personas:
            if persona.student_id == speaker_student_id:
                continue
            if persona.student_id.lower() in lowered or persona.name in text:
                mentions.append(persona.student_id)
        return mentions

    def _select_next_persona(self, stage: str, history: List[Dict[str, object]], teacher_feedback: str) -> StudentPersona:
        weights: List[float] = []
        last_student = str(history[-1].get("student_id", "")) if history else ""
        min_speaking_count = min((state.speaking_count for state in self.dynamic_states.values()), default=0)

        for persona in self.personas:
            state = self.dynamic_states[persona.student_id]
            traits = persona.hidden_traits
            weight = 1.0
            weight *= self._stage_affinity(persona, stage)
            weight += state.engagement * 0.7
            weight += traits.motivation * 0.4
            weight += traits.dominance * 0.45
            weight += state.turns_since_spoke * 0.28
            weight += max(0, state.speaking_count - min_speaking_count) * -0.25

            if persona.student_id == last_student:
                weight *= 0.35
            if self._student_was_mentioned(persona.student_id, history, teacher_feedback):
                weight += 1.4
            if teacher_feedback and (persona.student_id in teacher_feedback or persona.name.lower() in teacher_feedback.lower()):
                weight += 1.8
            if state.confusion > traits.confusion_tolerance and traits.help_seeking > 0.5:
                weight += 0.75
            if state.frustration > 0.55 and traits.dominance > 0.55:
                weight += 0.6

            weights.append(max(0.05, weight))

        selected = self.rng.choices(self.personas, weights=weights, k=1)[0]
        logger.info(
            "Speaker selected | student=%s | stage=%s | weights=%s",
            selected.student_id,
            stage,
            {persona.student_id: round(weight, 3) for persona, weight in zip(self.personas, weights)},
        )
        return selected

    def _update_states_before_turn(self, selected_student_id: str) -> None:
        for student_id, state in self.dynamic_states.items():
            if student_id == selected_student_id:
                state.turns_since_spoke = 0
                state.speaking_count += 1
                state.engagement = _clamp(state.engagement + 0.03)
            else:
                state.turns_since_spoke += 1
                state.engagement = _clamp(state.engagement - 0.02)
                if state.turns_since_spoke >= 3:
                    state.frustration = _clamp(state.frustration + 0.05)
                    state.trust_in_group = _clamp(state.trust_in_group - 0.04)

    def _apply_intervention_effect(self, target_scope: str, content: str, speaker_student_id: str) -> None:
        if not content:
            return
        affected_ids = list(self.dynamic_states) if target_scope in {"group", "all"} else [target_scope]
        if target_scope in {"individual", speaker_student_id}:
            affected_ids = [speaker_student_id]

        for student_id in affected_ids:
            if student_id not in self.dynamic_states:
                continue
            state = self.dynamic_states[student_id]
            state.last_intervention_received = content[:120]
            state.confusion = _clamp(state.confusion - 0.18)
            state.frustration = _clamp(state.frustration - 0.16)
            state.engagement = _clamp(state.engagement + 0.12)
            state.trust_in_group = _clamp(state.trust_in_group + 0.08)
            content_lower = content.lower()
            correction_tokens = ["misconception", "error", "误解", "误区", "混淆", "修正", "概念", "代码", "运行"]
            if any(token in content_lower for token in correction_tokens):
                state.corrected_misconceptions.append(content[:80])

    def _apply_discussion_drift(self, speaker_student_id: str, response: Dict[str, str], intervention_content: str) -> None:
        speaker_state = self.dynamic_states[speaker_student_id]
        message = response.get("message", "").lower()
        confusion_tokens = ["not sure", "confused", "don't understand", "unclear", "mixing up", "不确定", "困惑", "不太懂", "不清楚", "混在一起", "搞混"]
        if any(token in message for token in confusion_tokens):
            speaker_state.confusion = _clamp(speaker_state.confusion + 0.12)
        else:
            speaker_state.confusion = _clamp(speaker_state.confusion - 0.04)
        frustration_tokens = ["wrong", "disagree", "doesn't make sense", "can't", "不对", "不同意", "说不通", "不可能", "错了"]
        if any(token in message for token in frustration_tokens):
            speaker_state.frustration = _clamp(speaker_state.frustration + 0.08)
        else:
            speaker_state.frustration = _clamp(speaker_state.frustration - 0.03)
        if not intervention_content:
            for state in self.dynamic_states.values():
                state.last_intervention_received = ""

    def run(
        self,
        workflow,
        max_turns: int | None = None,
        on_record: Callable[[Dict[str, object]], None] | None = None,
    ) -> List[Dict[str, object]]:
        history: List[Dict[str, object]] = []
        records: List[Dict[str, object]] = []
        teacher_feedback = ""
        if max_turns is None:
            max_turns = self.rng.randint(30, 50)
        stage_plan = [
            "问题界定",
            "概念澄清",
            "样例推演",
            "错误辨析",
            "整合解释",
            "迁移应用",
            "反思总结",
            "反思总结",
        ]

        for turn_index in range(max_turns):
            stage = self._stage_for_turn(turn_index=turn_index, max_turns=max_turns, stage_plan=stage_plan)
            persona = self._select_next_persona(stage=stage, history=history, teacher_feedback=teacher_feedback)
            self._update_states_before_turn(persona.student_id)
            agent = self.student_agents[persona.student_id]
            dynamic_state = self.dynamic_states[persona.student_id]
            logger.info(
                "Turn start | turn=%s/%s | student=%s | stage=%s",
                turn_index + 1,
                max_turns,
                persona.student_id,
                stage,
            )
            response = agent.generate(
                history=history,
                teacher_feedback=teacher_feedback,
                stage=stage,
                dynamic_state=dynamic_state,
                topic_knowledge_map=self.topic_knowledge_map,
            )
            response["dynamic_state_before_workflow"] = dynamic_state.to_prompt_text()
            response["hidden_traits"] = persona.hidden_traits.to_prompt_text()
            utterance = StudentUtterance(
                turn_id=f"t{turn_index + 1}",
                student_id=persona.student_id,
                speaker_name=persona.name,
                text=response["message"],
                timestamp=f"2026-04-23T09:{turn_index:02d}:00",
                mentions=self._infer_mentions(response["message"], persona.student_id),
                metadata=response,
            )
            logger.info(
                "Turn utterance ready | turn_id=%s | student=%s | preview=%s",
                utterance.turn_id,
                utterance.student_id,
                utterance.text[:80],
            )
            result = workflow.run_turn(utterance)
            record = {
                "turn": utterance.to_dict(),
                "persona": {
                    "student_id": persona.student_id,
                    "name": persona.name,
                    "gender": persona.gender,
                    "age": persona.age,
                    "major": persona.major,
                    "ability_level": persona.learning_level_label,
                    "personality": persona.personality,
                    "learning_style": persona.learning_style,
                    "emotional_motivation_state": persona.emotional_motivation_state,
                },
                "student_response": response,
                "written_notes": [note.to_dict() for note in result.get("written_notes", [])],
                "retrieved_notes": [note.to_dict() for note in result.get("retrieved_notes", [])],
                "decision": result["decision"].to_dict(),
                "intervention": result["intervention"].to_dict(),
                "individual_profile": result["individual_profile"].to_dict(),
                "group_profile": result["group_profile"].to_dict(),
                "dynamic_states_before_turn_update": {
                    student_id: state.to_prompt_text()
                    for student_id, state in self.dynamic_states.items()
                },
            }
            records.append(record)
            history.append(
                {
                    "speaker_name": persona.name,
                    "student_id": persona.student_id,
                    "message": response["message"],
                    "stage": stage,
                    "teacher_feedback": teacher_feedback,
                }
            )
            intervention_content = result["intervention"].content
            self._apply_discussion_drift(
                speaker_student_id=persona.student_id,
                response=response,
                intervention_content=intervention_content,
            )
            if result["intervention"].content:
                history.append(
                    {
                        "speaker_name": "Meta-Agent",
                        "student_id": "meta",
                        "message": result["intervention"].content,
                        "stage": "intervention",
                    }
                )
                teacher_feedback = result["intervention"].content
                self._apply_intervention_effect(
                    target_scope=result["intervention"].target_scope,
                    content=result["intervention"].content,
                    speaker_student_id=persona.student_id,
                )
                logger.info(
                    "Meta intervention emitted | turn_id=%s | target=%s | preview=%s",
                    utterance.turn_id,
                    result["intervention"].target_scope,
                    result["intervention"].content[:80],
                )
            else:
                teacher_feedback = ""
            record["dynamic_states_after_turn_update"] = {
                student_id: state.to_prompt_text()
                for student_id, state in self.dynamic_states.items()
            }
            if on_record:
                on_record(record)
            if "we have finished the discussion" in response["message"].lower():
                logger.info("Early stop triggered | turn_id=%s", utterance.turn_id)
                break
            logger.info("Turn complete | turn_id=%s", utterance.turn_id)
        return records

    def _stage_for_turn(self, turn_index: int, max_turns: int, stage_plan: List[str]) -> str:
        if max_turns <= len(stage_plan):
            return stage_plan[turn_index]
        progress = turn_index / max(max_turns - 1, 1)
        if progress < 0.12:
            return "问题界定"
        if progress < 0.28:
            return "概念澄清"
        if progress < 0.45:
            return "样例推演"
        if progress < 0.62:
            return "错误辨析"
        if progress < 0.78:
            return "整合解释"
        if progress < 0.90:
            return "迁移应用"
        return "反思总结"
