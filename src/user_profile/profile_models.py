"""
用户画像数据模型

定义画像的领域结构、事实格式和完整画像模型。
基于 schema.md 中的 14 个领域定义。
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DomainCategory(str, Enum):
    """领域类别"""
    BASELINE = "baseline"               # 基线信息，相对稳定
    DYNAMIC = "dynamic"                 # 动态信息，随咨询进程变化
    EMOTIONAL_WORLD = "emotional_world" # 情感世界
    COMMUNICATION = "communication"     # 沟通风格


class FactType(str, Enum):
    """事实类型"""
    OBSERVATION = "observation"           # 咨询师直接观察到的事实
    SELF_REPORT = "self_report"           # TA 自我报告的内容
    INFERENCE = "inference"               # 基于证据的推断
    PATTERN = "pattern"                   # 识别出的重复模式
    RISK = "risk"                         # 风险评估发现
    TREATMENT_RESPONSE = "treatment_response"  # 干预效果观察


class Fact(BaseModel):
    """单条事实记录"""
    id: str = Field(description="全局唯一 ID，如 fact_001")
    type: FactType = Field(description="事实类型")
    statement: str = Field(description="事实陈述")
    evidence: list[str] = Field(default_factory=list, description="来源会话 ID 列表")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度 0.0-1.0")
    relates_to: list[str] = Field(default_factory=list, description="关联的其他事实 ID")


class DomainInfo(BaseModel):
    """领域定义（静态元数据）"""
    domain_id: int = Field(description="领域编号 1-14")
    name: str = Field(description="领域名称")
    category: DomainCategory = Field(description="所属类别")
    description: str = Field(description="领域说明")
    sub_fields: list[str] = Field(default_factory=list, description="子字段列表")


# ===== 14 个领域注册表 =====
DOMAIN_REGISTRY: list[DomainInfo] = [
    DomainInfo(
        domain_id=1, name="主诉与现状", category=DomainCategory.DYNAMIC,
        description="TA 目前最大的困扰是什么？",
        sub_fields=["presenting_problem", "symptom_profile", "functional_impact", "treatment_history", "stage_assessment"],
    ),
    DomainInfo(
        domain_id=2, name="成长与发展史", category=DomainCategory.BASELINE,
        description="什么经历塑造了今天的 TA？",
    ),
    DomainInfo(
        domain_id=3, name="易感因素", category=DomainCategory.BASELINE,
        description="什么让 TA 对当前问题更脆弱？",
    ),
    DomainInfo(
        domain_id=4, name="诱发因素", category=DomainCategory.DYNAMIC,
        description="为什么是现在来求助？",
    ),
    DomainInfo(
        domain_id=5, name="维持因素", category=DomainCategory.DYNAMIC,
        description="什么在让问题持续？",
    ),
    DomainInfo(
        domain_id=6, name="保护因素", category=DomainCategory.DYNAMIC,
        description="什么在支撑着 TA？",
    ),
    DomainInfo(
        domain_id=7, name="关系模式", category=DomainCategory.BASELINE,
        description="TA 如何与重要他人互动？",
    ),
    DomainInfo(
        domain_id=8, name="干预反应", category=DomainCategory.DYNAMIC,
        description="什么对 TA 有效 / 无效？",
        sub_fields=["effective_interventions", "ineffective_or_harmful", "alliance_quality", "treatment_plan_alignment"],
    ),
    DomainInfo(
        domain_id=9, name="风险评估", category=DomainCategory.DYNAMIC,
        description="TA 有哪些需要警惕的风险？",
        sub_fields=["self_harm_risk", "suicide_risk", "violence_risk", "dropout_risk", "crisis_history"],
    ),
    DomainInfo(
        domain_id=10, name="文化/背景因素", category=DomainCategory.BASELINE,
        description="文化、家庭、社会如何影响 TA？",
    ),
    DomainInfo(
        domain_id=11, name="人格印象", category=DomainCategory.BASELINE,
        description="TA 是个什么样的人？",
    ),
    DomainInfo(
        domain_id=12, name="情感世界", category=DomainCategory.EMOTIONAL_WORLD,
        description="TA 的深层渴望与恐惧是什么？",
    ),
    DomainInfo(
        domain_id=13, name="沟通风格", category=DomainCategory.COMMUNICATION,
        description="TA 习惯如何表达和接收？",
        sub_fields=["language_style", "metaphor_preference", "pacing_preference", "response_receptivity"],
    ),
    DomainInfo(
        domain_id=14, name="求助与改变模式", category=DomainCategory.DYNAMIC,
        description="TA 如何面对改变？",
        sub_fields=["change_stage", "help_seeking_style", "resistance_pattern", "counseling_expectation"],
    ),
]


def get_domain_info(domain_id: int) -> Optional[DomainInfo]:
    """按编号获取领域定义"""
    for d in DOMAIN_REGISTRY:
        if d.domain_id == domain_id:
            return d
    return None


def get_domains_by_category(category: DomainCategory) -> list[DomainInfo]:
    """按类别获取领域列表"""
    return [d for d in DOMAIN_REGISTRY if d.category == category]


class ProfileDomain(BaseModel):
    """单个领域的内容"""
    summary: str = Field(default="", description="一句话摘要")
    narrative: str = Field(default="", description="详细叙事描述")
    facts: list[Fact] = Field(default_factory=list, description="事实列表")
    last_updated: str = Field(default="", description="最后更新时间 ISO 格式")


class Profile(BaseModel):
    """完整用户画像"""
    user_id: str = Field(default="", description="用户 ID")
    version: int = Field(default=0, description="版本号，从 0 开始，每次更新 +1")
    created_at: str = Field(default="", description="创建时间 ISO 格式")
    updated_at: str = Field(default="", description="最后更新时间 ISO 格式")
    version_history: list[dict] = Field(
        default_factory=list,
        description="版本历史，每条记录包含 version/changed_domains/timestamp/source_session",
    )
    source_sessions: list[str] = Field(default_factory=list, description="提供数据的会话 ID 列表")
    domains: dict[int, ProfileDomain] = Field(
        default_factory=dict,
        description="领域内容，key 为 domain_id (1-14)",
    )

    def get_domain(self, domain_id: int) -> Optional[ProfileDomain]:
        """获取指定领域的内容"""
        return self.domains.get(domain_id)

    def is_empty(self) -> bool:
        """检查画像是否为空（尚未初始化）"""
        return len(self.domains) == 0 or all(
            d.summary == "" for d in self.domains.values()
        )


class ScreeningOutput(BaseModel):
    """更新筛选输出：决定哪些领域需要更新"""
    domains_to_update: list[int] = Field(description="需要更新的领域编号列表（1-14）")
    reason: str = Field(description="判断理由")
