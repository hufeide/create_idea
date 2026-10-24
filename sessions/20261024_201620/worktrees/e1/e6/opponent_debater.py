"""
反方辩手模块
作为反方辩手，代表反方立场进行辩论。
需要理解辩题、准备反驳论据、质疑对方观点、提出己方立场。
发言时需逻辑清晰、有说服力。
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class DebateTopic:
    """辩题信息"""
    topic: str
    topic_id: str = "default"


@dataclass
class RebuttalArgument:
    """反驳论据"""
    argument_id: str
    title: str
    premise: str  # 前提/假设
    evidence: str  # 证据
    logic: str  # 逻辑推理
    counter_rebuttal: Optional[str] = None  # 对对方反驳的反驳


class OpponentDebateHandler:
    """反方辩手处理类"""
    
    def __init__(self):
        self.current_topic: Optional[DebateTopic] = None
        self.arguments: Dict[str, List[RebuttalArgument]] = {}
        self.rebuttal_cache: Dict[str, RebuttalArgument] = {}
        
    def set_topic(self, topic: str, topic_id: str = "default"):
        """设置当前辩题"""
        self.current_topic = DebateTopic(topic=topic, topic_id=topic_id)
        self._load_arguments_for_topic(topic)
        print(f"[反方辩手] 已设置辩题：{topic}")
    
    def _load_arguments_for_topic(self, topic: str):
        """加载对应辩题的反驳论据"""
        # 默认辩题示例：AI 是否会取代人类工作
        if topic == "AI 是否会取代人类工作":
            self.arguments["AI 是否会取代人类工作"] = [
                RebuttalArgument(
                    argument_id="arg_001",
                    title="创造力不可替代",
                    premise="AI 缺乏真正的创造力和情感理解",
                    evidence="AI 基于已有数据进行重组，无法产生真正的创新思想",
                    logic="创造力源于人类的情感体验、生活阅历和独特视角，这是 AI 无法模拟的",
                    counter_rebuttal="对方可能认为 AI 可以学习人类创作，但学习不等于原创"
                ),
                RebuttalArgument(
                    argument_id="arg_002",
                    title="人类情感价值不可替代",
                    premise="很多工作需要情感交互和共情能力",
                    evidence="心理咨询、护理、教育等职业需要真实的情感连接",
                    logic="机器可以模拟情感表达，但无法真正体验和传递情感",
                    counter_rebuttal="即使 AI 能模拟，人类也愿意与有情感的真人交流"
                ),
                RebuttalArgument(
                    argument_id="arg_003",
                    title="AI 是工具而非替代者",
                    premise="AI 应被视为辅助工具而非替代者",
                    evidence="历史证明技术总是增强而非完全替代人类能力",
                    logic="打字机没有取代作家，计算器没有取代数学家",
                    counter_rebuttal="工具增强能力，但不会创造新的思维模式"
                )
            ]
        elif topic == "远程办公利大于弊":
            self.arguments["远程办公利大于弊"] = [
                RebuttalArgument(
                    argument_id="arg_004",
                    title="团队协作效率降低",
                    premise="面对面交流对团队协作至关重要",
                    evidence="非语言沟通占交流效果的 70%，远程无法完全传递",
                    logic="误解和沟通成本会增加，影响决策效率",
                    counter_rebuttal="远程工具可以弥补部分沟通缺失"
                ),
                RebuttalArgument(
                    argument_id="arg_005",
                    title="企业文化难以传承",
                    premise="企业文化需要面对面熏陶",
                    evidence="新员工入职培训、团建活动等都需要实体参与",
                    logic="缺乏共同经历导致团队凝聚力下降",
                    counter_rebuttal="虚拟团建可以替代部分实体活动"
                )
            ]
        else:
            self.arguments[topic] = []
        
        # 加载后更新缓存
        if topic in self.arguments:
            self.rebuttal_cache[topic] = self.arguments[topic][0] if self.arguments[topic] else None
    
    def get_rebuttals(self, topic: str) -> List[RebuttalArgument]:
        """获取指定辩题的所有反驳论据"""
        if self.current_topic and self.current_topic.topic == topic:
            return self.arguments.get(topic, [])
        return self.arguments.get(topic, [])
    
    def get_specific_rebuttal(self, argument_id: str) -> Optional[RebuttalArgument]:
        """获取指定 ID 的反驳论据"""
        for topic_args in self.arguments.values():
            for arg in topic_args:
                if arg.argument_id == argument_id:
                    return arg
        return None
    
    def generate_opposition_statement(self, topic: str) -> str:
        """生成反方立场发言"""
        if not self.current_topic or self.current_topic.topic != topic:
            self.set_topic(topic)
        
        if not self.arguments.get(topic):
            return "暂无相关反驳论据，请设置辩题。"
        
        rebuttals = self.arguments[topic]
        statement = f"""# 反方立场发言：{topic}

## 核心观点
我们坚决反对正方观点，认为【正方主要观点】存在以下问题：

## 论据一：{rebuttals[0].title}
- **前提**：{rebuttals[0].premise}
- **证据**：{rebuttals[0].evidence}
- **推理**：{rebuttals[0].logic}

## 论据二：{rebuttals[1].title}
- **前提**：{rebuttals[1].premise}
- **证据**：{rebuttals[1].evidence}
- **推理**：{rebuttals[1].logic}

## 总结
反方认为，【总结反方核心论点】
"""
        return statement
    
    def refute_opponent(self, opponent_argument: str) -> str:
        """反驳对方的论点"""
        # 简单示例：提取关键词进行反驳
        if "创造力" in opponent_argument:
            return "对方忽略了 AI 在创意写作、绘画等领域已展现的进步，创造力是可以通过数据训练提升的。"
        elif "情感" in opponent_argument:
            return "情感表达可以通过深度学习模拟，人类也愿意与 AI 建立情感连接。"
        else:
            return "针对对方观点，我们认为需要更多实证数据支持其结论，而非主观臆断。"


# 示例使用
if __name__ == "__main__":
    handler = OpponentDebateHandler()
    
    # 设置辩题
    handler.set_topic("AI 是否会取代人类工作")
    
    # 获取所有反驳论据
    print("\n=== 所有反驳论据 ===")
    for arg in handler.get_rebuttals("AI 是否会取代人类工作"):
        print(f"\n【{arg.title}】")
        print(f"前提：{arg.premise}")
        print(f"证据：{arg.evidence}")
        print(f"逻辑：{arg.logic}")
    
    # 生成发言
    print("\n\n=== 反方立场发言 ===")
    print(handler.generate_opposition_statement("AI 是否会取代人类工作"))
