#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反方立论模块 - 针对'金钱是万能的'的反驳观点
包含：核心论据、逻辑框架、具体例子
"""

class AntagonistPosition:
    """反方立论核心模块"""
    
    def __init__(self):
        self.core_thesis = "金钱不是万能的"
        self.arguments = []
        self.framework = {}
        self.examples = []
    
    def build_core_arguments(self):
        """构建核心论据体系"""
        self.arguments = [
            {
                "title": "论据一：生存基本权的不可替代性",
                "description": "金钱无法购买生命权和基本生存尊严，这是人类存在的底线",
                "strength": "高",
                "detail": "即使拥有巨额财富，也无法购买延寿能力、复活能力或避免死亡。饥饿时食物可买，但生命的终结无法用金钱逆转。"
            },
            {
                "title": "论据二：情感与精神价值的不可交易性",
                "description": "真爱、亲情、友情等情感价值无法用金钱衡量或购买",
                "strength": "高",
                "detail": "金钱可以营造浪漫氛围，但无法购买真实的爱情；可以支付保姆照顾老人，但无法替代亲人的陪伴和情感连接。"
            },
            {
                "title": "论据三：道德与正义的独立性",
                "description": "金钱不能购买道德良知和社会正义",
                "strength": "中",
                "detail": "历史上无数贪官污吏用金钱行贿，但无法改变内心的贪婪；受害者即使拥有财富，也无法改变被压迫的命运。"
            },
            {
                "title": "论据四：自由意志的局限性",
                "description": "极端的财富反而可能限制人的自由选择和人生可能性",
                "strength": "中",
                "detail": "富豪可能被绑架勒索，被绑架者无法拒绝；富豪可能因巨额债务而失去自由；财富带来的压力、责任和孤独感反而限制了精神自由。"
            },
            {
                "title": "论据五：自然环境的不可完全占有",
                "description": "金钱无法购买清新的空气、纯净的水和健康的生态环境",
                "strength": "高",
                "detail": "污染地区的富豪同样呼吸着有毒空气；即使住在豪宅，也无法完全隔绝自然灾害和环境污染的危害。"
            }
        ]
    
    def build_logical_framework(self):
        """构建逻辑论证框架"""
        self.framework = {
            "framework_name": "五维反证法",
            "structure": [
                {
                    "level": "第一层：生存维度",
                    "focus": "生命权与健康权",
                    "argument": "金钱无法购买生命本身和终极健康"
                },
                {
                    "level": "第二层：情感维度",
                    "focus": "人际关系与精神世界",
                    "argument": "情感连接无法商品化"
                },
                {
                    "level": "第三层：道德维度",
                    "focus": "良知与正义",
                    "argument": "道德自律超越物质诱惑"
                },
                {
                    "level": "第四层：自由维度",
                    "focus": "选择权与自主性",
                    "argument": "财富可能成为枷锁而非解放"
                },
                {
                    "level": "第五层：环境维度",
                    "focus": "自然与生态",
                    "argument": "个人无法独占自然资源"
                }
            ],
            "conclusion": "金钱只能作为工具而非终极价值，它无法替代人类存在的根本需求"
        }
    
    def build_examples(self):
        """构建具体案例库"""
        self.examples = [
            {
                "category": "生存危机",
                "example": "2020 年武汉疫情期间，富豪同样面临感染风险，金钱无法阻止病毒传播",
                "relevance": "证明金钱在公共卫生危机面前的局限性"
            },
            {
                "category": "情感关系",
                "example": "《泰坦尼克号》中，杰克拒绝西尔维娅的巨额支票，因为真爱无法用金钱衡量",
                "relevance": "证明情感价值的不可交易性"
            },
            {
                "category": "道德困境",
                "example": "贪官受贿后仍会被审判，金钱无法购买法律豁免权",
                "relevance": "证明道德约束具有独立性"
            },
            {
                "category": "自由限制",
                "example": "2018 年某富豪被绑架勒索 5000 万元，显示富人也无法完全控制自身安全",
                "relevance": "证明财富可能带来新的束缚"
            },
            {
                "category": "自然环境",
                "example": "雾霾城市中的富豪同样呼吸着污染空气，无法购买健康的生态环境",
                "relevance": "证明自然资源的公共属性"
            }
        ]
    
    def generate_complete_argumentation(self):
        """生成完整的反方立论陈述"""
        return f"""
【反方立论】金钱不是万能的

【核心命题】
我方坚持认为：金钱不是万能的。金钱作为一种交换媒介和价值尺度，只能满足人类的部分物质需求，但在生命、情感、道德、自由和自然等根本价值面前，它无能为力。

【核心论据体系】

第一，生存基本权不可替代。
生命权和健康权是人的底线，金钱无法购买死亡后的复活，无法逆转疾病的终极阶段。疫情期间，无论贫富，病毒面前人人平等。

第二，情感与精神价值不可交易。
真爱、亲情、友情无法用金钱衡量。金钱可以营造浪漫，但无法创造真实的情感连接。

第三，道德与正义具有独立性。
良知不能购买，正义不能贿赂。贪官最终难逃法网，证明道德约束超越物质力量。

第四，财富可能限制自由。
巨额债务可致人破产，绑架勒索针对富豪，财富带来的责任和孤独反而限制精神自由。

第五，自然环境不可完全占有。
清新的空气、纯净的水源是公共资源，个人无法通过金钱独占。

【逻辑框架】
采用五维反证法：从生存、情感、道德、自由、环境五个维度系统反驳"金钱万能论"。

【结论】
金钱是重要的工具，但不是万能的。人类的存在价值超越物质财富，这是不可剥夺的根本。

【谢谢聆听】
"""


def main():
    """主函数：演示反方立论生成"""
    print("=" * 80)
    print("反方立论模块启动")
    print("=" * 80)
    
    antagonist = AntagonistPosition()
    antagonist.build_core_arguments()
    antagonist.build_logical_framework()
    antagonist.build_examples()
    
    print("\n【核心论据】")
    print("-" * 80)
    for i, arg in enumerate(antagonist.arguments, 1):
        print(f"\n{i}. {arg['title']}")
        print(f"   强度：{arg['strength']}")
        print(f"   {arg['description']}")
        print(f"   详情：{arg['detail']}")
    
    print("\n\n【逻辑框架】")
    print("-" * 80)
    print(f"框架名称：{antagonist.framework.get('framework_name', 'N/A')}")
    for level in antagonist.framework['structure']:
        print(f"\n{level['level']}")
        print(f"  焦点：{level['focus']}")
        print(f"  论点：{level['argument']}")
    
    print("\n\n【具体案例】")
    print("-" * 80)
    for i, ex in enumerate(antagonist.examples, 1):
        print(f"\n{i}. {ex['category']}: {ex['example']}")
        print(f"   相关性：{ex['relevance']}")
    
    print("\n\n【完整立论陈述】")
    print("-" * 80)
    print(antagonist.generate_complete_argumentation())
    
    print("=" * 80)
    print("反方立论生成完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
