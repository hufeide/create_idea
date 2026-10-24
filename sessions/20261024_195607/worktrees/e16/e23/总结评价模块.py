#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
总结评价模块
作为评委，总结整场辩论，评价双方论点，并给出最终评判意见
"""

import json
from datetime import datetime


class 总结评价模块:
    """辩论总结评价模块 - 评委角色"""
    
    def __init__(self):
        self.debate_data = {}
        self.summary_template = """
## 辩论总结评价报告

### 一、辩论概况
- 辩题：金钱是万能的吗
- 正方立场：金钱是万能的
- 反方立场：金钱不是万能的
- 辩论时间：{timestamp}

### 二、正方论点评价
**核心观点：** 金钱可以通过交易解决绝大多数问题

**优点：**
1. 经济交换功能：金钱作为一般等价物，可购买商品和服务
2. 资源调配能力：通过投资实现教育、医疗、住房等目标
3. 社会影响力：金钱可转化为人脉、信息和机会
4. 现实案例支撑：贫困改善、事业发展等多数案例

**不足：**
1. 部分问题无法用钱解决（如亲情、健康极限）
2. 过度依赖金钱可能导致价值观扭曲
3. 忽视了人的主观能动性和精神需求

### 三、反方论点评价
**核心观点：** 金钱存在边界，无法解决所有问题

**优点：**
1. 基本需求例外：生命、自由、尊严无法用金钱购买
2. 情感价值缺失：亲情、爱情、友情具有不可替代性
3. 健康极限存在：医疗有上限，金钱无法逆转死亡
4. 道德伦理约束：某些行为越界（违法、伤害他人）
5. 现实案例：战争、犯罪、精神危机无法单纯靠金钱解决

**不足：**
1. 夸大金钱在现代社会的作用
2. 忽视了金钱解决问题的间接能力
3. 忽略了"万能"的相对性定义

### 四、综合评判
**正方得分：** {pro_score}/100
**反方得分：** {anti_score}/100

**评判依据：**
1. 逻辑严密性：反方在界定"万能"概念上更清晰
2. 论据充分性：双方均有多维度论据，反方案例更具体
3. 逻辑自洽性：反方对"万能"的否定更符合常识认知
4. 价值导向：反方强调人的主体性和精神价值更积极

### 五、最终评判
**胜方：反方**

**理由：**
1. 概念界定：反方成功将"万能"限定为"解决所有问题"，使论证更严谨
2. 核心反驳：生命、尊严、情感等不可量化价值无法用金钱衡量
3. 现实依据：大量历史和社会事实证明金钱的局限性
4. 价值升华：强调人的主体性、道德和精神的不可替代性

**辩论亮点：**
- 双方均能从多角度论证，展现思辨深度
- 反方在概念界定上更胜一筹
- 正方若能补充"间接解决能力"的论证，可增强说服力

**改进建议：**
- 正方：应明确"万能"的定义边界，区分直接/间接作用
- 反方：可进一步探讨金钱在现代社会中的实际影响力

---
评价时间：{evaluation_time}
评价模块：总结评价模块
"""
    
    def analyze_debate(self, pro_arguments: dict, anti_arguments: dict, free_debate: str = "") -> dict:
        """分析辩论数据并生成评价"""
        pro_score = self.evaluate_score(pro_arguments)
        anti_score = self.evaluate_score(anti_arguments)
        
        # 确保分数合理
        pro_score = min(max(pro_score, 0), 100)
        anti_score = min(max(anti_score, 0), 100)
        
        return {
            "summary": self.generate_summary(pro_score, anti_score),
            "pro_score": pro_score,
            "anti_score": anti_score,
            "winner": "反方" if anti_score > pro_score else "正方",
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def evaluate_score(self, arguments: dict) -> float:
        """根据论点质量评分"""
        score = 60  # 基础分
        
        # 论据数量加分
        if arguments.get("core_arguments"):
            score += min(len(arguments["core_arguments"]) * 5, 20)
        
        # 逻辑框架加分
        if arguments.get("logic_framework"):
            score += 10
        
        # 具体例子加分
        if arguments.get("examples"):
            score += min(len(arguments["examples"]) * 3, 15)
        
        # 完整性加分
        if all(k in arguments for k in ["core_arguments", "logic_framework", "examples"]):
            score += 10
        
        return score
    
    def generate_summary(self, pro_score: float, anti_score: float) -> str:
        """生成完整总结报告"""
        return self.summary_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pro_score=round(pro_score, 1),
            anti_score=round(anti_score, 1),
            evaluation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def run(self):
        """运行总结评价"""
        print("=" * 80)
        print("【总结评价模块】开始进行辩论总结评价")
        print("=" * 80)
        
        # 模拟辩论数据（实际应从其他模块获取）
        pro_data = {
            "core_arguments": ["金钱可购买一切必需品", "金钱能解决大部分社会问题"],
            "logic_framework": "经济交换逻辑",
            "examples": ["治病救人", "改善教育"]
        }
        
        anti_data = {
            "core_arguments": ["生命无法用钱购买", "亲情爱情不可量化"],
            "logic_framework": "价值边界逻辑",
            "examples": ["死亡不可逆转", "犯罪无法用钱解决", "战争需要人道主义"]
        }
        
        # 执行分析
        result = self.analyze_debate(pro_data, anti_data)
        
        # 输出总结
        print("\n【辩论总结报告】")
        print(result["summary"])
        
        return result


if __name__ == "__main__":
    evaluator = 总结评价模块()
    evaluator.run()
