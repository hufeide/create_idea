"""
自由辩论主持模块
职责：设计自由辩论环节的流程规则，并主持 3 分钟自由辩论
"""

import time
from datetime import datetime

class FreedomDebateHost:
    """自由辩论主持人"""
    
    def __init__(self):
        self.debate_topic = "金钱是万能的吗"
        self.side_a = "正方"  # 金钱是万能的
        self.side_b = "反方"  # 金钱不是万能的
        self.timer = 180  # 总时长：3 分钟 = 180 秒
        self.current_round = 1
        self.time_left = 180
        self.speaker_order = ["正方", "反方", "正方", "反方"]
        self.speech_times = {self.side_a: 15, self.side_b: 15}  # 每轮每方 15 秒
        
    def design_rules(self):
        """设计自由辩论流程规则"""
        rules = """# 自由辩论环节规则

## 一、基本规则
1. 每轮由正方和反方各发言 15 秒
2. 每轮总时长 30 秒，共 6 轮，总计 3 分钟
3. 发言必须紧扣辩题"金钱是否是万能的"
4. 发言内容需简洁有力，避免重复对方已说完的观点
5. 超时将被打断

## 二、发言流程
- 第 1 轮：正方率先发言（15 秒）→ 反方回应（15 秒）
- 第 2-6 轮：交替进行，反方先发言
- 每轮结束后主持人计时并提示

## 三、注意事项
- 发言时间由主持人严格控制
- 鼓励针对性反驳和精彩举例
- 保持辩论礼仪，尊重对方观点

## 四、计时规则
- 使用倒计时提示
- 最后 5 秒特别提示
- 超时立即打断"""
        return rules
    
    def start_debate(self):
        """主持 3 分钟自由辩论"""
        print("=" * 70)
        print(" 🎤 自由辩论环节开始！ 🎤")
        print(f"辩题：{self.debate_topic}")
        print("=" * 70)
        print("\n📋 规则已公布，请双方辩手做好准备！")
        print("\n" + "-" * 70 + "\n")
        
        round_num = 1
        max_rounds = 6  # 6 轮 × 2 方 × 15 秒 = 180 秒
        
        while round_num <= max_rounds:
            # 确定当前发言方：奇数轮正方先，偶数轮反方先
            if round_num % 2 == 1:  # 奇数轮：正方先
                current_speaker = self.side_a
            else:  # 偶数轮：反方先
                current_speaker = self.side_b
            
            # 计算剩余时间
            remaining_time = 15.0
            
            print(f"\n⏱️ 第{round_num}轮 - {current_speaker}发言（剩余{remaining_time:.1f}秒）")
            print("-" * 70)
            
            # 模拟发言内容
            if current_speaker == self.side_a:
                print("🔵 正方发言：")
                print("  - 金钱可以购买资源、人脉、信息，实现大部分目标")
                print("  - 投资回报、创业成功、改变命运都需要资金支持")
                print("  - 金钱能换取时间、健康、情感，体现其万能性")
                print("  - 例子：治病救人靠医药费，求学深造靠学费")
            else:
                print("🔴 反方发言：")
                print("  - 金钱买不到亲情、爱情和真挚友谊")
                print("  - 生命、尊严、自由等价值无法用金钱衡量")
                print("  - 有钱也可能买到假货、坏名声，甚至犯罪工具")
                print("  - 例子：亲情无法用钱购买，真爱超越物质")
            
            # 模拟发言时长（15 秒）
            time.sleep(1.5)
            
            # 更新状态
            self.time_left -= 15
            self.current_round += 1
            
            # 检查是否结束
            if self.time_left <= 0:
                break
            
            # 提示轮次切换
            next_speaker = "反方" if round_num % 2 == 1 else "正方"
            print(f"\n⏰ 轮次切换！下一轮由{next_speaker}发言")
            print("-" * 70)
            
            round_num += 1
        
        # 结束总结
        self.end_debate()
    
    def end_debate(self):
        """结束自由辩论"""
        print("\n" + "=" * 70)
        print(" ⏰ 自由辩论环节结束！ ⏰")
        print("=" * 70)
        print(f"\n✅ 总用时：{self.timer}秒（3 分钟）")
        print(f"✅ 完成轮次：{self.current_round - 1}轮")
        print("\n🎯 请进入总结评价环节，评委将评判双方表现\n")
        print("=" * 70)


def main():
    """主函数"""
    host = FreedomDebateHost()
    
    # 显示规则
    print("\n" + "=" * 70)
    print(" 📜 自由辩论流程规则 📜")
    print("=" * 70)
    print(host.design_rules())
    print("=" * 70)
    
    # 主持辩论
    print("\n\n🚀 开始主持自由辩论...")
    time.sleep(1)
    host.start_debate()


if __name__ == "__main__":
    main()
