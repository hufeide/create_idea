#!/usr/bin/env python3
"""
《春晓》与《悯农》诗歌对比分析
比较两首诗在农田场景的适用性
"""

class ChunXiao:
    """春晓 - 孟浩然"""
    title = "春晓"
    author = "孟浩然"
    poem = """春眠不觉晓，
处处闻啼鸟。
夜来风雨声，
花落知多少。"""
    
    content = {
        "主题": "春日清晨的宁静与美好",
        "情感": "惜春之情，对自然变化的敏感",
        "意象": ["春眠", "啼鸟", "风雨", "落花"],
        "适用场景": "卧室、庭院、书房",
        "情感基调": "轻柔、闲适、略带惆怅"
    }

class MinNong:
    """悯农 - 李绅"""
    title = "悯农"
    author = "李绅"
    poem = """春种一粒粟，
秋收万颗子。
四海无闲田，
农夫犹饿死。"""
    
    content = {
        "主题": "农民辛勤劳作却仍受剥削的悲愤",
        "情感": "同情农民疾苦，揭露社会不公",
        "意象": ["春种", "秋收", "闲田", "饿死"],
        "适用场景": "农田、乡村、劳动场所",
        "情感基调": "沉重、悲壮、充满社会批判"
    }

def compare_for_farmland():
    """比较两首诗在农田场景的适用性"""
    
    print("=" * 70)
    print("《春晓》与《悯农》 - 农田场景适用性对比分析")
    print("=" * 70)
    print()
    
    print("【诗歌内容对比】")
    print("-" * 70)
    print(f"《春晓》- {ChunXiao.author}")
    print(ChunXiao.poem)
    print()
    print(f"《悯农》- {MinNong.author}")
    print(MinNong.poem)
    print()
    
    print("【主题与情感对比】")
    print("-" * 70)
    print(f"《春晓》主题：{ChunXiao.content['主题']}")
    print(f"《春农》主题：{MinNong.content['主题']}")
    print()
    
    print("【意象对比】")
    print("-" * 70)
    print(f"《春晓》意象：{', '.join(ChunXiao.content['意象'])}")
    print(f"《悯农》意象：{', '.join(MinNong.content['意象'])}")
    print()
    
    print("【农田场景分析】")
    print("-" * 70)
    print("《春晓》:")
    print("  - 意象：春眠、啼鸟、风雨、落花")
    print("  - 氛围：轻柔、闲适、略带惆怅")
    print("  - 与农田的关联度：低")
    print("  - 适合场景：室内、庭院，不适合农田劳作时念诵")
    print()
    
    print("《悯农》:")
    print("  - 意象：春种、秋收、闲田、饿死")
    print("  - 氛围：沉重、悲壮、充满社会批判")
    print("  - 与农田的关联度：极高")
    print("  - 内容直接描述农业生产过程（春种、秋收）")
    print("  - 适合场景：农田、劳动场所、乡村")
    print()
    
    print("=" * 70)
    print("【结论】")
    print("=" * 70)
    print()
    print("在农田里念诵，**《悯农》更适合**。")
    print()
    print("理由：")
    print("1. 主题契合：《悯农》直接描写农业生产（春种、秋收、四海无闲田）")
    print("2. 场景关联：诗中的意象与农田劳作高度相关")
    print("3. 情感共鸣：在劳动场所念诵更能引起情感共鸣")
    print("4. 文化传统：《悯农》本身就是为关注农民疾苦而作，常用于农业相关场景")
    print()
    print("《春晓》虽然意境优美，但更适合在室内、庭院等休闲场景欣赏，")
    print("与农田劳作的实际场景关联度较低。")
    print("=" * 70)

if __name__ == "__main__":
    compare_for_farmland()
