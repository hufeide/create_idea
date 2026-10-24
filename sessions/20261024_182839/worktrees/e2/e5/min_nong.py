#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《悯农》- 唐代·李绅

这是一首著名的五言绝句，描写农民劳作的艰辛，表达对劳动人民的同情。
"""

def display_poem():
    """显示《悯农》全诗"""
    poem = [
        "春种一粒粟，",
        "秋收万颗子。",
        "四海无闲田，",
        "农夫犹饿死。"
    ]
    
    print("=" * 40)
    print("《悯农》")
    print("—— 唐代·李绅")
    print("=" * 40)
    
    for line in poem:
        print(line)
    
    print("=" * 40)
    print("\n【诗意】")
    print("春天播种下一粒种子，")
    print("到了秋天就能收获万颗粮食。")
    print("天下没有荒废的田地，")
    print("但辛勤劳作的农民仍然会饿死。")
    print("\n【主题】")
    print("这首诗通过对比丰收的景象与农民的悲惨命运，")
    print("深刻揭露了封建社会的残酷现实，表达了对劳动人民的深切同情。")
    
if __name__ == "__main__":
    display_poem()
