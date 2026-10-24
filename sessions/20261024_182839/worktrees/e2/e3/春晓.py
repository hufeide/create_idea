#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《春晓》 - 唐代·孟浩然
这是一首著名的五言绝句，描绘了春日清晨的景色和诗人的感受。
"""


class ChunXiao:
    """春晓诗模块"""
    
    def __init__(self):
        self.author = "孟浩然"
        self.dynasty = "唐代"
        self.title = "春晓"
        
    def get_poem(self):
        """返回《春晓》的完整诗句"""
        return [
            "春眠不觉晓，",
            "处处闻啼鸟。",
            "夜来风雨声，",
            "花落知多少。"
        ]
    
    def display(self):
        """展示诗歌的完整格式"""
        print("=" * 40)
        print(f"{'《' + self.title + '》':^40}")
        print(f"{'作者：' + self.author + '（' + self.dynasty + '）':^40}")
        print("=" * 40)
        
        for i, line in enumerate(self.get_poem(), 1):
            print(f"{i}. {line}")
        
        print("=" * 40)
        print("\n【注释】")
        print("春眠：春天的睡眠")
        print("不觉晓：不知不觉天已破晓")
        print("处处：到处")
        print("啼鸟：鸣叫的鸟儿")
        print("夜来：昨夜")
        print("风雨声：刮风下雨的声音")
        print("花落：花凋落")
        print("=" * 40)


def main():
    """主函数"""
    poem = ChunXiao()
    poem.display()
    
    # 返回诗歌内容供程序使用
    return poem.get_poem()


if __name__ == "__main__":
    main()
