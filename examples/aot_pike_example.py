#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AoT-PIKE示例脚本
此脚本展示了如何使用AoT-PIKE进行问题回答
"""

import os
import sys
import json
import argparse
from pathlib import Path

from pikerag.utils.config_loader import load_workflow
from pikerag.workflows.common import BaseQaData


class SimpleQaData(BaseQaData):
    """简单的问答数据类"""

    def __init__(self, questions, answers=None):
        """初始化问答数据

        Args:
            questions: 问题列表
            answers: 答案列表，可选
        """
        self.questions = questions
        self.answers = answers or [None] * len(questions)

    def get_question(self, idx):
        """获取指定索引的问题"""
        return self.questions[idx]

    def get_answer(self, idx):
        """获取指定索引的答案"""
        return self.answers[idx]

    def get_size(self):
        """获取问题数量"""
        return len(self.questions)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AoT-PIKE示例脚本")
    parser.add_argument("--config", type=str, default="examples/hotpotqa/configs/aot_pike.yml",
                        help="配置文件路径")
    parser.add_argument("--question", type=str, default="谁是《哈利·波特与魔法石》的作者，她出生于哪一年？",
                        help="要回答的问题")
    parser.add_argument("--output", type=str, default="aot_pike_result.json",
                        help="输出结果文件路径")
    args = parser.parse_args()

    # 加载配置文件
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载工作流
    workflow = load_workflow(config_path)

    # 创建问答数据
    qa_data = SimpleQaData([args.question])

    # 回答问题
    print(f"问题: {args.question}")
    print("正在使用AoT-PIKE回答问题...")
    result = workflow.answer(qa_data, 0)

    # 输出结果
    print(f"\n回答: {result.get('answer', '未找到答案')}")
    
    # 保存详细结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {args.output}")

    # 输出推理过程摘要
    if "reasoning_process" in result:
        process = result["reasoning_process"]
        print("\n推理过程摘要:")
        print(f"原始问题: {process.get('original_question', '')}")
        
        for i, iteration in enumerate(process.get("iterations", [])):
            print(f"\n迭代 {i+1}:")
            print(f"当前状态: {iteration.get('current_state', '')}")
            
            if "is_answerable" in iteration and iteration["is_answerable"]:
                print(f"状态可直接回答: {iteration.get('reason', '')}")
                continue
                
            if "dag_decompose_failed" in iteration:
                print("DAG分解失败，尝试直接回答")
                continue
                
            if "dag" in iteration and "nodes" in iteration["dag"]:
                nodes = iteration["dag"]["nodes"]
                print(f"分解为 {len(nodes)} 个节点:")
                for node in nodes:
                    print(f"  - {node.get('id', '')}: {node.get('question', '')}")
            
            if "node_results" in iteration:
                results = iteration["node_results"]
                print(f"解决了 {len(results)} 个节点:")
                for node_id, result in results.items():
                    print(f"  - {node_id}: {result.get('answer', '')}")
            
            if "new_state" in iteration:
                print(f"新状态: {iteration.get('new_state', '')}")


if __name__ == "__main__":
    main() 