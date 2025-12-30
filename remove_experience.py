#!/usr/bin/env python3
"""
脚本功能：删除jsonl数据集中system prompt中的"RELEVANT EXPERIENCE FROM SIMILAR TASKS"部分
"""

import json
import argparse
import os
import re
from pathlib import Path


def remove_experience_section(content: str) -> str:
    """
    从system prompt中删除"RELEVANT EXPERIENCE FROM SIMILAR TASKS"部分
    
    Args:
        content: system prompt的原始内容
        
    Returns:
        删除经验部分后的内容
    """
    # 定义要删除的部分的开始标记
    start_marker = "==================================================\nRELEVANT EXPERIENCE FROM SIMILAR TASKS\n=================================================="
    
    # 定义下一个部分的开始标记（经验部分结束的地方）
    end_marker = "==================================================\nOUTPUT FORMAT\n=================================================="
    
    # 查找开始标记的位置
    start_idx = content.find(start_marker)
    
    if start_idx == -1:
        # 如果没有找到开始标记，说明这个样本没有经验部分，直接返回原内容
        return content
    
    # 查找结束标记的位置（在开始标记之后）
    end_idx = content.find(end_marker, start_idx)
    
    if end_idx == -1:
        # 如果没有找到结束标记，可能需要用其他方式处理
        # 尝试查找下一个"=================================================="
        pattern = r"==================================================\n[A-Z][A-Z\s]+\n=================================================="
        matches = list(re.finditer(pattern, content[start_idx + len(start_marker):]))
        if matches:
            # 找到下一个分隔符部分
            end_idx = start_idx + len(start_marker) + matches[0].start()
        else:
            # 如果还是找不到，就删除从开始标记到文件末尾的所有内容
            # 但这种情况应该很少见，我们保留原内容并打印警告
            print(f"警告：无法找到经验部分的结束位置，保留原内容")
            return content
    
    # 删除经验部分（包括开始标记和结束标记之间的所有内容，但不包括结束标记本身）
    # 我们需要保留开始标记之前的内容和结束标记之后的内容
    before = content[:start_idx]
    after = content[end_idx:]
    
    # 移除开始标记前的多余换行（如果有的话）
    before = before.rstrip()
    if before and not before.endswith('\n'):
        before += '\n'
    
    # 确保after部分以换行开始（如果end_marker后面没有换行）
    if after and not after.startswith('\n'):
        after = '\n' + after
    
    return before + after


def process_jsonl(input_path: str, output_path: str):
    """
    处理jsonl文件，删除system prompt中的经验部分
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    processed_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_count += 1
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 查找system消息并处理
                if 'messages' in data:
                    for message in data['messages']:
                        if message.get('role') == 'system' and 'content' in message:
                            original_content = message['content']
                            new_content = remove_experience_section(original_content)
                            
                            if original_content != new_content:
                                processed_count += 1
                            
                            message['content'] = new_content
                
                # 写入处理后的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"警告：第 {total_count} 行JSON解析失败: {e}")
                continue
    
    print(f"处理完成！")
    print(f"总行数: {total_count}")
    print(f"已删除经验部分的行数: {processed_count}")
    print(f"输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='删除jsonl数据集中system prompt中的"RELEVANT EXPERIENCE FROM SIMILAR TASKS"部分'
    )
    parser.add_argument(
        'input_path',
        nargs='?',
        default='train_data/alfworld_valid_train_matts_top2_qwen3-8b.jsonl',
        help='输入jsonl文件路径（默认: train_data/alfworld_valid_train_matts_top2_qwen3-8b.jsonl）'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"错误：输入文件不存在: {input_path}")
        return
    
    # 生成输出文件路径（在原文件名前加"noexp"）
    output_path = input_path.parent / f"noexp_{input_path.name}"
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"开始处理...")
    
    process_jsonl(str(input_path), str(output_path))


if __name__ == '__main__':
    main()

