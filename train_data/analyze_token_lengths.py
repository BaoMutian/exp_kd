#!/usr/bin/env python3
"""
统计train_data目录下所有jsonl文件中system prompt、user prompt和assistant response的token长度分布
使用qwen3-8b的词表进行tokenization，并可视化显示结果
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_messages(data: List[Dict]) -> Dict[str, List[str]]:
    """
    从数据中提取system、user和assistant的content
    
    Returns:
        Dict with keys: 'system', 'user', 'assistant'
    """
    result = {
        'system': [],
        'user': [],
        'assistant': []
    }
    
    for item in data:
        if 'messages' in item:
            for msg in item['messages']:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role in result and content:
                    result[role].append(content)
    
    return result


def count_tokens(text: str, tokenizer) -> int:
    """统计文本的token数量"""
    # 使用tokenizer编码文本，不添加特殊token
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def analyze_file(file_path: Path, tokenizer) -> Dict[str, List[int]]:
    """分析单个文件的token长度"""
    print(f"正在处理: {file_path.name}")
    data = load_jsonl(file_path)
    
    token_lengths = {
        'system': [],
        'user': [],
        'assistant': [],
        'total': []  # 总长度
    }
    
    # 按样本处理，以便计算总长度
    for item in data:
        if 'messages' in item:
            sample_lengths = {'system': 0, 'user': 0, 'assistant': 0}
            
            for msg in item['messages']:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role in ['system', 'user', 'assistant'] and content:
                    length = count_tokens(content, tokenizer)
                    sample_lengths[role] = length
                    token_lengths[role].append(length)
            
            # 计算总长度（system + user + assistant）
            total_length = sample_lengths['system'] + sample_lengths['user'] + sample_lengths['assistant']
            token_lengths['total'].append(total_length)
    
    return token_lengths


def aggregate_statistics(all_lengths: Dict[str, List[int]]) -> Dict[str, Dict]:
    """聚合统计信息"""
    stats = {}
    for role, lengths in all_lengths.items():
        if lengths:
            stats[role] = {
                'count': len(lengths),
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'p25': np.percentile(lengths, 25),
                'p75': np.percentile(lengths, 75),
                'p95': np.percentile(lengths, 95),
                'p99': np.percentile(lengths, 99),
            }
        else:
            stats[role] = {
                'count': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'p25': 0,
                'p75': 0,
                'p95': 0,
                'p99': 0,
            }
    return stats


def print_statistics(stats: Dict[str, Dict]):
    """打印统计信息"""
    print("\n" + "="*80)
    print("Token长度统计信息")
    print("="*80)
    
    for role in ['system', 'user', 'assistant', 'total']:
        s = stats[role]
        role_name = 'TOTAL (System+User+Assistant)' if role == 'total' else role.upper()
        print(f"\n{role_name}:")
        print(f"  样本数量: {s['count']}")
        if s['count'] > 0:
            print(f"  平均值:   {s['mean']:.2f}")
            print(f"  中位数:   {s['median']:.2f}")
            print(f"  标准差:   {s['std']:.2f}")
            print(f"  最小值:   {s['min']}")
            print(f"  最大值:   {s['max']}")
            print(f"  P25:      {s['p25']:.2f}")
            print(f"  P75:      {s['p75']:.2f}")
            print(f"  P95:      {s['p95']:.2f}")
            print(f"  P99:      {s['p99']:.2f}")


def plot_distributions(all_lengths: Dict[str, List[int]], output_dir: Path):
    """绘制token长度分布图"""
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 绘制四个角色的直方图（分开显示，包括总长度）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    roles = ['system', 'user', 'assistant', 'total']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    role_labels = ['SYSTEM', 'USER', 'ASSISTANT', 'TOTAL (System+User+Assistant)']
    
    for idx, (role, color, label) in enumerate(zip(roles, colors, role_labels)):
        ax = axes[idx // 2, idx % 2]
        lengths = all_lengths[role]
        
        if lengths:
            ax.hist(lengths, bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Token Length', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{label} Token Length Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean = np.mean(lengths)
            median = np.median(lengths)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
            ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No {role} data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{label} Token Length Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_length_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n已保存分布图: {output_dir / 'token_length_distributions.png'}")
    
    # 2. 绘制四个角色的箱线图（对比，包括总长度）
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    data_to_plot = [all_lengths[role] for role in roles if all_lengths[role]]
    labels = [role_labels[roles.index(role)] for role in roles if all_lengths[role]]
    plot_colors = [colors[roles.index(role)] for role in roles if all_lengths[role]]
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Token Length', fontsize=12)
        ax.set_title('Token Length Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'token_length_boxplot.png', dpi=300, bbox_inches='tight')
        print(f"已保存箱线图: {output_dir / 'token_length_boxplot.png'}")
    
    # 3. 绘制累积分布函数（CDF，包括总长度）
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for role, color, label in zip(roles, colors, role_labels):
        lengths = all_lengths[role]
        if lengths:
            sorted_lengths = np.sort(lengths)
            cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
            ax.plot(sorted_lengths, cumulative, label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Token Length', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'token_length_cdf.png', dpi=300, bbox_inches='tight')
    print(f"已保存CDF图: {output_dir / 'token_length_cdf.png'}")
    
    # 4. 绘制重叠的密度图（KDE，包括总长度）
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for role, color, label in zip(roles, colors, role_labels):
        lengths = all_lengths[role]
        if lengths:
            ax.hist(lengths, bins=50, density=True, alpha=0.5, 
                   label=label, color=color, edgecolor='black')
    
    ax.set_xlabel('Token Length', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Token Length Distribution Comparison (Overlapped)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'token_length_overlapped.png', dpi=300, bbox_inches='tight')
    print(f"已保存重叠分布图: {output_dir / 'token_length_overlapped.png'}")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='统计jsonl文件中system、user和assistant的token长度分布'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='train_data',
        help='数据目录路径（默认: train_data）'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen3-8B',
        help='模型名称（默认: Qwen/Qwen3-8B）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='train_data/token_analysis',
        help='输出目录（默认: train_data/token_analysis）'
    )
    
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"错误：数据目录不存在: {data_dir}")
        return
    
    # 加载tokenizer
    print(f"正在加载tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"错误：无法加载tokenizer: {e}")
        return
    
    # 查找所有jsonl文件
    jsonl_files = list(data_dir.glob('*.jsonl'))
    if not jsonl_files:
        print(f"警告：在 {data_dir} 中未找到jsonl文件")
        return
    
    print(f"\n找到 {len(jsonl_files)} 个jsonl文件:")
    for f in jsonl_files:
        print(f"  - {f.name}")
    
    # 聚合所有文件的token长度
    all_lengths = {
        'system': [],
        'user': [],
        'assistant': [],
        'total': []
    }
    
    for jsonl_file in jsonl_files:
        token_lengths = analyze_file(jsonl_file, tokenizer)
        for role in ['system', 'user', 'assistant', 'total']:
            all_lengths[role].extend(token_lengths[role])
    
    # 计算统计信息
    stats = aggregate_statistics(all_lengths)
    print_statistics(stats)
    
    # 绘制可视化图表
    print("\n正在生成可视化图表...")
    plot_distributions(all_lengths, output_dir)
    
    # 保存统计信息到文件
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型以便JSON序列化
        stats_json = {}
        for role, s in stats.items():
            stats_json[role] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) 
                               for k, v in s.items()}
        json.dump(stats_json, f, indent=2, ensure_ascii=False)
    print(f"已保存统计信息: {stats_file}")
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()

