#!/usr/bin/env python3
"""整理三个方法的实验结果并生成 Markdown 文档。"""

import json
import os
from pathlib import Path


def load_json_file(filepath):
    """加载 JSON 文件。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_accuracy(value):
    """格式化准确率为百分比。"""
    return f"{value * 100:.2f}%"


def generate_markdown():
    """生成 Markdown 文档。"""
    base_dir = Path(__file__).parent
    
    # 文件路径（优先查找 results_celltype 目录）
    vime_file = base_dir / "VIME" / "results_celltype" / "summary_celltype_labels1000_iters50.json"
    
    # VAT 和 VAT_VIME_Feature 可能的结果文件路径
    vat_file = base_dir / "VAT" / "results_celltype" / "summary_mnist_noise0_labels1000_iters50.json"
    if not vat_file.exists():
        vat_file = base_dir / "VAT" / "results" / "summary_mnist_noise0_labels1000_iters50.json"
    
    vat_vime_file = base_dir / "VAT_VIME_Feature" / "results_celltype" / "summary_mnist_noise0_labels1000_VAT_iters50.json"
    if not vat_vime_file.exists():
        vat_vime_file = base_dir / "VAT_VIME_Feature" / "results" / "summary_mnist_noise0_labels1000_VAT_iters50.json"
    
    # 加载数据
    vime_data = load_json_file(vime_file) if vime_file.exists() else None
    vat_data = load_json_file(vat_file) if vat_file.exists() else None
    vat_vime_data = load_json_file(vat_vime_file) if vat_vime_file.exists() else None
    
    # 生成 Markdown
    md_lines = []
    md_lines.append("# PBMC 5k 细胞类型分类实验结果汇总")
    md_lines.append("")
    md_lines.append("## 实验配置")
    md_lines.append("")
    
    # 从 VIME 数据中提取元数据（因为它有完整的 cell type 信息）
    if vime_data and 'metadata' in vime_data:
        meta = vime_data['metadata']
        md_lines.append(f"- **数据集**: {meta.get('dataset', 'N/A')}")
        md_lines.append(f"- **标签类型**: {meta.get('label_type', 'N/A')}")
        md_lines.append(f"- **细胞类型**: {', '.join(meta.get('cell_types', []))}")
        md_lines.append(f"- **迭代次数**: {meta.get('iterations', 'N/A')}")
        md_lines.append(f"- **标签数据比例**: {meta.get('label_data_rate', 'N/A')}")
        md_lines.append(f"- **模型**: {meta.get('model_name', 'N/A')}")
        md_lines.append(f"- **隐藏层维度**: {meta.get('hidden_dim', 'N/A')}")
        md_lines.append("")
    
    md_lines.append("## 方法对比")
    md_lines.append("")
    
    # VIME 结果
    if vime_data and 'methods' in vime_data and 'VIME' in vime_data['methods']:
        md_lines.append("### VIME")
        md_lines.append("")
        md_lines.append("| 方法 | 均值 | 标准差 | 最小值 | 最大值 |")
        md_lines.append("|------|------|--------|--------|--------|")
        
        summary = vime_data['methods']['VIME'].get('summary', {})
        for method_name in ['Supervised', 'Autoencoder', 'VIME-Self', 'VIME']:
            if method_name in summary:
                stats = summary[method_name]
                md_lines.append(
                    f"| {method_name} | {format_accuracy(stats['mean'])} | "
                    f"{format_accuracy(stats['std'])} | {format_accuracy(stats['min'])} | "
                    f"{format_accuracy(stats['max'])} |"
                )
        md_lines.append("")
    
    # VAT 结果
    if vat_data and 'methods' in vat_data and 'VAT' in vat_data['methods']:
        md_lines.append("### VAT")
        md_lines.append("")
        md_lines.append("| 方法 | 均值 | 标准差 | 最小值 | 最大值 |")
        md_lines.append("|------|------|--------|--------|--------|")
        
        summary = vat_data['methods']['VAT'].get('summary', {})
        for method_name in ['Supervised', 'Autoencoder', 'VIME-Self', 'VAT']:
            if method_name in summary:
                stats = summary[method_name]
                md_lines.append(
                    f"| {method_name} | {format_accuracy(stats['mean'])} | "
                    f"{format_accuracy(stats['std'])} | {format_accuracy(stats['min'])} | "
                    f"{format_accuracy(stats['max'])} |"
                )
        md_lines.append("")
    
    # VAT_VIME_Feature 结果
    if vat_vime_data and 'methods' in vat_vime_data and 'VAT' in vat_vime_data['methods']:
        md_lines.append("### VAT_VIME_Feature")
        md_lines.append("")
        md_lines.append("| 方法 | 均值 | 标准差 | 最小值 | 最大值 |")
        md_lines.append("|------|------|--------|--------|--------|")
        
        summary = vat_vime_data['methods']['VAT'].get('summary', {})
        # VAT_VIME_Feature 的方法名不同
        method_names = ['Supervised', 'Autoencoder', 'VIME-Layer3', 'VAT-on-X', 
                       'RandomNoise-on-X', 'VAT-on-Layer1', 'VAT-on-Layer2', 
                       'VAT-on-Layer3', 'VAT-on-Concat']
        for method_name in method_names:
            if method_name in summary:
                stats = summary[method_name]
                md_lines.append(
                    f"| {method_name} | {format_accuracy(stats['mean'])} | "
                    f"{format_accuracy(stats['std'])} | {format_accuracy(stats['min'])} | "
                    f"{format_accuracy(stats['max'])} |"
                )
        md_lines.append("")
    
    # 综合对比表
    md_lines.append("## 综合对比（主要方法）")
    md_lines.append("")
    md_lines.append("| 方法 | VIME | VAT | VAT_VIME_Feature |")
    md_lines.append("|------|------|-----|------------------|")
    
    # 提取主要方法的均值
    methods_to_compare = ['Supervised', 'Autoencoder']
    
    for method in methods_to_compare:
        vime_val = ""
        vat_val = ""
        vat_vime_val = ""
        
        if vime_data and 'methods' in vime_data and 'VIME' in vime_data['methods']:
            summary = vime_data['methods']['VIME'].get('summary', {})
            if method in summary:
                vime_val = format_accuracy(summary[method]['mean'])
        
        if vat_data and 'methods' in vat_data and 'VAT' in vat_data['methods']:
            summary = vat_data['methods']['VAT'].get('summary', {})
            if method in summary:
                vat_val = format_accuracy(summary[method]['mean'])
        
        if vat_vime_data and 'methods' in vat_vime_data and 'VAT' in vat_vime_data['methods']:
            summary = vat_vime_data['methods']['VAT'].get('summary', {})
            if method in summary:
                vat_vime_val = format_accuracy(summary[method]['mean'])
        
        md_lines.append(f"| {method} | {vime_val} | {vat_val} | {vat_vime_val} |")
    
    # 添加各方法的特有方法
    if vime_data and 'methods' in vime_data and 'VIME' in vime_data['methods']:
        summary = vime_data['methods']['VIME'].get('summary', {})
        if 'VIME' in summary:
            vime_val = format_accuracy(summary['VIME']['mean'])
            md_lines.append(f"| VIME | {vime_val} | - | - |")
    
    if vat_data and 'methods' in vat_data and 'VAT' in vat_data['methods']:
        summary = vat_data['methods']['VAT'].get('summary', {})
        if 'VAT' in summary:
            vat_val = format_accuracy(summary['VAT']['mean'])
            md_lines.append(f"| VAT | - | {vat_val} | - |")
    
    if vat_vime_data and 'methods' in vat_vime_data and 'VAT' in vat_vime_data['methods']:
        summary = vat_vime_data['methods']['VAT'].get('summary', {})
        best_vat_vime = None
        best_val = 0
        for method_name in ['VAT-on-X', 'VAT-on-Layer1', 'VAT-on-Layer2', 
                           'VAT-on-Layer3', 'VAT-on-Concat']:
            if method_name in summary:
                if summary[method_name]['mean'] > best_val:
                    best_val = summary[method_name]['mean']
                    best_vat_vime = method_name
        if best_vat_vime:
            md_lines.append(f"| {best_vat_vime} (最佳) | - | - | {format_accuracy(best_val)} |")
    
    md_lines.append("")
    md_lines.append("## 数据来源")
    md_lines.append("")
    md_lines.append("| 方法 | 结果文件位置 |")
    md_lines.append("|------|-------------|")
    
    if vime_file.exists():
        md_lines.append(f"| VIME | `{vime_file.relative_to(base_dir)}` |")
    if vat_file.exists():
        md_lines.append(f"| VAT | `{vat_file.relative_to(base_dir)}` |")
    if vat_vime_file.exists():
        md_lines.append(f"| VAT_VIME_Feature | `{vat_vime_file.relative_to(base_dir)}` |")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("*结果基于 50 次独立实验的平均值*")
    
    return "\n".join(md_lines)


if __name__ == "__main__":
    md_content = generate_markdown()
    output_file = Path(__file__).parent / "results_summary_celltype.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"结果已保存到: {output_file}")

