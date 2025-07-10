#!/usr/bin/env python3
"""
分析不同persona下接受AI建议的比例和决策切换模式
"""

import pandas as pd
import json
import sys
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_simulation_results(filepath: str) -> pd.DataFrame:
    """Load simulation results from CSV file"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} simulation results from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading results: {e}")
        return pd.DataFrame()

def analyze_ai_acceptance_by_persona(df: pd.DataFrame) -> Dict[str, Any]:
    """分析不同persona类型下AI建议的接受率"""
    
    # 清理数据 - 只保留成功的运行
    df_clean = df[df['final_decision_choice'].notna()].copy()
    
    if len(df_clean) == 0:
        print("No valid results to analyze")
        return {}
    
    print(f"\n=== AI建议接受率分析 ===")
    print(f"有效结果数量: {len(df_clean)}")
    
    # 总体接受率
    total_accept = len(df_clean[df_clean['final_decision_choice'] == 'ACCEPT'])
    total_reject = len(df_clean[df_clean['final_decision_choice'] == 'REJECT'])
    overall_acceptance_rate = total_accept / len(df_clean) * 100
    
    print(f"总体AI建议接受率: {overall_acceptance_rate:.1f}% ({total_accept}/{len(df_clean)})")
    print(f"总体AI建议拒绝率: {100-overall_acceptance_rate:.1f}% ({total_reject}/{len(df_clean)})")
    
    analysis = {
        'overall_stats': {
            'total_cases': len(df_clean),
            'accept_count': total_accept,
            'reject_count': total_reject,
            'acceptance_rate': overall_acceptance_rate
        }
    }
    
    # 按专业水平分析
    if 'expertise_level' in df_clean.columns:
        print(f"\n--- 按专业水平分析 ---")
        expertise_analysis = {}
        for expertise in df_clean['expertise_level'].unique():
            subset = df_clean[df_clean['expertise_level'] == expertise]
            accept_count = len(subset[subset['final_decision_choice'] == 'ACCEPT'])
            acceptance_rate = accept_count / len(subset) * 100
            expertise_analysis[expertise] = {
                'total_cases': len(subset),
                'accept_count': accept_count,
                'acceptance_rate': acceptance_rate
            }
            print(f"  {expertise}: {acceptance_rate:.1f}% ({accept_count}/{len(subset)})")
        analysis['expertise_analysis'] = expertise_analysis
    
    # 按AI信任水平分析
    if 'ai_trust_level' in df_clean.columns:
        print(f"\n--- 按AI信任水平分析 ---")
        trust_analysis = {}
        for trust_level in sorted(df_clean['ai_trust_level'].unique()):
            subset = df_clean[df_clean['ai_trust_level'] == trust_level]
            accept_count = len(subset[subset['final_decision_choice'] == 'ACCEPT'])
            acceptance_rate = accept_count / len(subset) * 100
            trust_analysis[trust_level] = {
                'total_cases': len(subset),
                'accept_count': accept_count,
                'acceptance_rate': acceptance_rate
            }
            trust_desc = {-2: "非常不信任", -1: "有些不信任", 0: "中立", 1: "有些信任", 2: "非常信任"}
            print(f"  {trust_level} ({trust_desc.get(trust_level, 'Unknown')}): {acceptance_rate:.1f}% ({accept_count}/{len(subset)})")
        analysis['trust_analysis'] = trust_analysis
    
    # 按认知需求分析 (如果有NCS分数)
    if 'ncs_score' in df_clean.columns and df_clean['ncs_score'].notna().any():
        print(f"\n--- 按认知需求水平分析 ---")
        # 创建认知需求水平分类
        df_clean['ncs_level'] = pd.cut(df_clean['ncs_score'], 
                                     bins=[0, 59, 74, 90], 
                                     labels=['Low', 'Medium', 'High'])
        
        ncs_analysis = {}
        for ncs_level in ['Low', 'Medium', 'High']:
            subset = df_clean[df_clean['ncs_level'] == ncs_level]
            if len(subset) > 0:
                accept_count = len(subset[subset['final_decision_choice'] == 'ACCEPT'])
                acceptance_rate = accept_count / len(subset) * 100
                ncs_analysis[ncs_level] = {
                    'total_cases': len(subset),
                    'accept_count': accept_count,
                    'acceptance_rate': acceptance_rate
                }
                print(f"  {ncs_level} NCS: {acceptance_rate:.1f}% ({accept_count}/{len(subset)})")
        analysis['ncs_analysis'] = ncs_analysis
    
    # 按自我效能感分析 (如果有GSE分数)
    if 'gse_score' in df_clean.columns and df_clean['gse_score'].notna().any():
        print(f"\n--- 按自我效能感水平分析 ---")
        # 创建自我效能感水平分类
        df_clean['gse_level'] = pd.cut(df_clean['gse_score'], 
                                     bins=[0, 29, 34, 40], 
                                     labels=['Low', 'Medium', 'High'])
        
        gse_analysis = {}
        for gse_level in ['Low', 'Medium', 'High']:
            subset = df_clean[df_clean['gse_level'] == gse_level]
            if len(subset) > 0:
                accept_count = len(subset[subset['final_decision_choice'] == 'ACCEPT'])
                acceptance_rate = accept_count / len(subset) * 100
                gse_analysis[gse_level] = {
                    'total_cases': len(subset),
                    'accept_count': accept_count,
                    'acceptance_rate': acceptance_rate
                }
                print(f"  {gse_level} GSE: {acceptance_rate:.1f}% ({accept_count}/{len(subset)})")
        analysis['gse_analysis'] = gse_analysis
    
    return analysis

def analyze_decision_switching(df: pd.DataFrame) -> Dict[str, Any]:
    """分析决策切换模式 - 初始决策与最终决策的对比"""
    
    df_clean = df[df['final_decision_choice'].notna() & df['initial_decision'].notna()].copy()
    
    if len(df_clean) == 0:
        print("No valid decision data to analyze switching patterns")
        return {}
    
    print(f"\n=== 决策切换模式分析 ===")
    
    # 创建决策对比
    df_clean['decision_switch'] = df_clean.apply(lambda row: 
        'Switched' if (
            (row['initial_decision'] == 'Approve' and row['final_decision_choice'] == 'REJECT') or
            (row['initial_decision'] == 'Reject' and row['final_decision_choice'] == 'ACCEPT')
        ) else 'Consistent', axis=1)
    
    switch_count = len(df_clean[df_clean['decision_switch'] == 'Switched'])
    consistent_count = len(df_clean[df_clean['decision_switch'] == 'Consistent'])
    switch_rate = switch_count / len(df_clean) * 100
    
    print(f"决策切换率: {switch_rate:.1f}% ({switch_count}/{len(df_clean)})")
    print(f"决策一致率: {100-switch_rate:.1f}% ({consistent_count}/{len(df_clean)})")
    
    switching_analysis = {
        'overall_switching': {
            'total_cases': len(df_clean),
            'switched_count': switch_count,
            'consistent_count': consistent_count,
            'switch_rate': switch_rate
        }
    }
    
    # 按专业水平分析切换模式
    if 'expertise_level' in df_clean.columns:
        print(f"\n--- 按专业水平的切换模式 ---")
        expertise_switching = {}
        for expertise in df_clean['expertise_level'].unique():
            subset = df_clean[df_clean['expertise_level'] == expertise]
            switched = len(subset[subset['decision_switch'] == 'Switched'])
            switch_rate_exp = switched / len(subset) * 100
            expertise_switching[expertise] = {
                'total_cases': len(subset),
                'switched_count': switched,
                'switch_rate': switch_rate_exp
            }
            print(f"  {expertise}: {switch_rate_exp:.1f}% switched ({switched}/{len(subset)})")
        switching_analysis['expertise_switching'] = expertise_switching
    
    # 按AI信任水平分析切换模式
    if 'ai_trust_level' in df_clean.columns:
        print(f"\n--- 按AI信任水平的切换模式 ---")
        trust_switching = {}
        for trust_level in sorted(df_clean['ai_trust_level'].unique()):
            subset = df_clean[df_clean['ai_trust_level'] == trust_level]
            switched = len(subset[subset['decision_switch'] == 'Switched'])
            switch_rate_trust = switched / len(subset) * 100
            trust_switching[trust_level] = {
                'total_cases': len(subset),
                'switched_count': switched,
                'switch_rate': switch_rate_trust
            }
            trust_desc = {-2: "非常不信任", -1: "有些不信任", 0: "中立", 1: "有些信任", 2: "非常信任"}
            print(f"  {trust_level} ({trust_desc.get(trust_level, 'Unknown')}): {switch_rate_trust:.1f}% switched ({switched}/{len(subset)})")
        switching_analysis['trust_switching'] = trust_switching
    
    return switching_analysis

def generate_summary_report(acceptance_analysis: Dict[str, Any], 
                          switching_analysis: Dict[str, Any]) -> str:
    """生成总结报告"""
    
    report = "\n" + "="*80 + "\n"
    report += "                    PERSONA行为分析报告\n"
    report += "="*80 + "\n"
    
    # 总体统计
    if 'overall_stats' in acceptance_analysis:
        stats = acceptance_analysis['overall_stats']
        report += f"\n📊 总体统计:\n"
        report += f"   • 总案例数: {stats['total_cases']}\n"
        report += f"   • AI建议接受率: {stats['acceptance_rate']:.1f}%\n"
        
    if 'overall_switching' in switching_analysis:
        switch_stats = switching_analysis['overall_switching']
        report += f"   • 决策切换率: {switch_stats['switch_rate']:.1f}%\n"
    
    # 专业水平影响
    if 'expertise_analysis' in acceptance_analysis:
        report += f"\n🎓 专业水平对AI接受度的影响:\n"
        exp_data = acceptance_analysis['expertise_analysis']
        for level in ['beginner', 'intermediate', 'expert']:
            if level in exp_data:
                rate = exp_data[level]['acceptance_rate']
                report += f"   • {level.title()}: {rate:.1f}% 接受率\n"
    
    # AI信任水平影响
    if 'trust_analysis' in acceptance_analysis:
        report += f"\n🤖 AI信任水平对接受度的影响:\n"
        trust_data = acceptance_analysis['trust_analysis']
        trust_labels = {-2: "非常不信任", 0: "中立", 2: "非常信任"}
        for level in [-2, 0, 2]:
            if level in trust_data:
                rate = trust_data[level]['acceptance_rate']
                report += f"   • {trust_labels[level]}: {rate:.1f}% 接受率\n"
    
    # 心理量表影响
    if 'ncs_analysis' in acceptance_analysis:
        report += f"\n🧠 认知需求水平影响:\n"
        ncs_data = acceptance_analysis['ncs_analysis']
        for level in ['Low', 'Medium', 'High']:
            if level in ncs_data:
                rate = ncs_data[level]['acceptance_rate']
                report += f"   • {level} NCS: {rate:.1f}% 接受率\n"
    
    if 'gse_analysis' in acceptance_analysis:
        report += f"\n💪 自我效能感水平影响:\n"
        gse_data = acceptance_analysis['gse_analysis']
        for level in ['Low', 'Medium', 'High']:
            if level in gse_data:
                rate = gse_data[level]['acceptance_rate']
                report += f"   • {level} GSE: {rate:.1f}% 接受率\n"
    
    # 决策切换模式
    if 'trust_switching' in switching_analysis:
        report += f"\n🔄 决策切换模式 (按AI信任水平):\n"
        switch_data = switching_analysis['trust_switching']
        trust_labels = {-2: "非常不信任", 0: "中立", 2: "非常信任"}
        for level in [-2, 0, 2]:
            if level in switch_data:
                rate = switch_data[level]['switch_rate']
                report += f"   • {trust_labels[level]}: {rate:.1f}% 切换率\n"
    
    report += "\n" + "="*80 + "\n"
    return report

def main():
    """Main analysis function"""
    
    # Check if results file path is provided
    if len(sys.argv) < 2:
        print("Usage: python analyze_persona_decisions.py <results_csv_file>")
        print("Looking for latest results file...")
        
        # Try to find the latest results file
        import glob
        results_files = glob.glob("results/simulation_results_*.csv")
        if results_files:
            latest_file = max(results_files)
            print(f"Found latest file: {latest_file}")
        else:
            print("No results files found. Please run a simulation first.")
            return
    else:
        latest_file = sys.argv[1]
    
    # Load and analyze results
    df = load_simulation_results(latest_file)
    if df.empty:
        return
    
    # Perform analyses
    acceptance_analysis = analyze_ai_acceptance_by_persona(df)
    switching_analysis = analyze_decision_switching(df)
    
    # Generate and display report
    report = generate_summary_report(acceptance_analysis, switching_analysis)
    print(report)
    
    # Save detailed analysis to JSON
    detailed_analysis = {
        'ai_acceptance_analysis': acceptance_analysis,
        'decision_switching_analysis': switching_analysis,
        'analysis_metadata': {
            'source_file': latest_file,
            'total_records': len(df),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Convert numpy/pandas types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    detailed_analysis = convert_types(detailed_analysis)
    
    output_file = latest_file.replace('.csv', '_persona_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细分析结果已保存到: {output_file}")

if __name__ == "__main__":
    main()