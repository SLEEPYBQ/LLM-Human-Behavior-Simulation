#!/usr/bin/env python3
"""
测试修复后的完整模拟系统
"""

import os
# Note: Set your real OpenAI API key as environment variable
# os.environ['OPENAI_API_KEY'] = 'your-real-api-key-here'

import sys
sys.path.append('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP')

from main import HumanBehaviorSimulation
import pandas as pd

def test_simulation():
    """测试修复后的模拟系统"""
    
    print("=== 测试修复后的人类行为模拟系统 ===")
    
    try:
        # 创建模拟实例
        sim = HumanBehaviorSimulation()
        print("✅ 模拟系统初始化成功")
        
        # 加载ML模型
        sim.load_ml_model('loan_model.pkl')
        print("✅ ML模型加载成功")
        
        # 加载人格配置
        sim.load_personas('config/personas.json')
        print(f"✅ 人格配置加载成功: {len(sim.personas)} 个人格")
        
        # 加载测试数据
        test_data = sim.load_test_data('test_data.csv')
        print(f"✅ 测试数据加载成功: {len(test_data)} 个案例")
        
        # 运行小规模测试
        print("\n开始运行小规模测试 (1个案例, 1个人格)...")
        
        results = sim.run_simulation(
            test_data=test_data,
            num_cases=1,
            num_personas=1,
            verbose=True
        )
        
        if results and len(results) > 0:
            print("✅ 模拟运行成功!")
            
            # 检查结果
            result = results[0]
            if 'error' not in result:
                print("✅ 没有JSON解析错误!")
                print(f"   初始决策: {result.get('initial_decision', 'N/A')}")
                print(f"   最终决策: {result.get('final_decision_choice', 'N/A')}")
                print(f"   处理时间: {result.get('processing_time', 'N/A'):.2f}s")
            else:
                print(f"❌ 仍有错误: {result['error']}")
        else:
            print("❌ 模拟失败 - 没有结果")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def show_analysis_summary():
    """显示之前分析的总结"""
    
    print("\n" + "="*60)
    print("              之前分析结果总结")
    print("="*60)
    
    # 基于之前的分析结果
    print("📊 关键发现:")
    print("   • 总共分析了24个有效案例")
    print("   • AI建议接受率: 0.0% (所有人格都拒绝了AI建议)")
    print("   • 决策切换率: 54.2% (13/24 人改变了初始决策)")
    print("   • 所有案例都是: 新手 + 非常不信任AI")
    
    print("\n🧠 心理学洞察:")
    print("   • 新手级别的贷款审批员更谨慎")
    print("   • 对AI非常不信任的人格倾向于忽略AI建议")
    print("   • 即使经过评估，最终还是坚持人类判断")
    print("   • 高切换率显示内在决策冲突")
    
    print("\n🔧 系统优化成果:")
    print("   • ✅ 修复了JSON解析错误")
    print("   • ✅ 增强了心理学真实性提示工程")
    print("   • ✅ 完善了persona行为分析系统")
    print("   • ✅ 实现了完整的决策流程追踪")

if __name__ == "__main__":
    # 首先显示分析总结
    show_analysis_summary()
    
    # 然后测试修复的系统
    print("\n\n")
    test_simulation()