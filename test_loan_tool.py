#!/usr/bin/env python3
"""
测试修复后的JSON解析问题
"""

import os
import sys
sys.path.append('/Users/zhangbaiqiao/Desktop/Simulate_Human_Behavior/EXP')

from agents.persona_agent import create_loan_analysis_tool
from persona_config import LoanApplication
import json

def test_loan_analysis_tool():
    """测试贷款分析工具是否能正确处理JSON数据"""
    
    # 创建测试知识库
    test_knowledge_base = {
        "cibil_score_ranges": {"excellent": 750, "good": 650, "average": 550},
        "debt_to_income_limit": 40,
        "employment_risk": {"salaried": "low risk", "self_employed": "high risk"}
    }
    
    # 创建工具
    loan_tool = create_loan_analysis_tool(test_knowledge_base)
    
    # 创建测试贷款申请
    test_loan = LoanApplication(
        loan_id=1,
        no_of_dependents=1,
        education="Graduate",
        self_employed="No",
        income_annum=500000,
        loan_amount=1000000,
        loan_term=10,
        cibil_score=720,
        residential_assets_value=2000000,
        commercial_assets_value=0,
        luxury_assets_value=500000,
        bank_asset_value=1000000
    )
    
    # 测试1: JSON字符串输入
    print("=== 测试1: JSON字符串输入 ===")
    loan_json = json.dumps(test_loan.to_dict(), indent=2)
    print("输入数据:")
    print(loan_json[:200] + "...")
    
    try:
        result1 = loan_tool.run(loan_json)
        print("✅ 成功处理JSON字符串输入")
        print("分析结果:")
        print(result1)
    except Exception as e:
        print(f"❌ JSON字符串处理失败: {e}")
    
    # 测试2: 字典输入
    print("\n=== 测试2: 字典输入 ===")
    try:
        result2 = loan_tool.run(test_loan.to_dict())
        print("✅ 成功处理字典输入")
        print("分析结果:")
        print(result2)
    except Exception as e:
        print(f"❌ 字典处理失败: {e}")
    
    # 测试3: 无效输入
    print("\n=== 测试3: 无效输入测试 ===")
    try:
        result3 = loan_tool.run("invalid json data")
        print("处理结果:")
        print(result3)
    except Exception as e:
        print(f"无效输入处理: {e}")

if __name__ == "__main__":
    test_loan_analysis_tool()