import sys
import traceback
from sympy import symbols, Not, And, Or, Implies, Equivalent

# --- 导入学生函数 ---
try:
    # 这里的 "A4_A0311920U" 必须与你的 .py 文件名和函数名完全一致
    from A4_A0311920U import A4_A0311920U
    print("成功导入 A4_A0311920U 函数。\n")
except ImportError:
    print("="*70)
    print("!! 错误：导入失败 !!")
    print("请确保你的作业文件名是 'A4_A0311920U.py'")
    print("并且这个测试文件 'test_A4_detailed.py' 与你的作业在同一个文件夹中。")
    print("="*70)
    sys.exit()
except Exception as e:
    print(f"导入时发生了一个意料之外的错误: {e}")
    traceback.print_exc()
    sys.exit()

def run_test(query, expected_result, test_name):
    """
    一个辅助函数，用于运行单个测试并打印结果。
    """
    print(f"--- 正在测试: {test_name} ---")
    print(f"    查询 (Query):   {query}")
    print(f"    预期 (Expected): {expected_result}")
    
    try:
        # 调用你的函数
        actual_result = A4_A0311920U(query)
        print(f"    实际 (Actual):   {actual_result}")
        
        # 比较结果
        if actual_result == expected_result:
            print("    结果: PASS ✅\n")
            return True
        else:
            print(f"    结果: FAIL ❌ (预期 '{expected_result}', 但得到 '{actual_result}')\n")
            return False
            
    except Exception as e:
        # 捕获你代码中可能出现的运行时错误
        print(f"    结果: ERROR ❗ (你的函数在运行时崩溃了)\n")
        print("--- 错误详情 ---")
        traceback.print_exc()
        print("------------------\n")
        return False

# --- 主测试程序 ---
if __name__ == "__main__":
    # 定义谜题所需的符号
    A, B, C = symbols('A, B, C')
    
    # -----------------------------------------------------------------
    # 谜题答案的预期结果
    # 经过逻辑推导，唯一可能的模型是：
    # 1. (A=False, B=True, C=True)
    # 2. (A=False, B=True, C=False)
    #
    # 基于这些模型：
    # - Alex (A) 在所有模型中都是 False。 (结论: "False")
    # - Ben (B) 在所有模型中都是 True。   (结论: "True")
    # - Chloe (C) 在一个模型中是 True，另一个中是 False。 (结论: "Not Sure")
    # -----------------------------------------------------------------
    
    print("="*70)
    print("开始详细测试 'A4_A0311920U'...")
    print("="*70)
    print("--- 谜题逻辑分析 (供参考) ---")
    print("正确的知识库 (KB) 应该只满足以下两个模型:")
    print("  1. {A: False, B: True, C: False}")
    print("  2. {A: False, B: True, C: True}")
    print("你的函数返回的结果应该与这个分析一致。")
    print("="*70)

    # 定义所有要运行的测试
    test_cases = [
        # --- 节 1: 基础测试 (作业的核心要求) ---
        (A, "False",    "基础 1: Alex (A) 是骑士吗?"),
        (B, "True",     "基础 2: Ben (B) 是骑士吗?"),
        (C, "Not Sure", "基础 3: Chloe (C) 是骑士吗?"),
        
        # --- 节 2: 健壮性测试 (否定的情况) ---
        (Not(A), "True",     "健壮性 1: Alex (A) 是说谎者吗? (Not A)"),
        (Not(B), "False",    "健壮性 2: Ben (B) 是说谎者吗? (Not B)"),
        (Not(C), "Not Sure", "健壮性 3: Chloe (C) 是说谎者吗? (Not C)"),
        
        # --- 节 3: 复杂逻辑测试 ---
        (And(Not(A), B), "True", "复杂 1: (Not A) AND B? (A是说谎者 且 B是骑士)"),
        (Or(A, B),       "True", "复杂 2: A OR B? (A是骑士 或 B是骑士)"),
        (Equivalent(A, B), "False", "复杂 3: A <=> B? (A 和 B 类型相同)"),
        (Not(Equivalent(A, B)), "True", "复杂 4: A XOR B? (A 和 B 类型不同)"),
        (Implies(B, A),  "False", "复杂 5: B => A? (如果B是骑士, 那么A也是骑士)"),
        (Implies(A, C),  "True", "复杂 6: A => C? (如果A是骑士, 那么C也是骑士)"),
        (Or(A, C),       "Not Sure", "复杂 7: A OR C? (A是骑士 或 C是骑士)"),

        # --- 节 4: 极端情况/完整性检查 ---
        (Or(A, Not(A)), "True", "完整性 1: A OR (Not A)? (重言式, 必须为 True)"),
        (And(B, Not(B)), "False", "完整性 2: B AND (Not B)? (矛盾式, 必须为 False)"),
    ]
    
    total_tests = len(test_cases)
    passed_tests = 0
    
    # 循环运行所有测试
    for query, expected, name in test_cases:
        if run_test(query, expected, name):
            passed_tests += 1
            
    # 打印最终总结
    print("="*70)
    print("--- 测试总结 ---")
    print(f"总共测试: {total_tests}")
    print(f"通过:      {passed_tests}")
    print(f"失败:      {total_tests - passed_tests}")
    print("="*70)

    if passed_tests == total_tests:
        print("🎉 恭喜！所有详细测试全部通过！你的代码看起来非常棒！")
    elif passed_tests > 0:
        print("⚠️ 部分测试失败。请仔细检查失败测试的 (Query), (Expected) 和 (Actual) 部分。")
        print("   - 如果基础测试失败, 可能是你的 KB 构造有误。")
        print("   - 如果只有复杂测试失败, 可能是你的 check_entailment 逻辑有误。")
        print("   - 如果有 'ERROR' 崩溃, 请检查 '错误详情'。")
    else:
        print("❌ 所有测试均失败或崩溃。请从头检查你的代码, 特别是 'check_entailment' 函数。")