import sys

from src.segmentation.expression_parser import build_and_evaluate

tests = ["3+5*2", "(10-4)/3", "7++2"]
for expr in tests:
    # Mô phỏng input của hàm là mảng ký tự
    chars = list(expr)
    print(f"Test: {expr}")
    # build_and_evaluate trả về (expression_str, result_str, error)
    print("Output:", build_and_evaluate(chars))
    print("-" * 30)
