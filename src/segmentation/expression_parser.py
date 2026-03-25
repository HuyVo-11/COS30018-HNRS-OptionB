import ast
import operator


def clean_characters(raw_chars):
    chars = [char.strip() for char in raw_chars if char.strip() != ""]

    operators = {"+", "-", "*", "/", "÷"}
    cleaned = []
    for char in chars:
        if cleaned and char in operators and cleaned[-1] == char:
            continue
        cleaned.append(char)

    return cleaned


def validate_expression(char_list):
    if not char_list:
        return False, "Expression is empty."

    expression = "".join(char_list)

    if char_list[0] in ("+", "*", "/", "÷", ")"):
        return False, f"Expression cannot start with '{char_list[0]}'."
    if char_list[-1] in ("+", "-", "*", "/", "÷", "("):
        return False, f"Expression cannot end with '{char_list[-1]}'."

    binary_ops = {"+", "-", "*", "/", "÷"}
    for index in range(1, len(char_list)):
        if char_list[index] in binary_ops and char_list[index - 1] in binary_ops:
            if not (char_list[index] == "-" and char_list[index - 1] == "("):
                return False, (
                    f"Two consecutive operators '{char_list[index - 1]}' and "
                    f"'{char_list[index]}' at position {index}."
                )

    depth = 0
    for index, char in enumerate(char_list):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth < 0:
            return False, f"Unmatched ')' at position {index}."

    if depth != 0:
        return False, "Unmatched '(' - parentheses are not balanced."

    if "()" in expression:
        return False, "Empty parentheses '()' found."

    return True, "OK"


def safe_eval(expr_str):
    expr_str = expr_str.replace("÷", "/")
    try:
        node = ast.parse(expr_str, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid syntax: {exc}") from exc

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(current):
        if isinstance(current, ast.Constant):
            return current.value
        if isinstance(current, ast.Num):
            return current.n
        if isinstance(current, ast.BinOp):
            left = _eval(current.left)
            right = _eval(current.right)
            if isinstance(current.op, ast.Div) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return operations[type(current.op)](left, right)
        if isinstance(current, ast.UnaryOp):
            return operations[type(current.op)](_eval(current.operand))
        if isinstance(current, ast.Call) and len(current.args) == 1:
            return _eval(current.func) * _eval(current.args[0])
        raise TypeError(f"Unsupported: {type(current)}")

    return _eval(node)


def auto_correct_chars(char_list):
    if not char_list:
        return char_list

    corrected = list(char_list)

    if corrected and corrected[0] == ")":
        corrected[0] = "9"
        print("[AUTO-CORRECT] Fixed ')' at beginning -> '9'")

    if corrected and corrected[-1] == "(":
        open_count = sum(1 for char in corrected[:-1] if char == "(")
        close_count = sum(1 for char in corrected[:-1] if char == ")")
        if open_count <= close_count:
            corrected[-1] = "9"
            print("[AUTO-CORRECT] Fixed '(' at end -> '9'")

    index = 0
    while index < len(corrected) and corrected[index] == ")":
        corrected[index] = "9"
        index += 1
    if index > 1:
        print(f"[AUTO-CORRECT] Fixed {index} leading ')' -> '9'")

    return corrected


def build_and_evaluate(raw_chars):
    cleaned = clean_characters(raw_chars)
    corrected = auto_correct_chars(cleaned)

    expression_str = "".join(corrected)

    if corrected and all(char.isdigit() for char in corrected):
        return expression_str, expression_str, None

    is_valid, message = validate_expression(corrected)
    if not is_valid:
        return expression_str, None, f"Validation failed: {message}"

    try:
        result = safe_eval(expression_str)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return expression_str, str(result), None
    except ZeroDivisionError:
        return expression_str, None, "Khong the chia cho 0!"
    except Exception as exc:
        return expression_str, None, f"Loi tinh toan: {exc}"
