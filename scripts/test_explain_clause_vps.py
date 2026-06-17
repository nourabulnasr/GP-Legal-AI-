import sys
sys.path.insert(0, "/app")
from app.local_llm import explain_violation, generate, strip_thinking_output

clause = """المادة الثالثة – محظورات العمل
يلتزم العامل بعدم إفشاء أسرار الشركة أو المعلومات السرية التي يطلع عليها أثناء العمل."""

expl = explain_violation(
    rule_id="CLAUSE_REVIEW",
    description="User requested explanation for this clause.",
    matched_text=clause,
    law_articles=[],
    max_new_tokens=200,
    language="ar",
)
print("EXPLANATION:", repr(expl[:500]))
