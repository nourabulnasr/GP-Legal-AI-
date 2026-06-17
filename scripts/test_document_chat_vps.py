import sys
sys.path.insert(0, "/app")
from app.routers.chat import _build_document_chat_prompt
from app.local_llm import generate, load_model, strip_thinking_output

ctx = """## نص العقد
عقد عمل محدد المدة بين صاحب عمل (مطعم) والعامل (طباخ).
المدة: سنة واحدة تبدأ من 2025/01/01.
الأجر: 8000 جنيه شهرياً.
ساعات العمل: 10 ساعات يومياً من السبت إلى الخميس.
يحق لصاحب العمل إنهاء العقد دون إنذار في أي وقت.
"""
msg = "اشرح المشاكل التي في العقد"
prompt = _build_document_chat_prompt(ctx, msg)
print("PROMPT_LEN", len(prompt))
load_model()
out = generate(prompt, max_new_tokens=400, do_sample=False, use_lora=False)
print("RAW_LEN", len(out))
print("HAS_THINK_CLOSE", ("`" + "/think" + "`") in out or "</think>" in out)
print("---OUTPUT---")
print(strip_thinking_output(out)[:1200])
