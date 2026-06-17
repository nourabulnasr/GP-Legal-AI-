import os, sys
sys.path.insert(0, "/app")
from app.local_llm import load_model, generate, _lora_path
print("lora", _lora_path())
load_model()
out = generate("أنت مساعد قانوني. اذكر فقط: لا يوجد نص قانوني كافٍ في المستند.\n\nالشرح:", max_new_tokens=40, do_sample=False)
print("GEN_OK", out[:120])
