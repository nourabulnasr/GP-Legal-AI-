import sys
sys.path.insert(0, "/app")
import app.main  # noqa: F401 — triggers load_dotenv like uvicorn
from app.routers.chat import _gemini_api_key, chat_assistant, AssistantChatRequest

print("key_after_main", len(_gemini_api_key()))

class U:
    id = 1

resp = chat_assistant(AssistantChatRequest(message="hi"), current_user=U())
print("content:", resp.content[:120])
