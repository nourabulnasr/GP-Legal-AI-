import os
import sys

sys.path.insert(0, "/app")
os.chdir("/app")

from app.routers.chat import chat_assistant, AssistantChatRequest

class U:
    id = 1
    role = "user"

req = AssistantChatRequest(message="hi", history=[])
resp = chat_assistant(req, current_user=U())
print("content:", resp.content[:200])
