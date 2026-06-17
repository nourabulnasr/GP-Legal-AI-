from app.routers.chat import _gemini_api_key, _get_gemini_client

print("key_len", len(_gemini_api_key()))
client = _get_gemini_client()
print("client_ok", client is not None)
