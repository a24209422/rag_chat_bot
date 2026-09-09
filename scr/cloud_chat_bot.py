# cloud_chat_gemini.py  (Gemini 版)
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-flash-latest"
SYSTEM = "你是友善的中文助理，回答簡潔。"

# ── 這個 list 還是「記憶」。角色跟 §1 的 messages 完全相同，只是長相不一樣 ──
history = []

while True:
    user = input("\n你 > ").strip()
    if user in ("exit", "quit", ""):
        break

    history.append({"role": "user", "parts": [{"text": user}]})        # ① 存我說的

    resp = client.models.generate_content(                              # ② 整段一起送
        model=MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,      # ← system 不在 list 裡！
            temperature=0.7,
        ),
    )
    reply = resp.text                                                   # ③ 一個屬性就拿到

    print("AI >", reply)
    history.append({"role": "model", "parts": [{"text": reply}]})       # ④ 存它說的