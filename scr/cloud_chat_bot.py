# cloud_chat_bot.py（RAG 版）
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import cloud_rag as rag                                   

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-flash-latest"
SYSTEM = ("你是客服助理。只依據提供的【資料】回答，"     # ← 換成 RAG 版
          "資料沒提到的就說「資料裡沒有」。")

history = []

while True:
    user = input("\n你 > ").strip()
    if user in ("exit", "quit", ""):
        break

    hits = rag.retrieve(user, k=2)                         # ← 新增：先檢索
    context = "\n".join(f"[{i+1}] {d}" for i, (d, _) in enumerate(hits))
    prompt = f"【資料】\n{context}\n\n【問題】\n{user}"

    for d, s in hits:                                      # ← 新增：看檢索品質
        print(f"  ↳ {s:.3f}  {d[:30]}…")

    history.append({"role": "user", "parts": [{"text": user}]})    # 歷史存乾淨的

    to_send = list(history)                                # ← 新增：分離送出的版本
    to_send[-1] = {"role": "user", "parts": [{"text": prompt}]}

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=to_send,                              # ← 送 to_send，不是 history
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0.2,                           # ← 降低
            ),
        )
    except Exception as e:
        history.pop()
        print(f"✗ {e}")
        continue

    reply = resp.text
    if not reply:
        history.pop()
        print("✗ 沒拿到內容")
        continue

    print("AI >", reply)
    history.append({"role": "model", "parts": [{"text": reply}]})