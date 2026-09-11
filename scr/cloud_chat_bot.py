# cloud_chat_bot.py（RAG 版）
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import cloud_rag as rag                                   

load_dotenv()

KEY = os.environ.get("GEMINI_API_KEY")          # ← 不要用 os.environ[...]
if not KEY:
    raise RuntimeError("找不到 GEMINI_API_KEY：本機放 .env，雲端放 App settings → Secrets")
client = genai.Client(api_key=KEY)
MODEL = "gemini-flash-latest"
SYSTEM = ("你是客服助理。只依據提供的【資料】回答，"     # ← 換成 RAG 版
          "資料沒提到的就說「資料裡沒有」。")


def ask(user, history, k=2):
    """檢索 + 生成。history 會就地更新，回傳 (reply, hits)。
    失敗時丟例外，history 維持呼叫前的樣子。"""
    hits = rag.retrieve(user, k=k)                        # ← 先檢索
    if not hits and not history:      # ← 第一句就離題才短路；有上下文交給模型判斷
        reply = "資料裡沒有。"        # 措辭跟 SYSTEM 一致，兩條路說法才不會打架
        history.append({"role": "user",  "parts": [{"text": user}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        return reply, hits            # 省掉生成那通 API（檢索那通還是打了）

    context = "\n".join(f"[{i+1}] {d}" for i, (d, _) in enumerate(hits))
    prompt = f"【資料】\n{context}\n\n【問題】\n{user}"

    history.append({"role": "user", "parts": [{"text": user}]})    # 歷史存乾淨的

    to_send = list(history)                               # ← 分離送出的版本
    to_send[-1] = {"role": "user", "parts": [{"text": prompt}]}

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=to_send,                             # ← 送 to_send，不是 history
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0.2,                          # ← 降低
            ),
        )
    except Exception:
        history.pop()
        raise

    reply = resp.text
    if not reply:
        history.pop()
        raise RuntimeError("沒拿到內容")

    history.append({"role": "model", "parts": [{"text": reply}]})
    return reply, hits


if __name__ == "__main__":
    history = []

    while True:
        user = input("\n你 > ").strip()
        if user in ("exit", "quit", ""):
            break

        try:
            reply, hits = ask(user, history)
        except Exception as e:
            print(f"✗ {e}")
            continue

        for d, s in hits:                                  # ← 看檢索品質
            print(f"  ↳ {s:.3f}  {d[:30]}…")

        print("AI >", reply)
