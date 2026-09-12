# cloud/chat_bot.py（RAG 版對話）
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import rag                  # 同資料夾的 rag.py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from knowledge import SYSTEM   # noqa: E402 ← 兩邊共用，見 shared/knowledge.py

load_dotenv()

KEY = os.environ.get("GEMINI_API_KEY")          # ← 不要用 os.environ[...]
if not KEY:
    raise RuntimeError("找不到 GEMINI_API_KEY：本機放 .env，雲端放 App settings → Secrets")
client = genai.Client(api_key=KEY)
MODEL = "gemini-flash-latest"


def ask(user, history, k=5):     # 5 個不同職缺。問「有哪些…」要撈得回一串，
                                 # k=2 的時代是 FAQ，一題只有一個正確答案
    """檢索 + 生成。history 會就地更新，回傳 (reply, hits, usage)。
    usage 是這次生成用掉的 token；短路那條路沒打生成 API，所以是 0。
    失敗時丟例外，history 維持呼叫前的樣子。"""
    hits = rag.retrieve(user, k=k)                        # ← 先檢索
    if not hits and not history:      # ← 第一句就離題才短路；有上下文交給模型判斷
        reply = "資料裡沒有。"        # 措辭跟 SYSTEM 一致，兩條路說法才不會打架
        history.append({"role": "user",  "parts": [{"text": user}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        return reply, hits, {"in": 0, "out": 0}   # 省掉生成那通 API（檢索那通還是打了）

    # 真正可靠的完整清單是 UI 的來源列（每一筆 hit 都會顯示，label 以代號開頭），
    # 模型的散文只當摘要看——實測 3B 模型不照指示抄代號，20 筆也只列 16~17 筆。
    if len(hits) > 5:          # 篩選型問句可能撈回 20 個職缺（例如「台北的職缺」），
        context = "\n".join("[%d] %s" % (i + 1, d.label)          # label 本身以代號開頭
                            for i, (d, _) in enumerate(hits))
    else:                      # 少數幾筆就給完整職缺，模型才答得出細節
        context = "\n\n".join("[%d] 代號 %s\n%s" % (i + 1, d.group, d.full)
                              for i, (d, _) in enumerate(hits))
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

    u = resp.usage_metadata                   # 每個欄位都是 Optional[int]，可能是 None
    usage = {
        "in": u.prompt_token_count or 0,
        # thoughts 是使用者看不到的內部草稿，但按輸出計費——不加會嚴重低估
        "out": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
    }

    history.append({"role": "model", "parts": [{"text": reply}]})
    return reply, hits, usage


if __name__ == "__main__":
    history = []

    while True:
        user = input("\n你 > ").strip()
        if user in ("exit", "quit", ""):
            break

        try:
            reply, hits, usage = ask(user, history)
        except Exception as e:
            print(f"✗ {e}")
            continue

        for d, s in hits:                                  # ← 看檢索品質
            print(f"  ↳ {s:.3f}  {d.label[:30]}…")
        print(f"  [token] 入 {usage['in']}  出 {usage['out']}（含思考）")

        print("AI >", reply)
